# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Adapted from https://github.com/allenai/scispacy/blob/master/scispacy/candidate_generation.py
# for use with spaCy KnowledgeBase

from typing import List, Dict, Tuple
import json
import datetime
from collections import defaultdict
from pathlib import Path

from timeit import default_timer as timer
import scipy
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from spacy.kb import Candidate, KnowledgeBase
from spacy.tokens import Doc, Span
from spacy.util import ensure_path, to_disk, from_disk
import srsly
import nmslib
from nmslib.dist import FloatIndex
from spacy_ann.models import AliasCandidate


class CandidateGenerator:
    def __init__(
        self, nlp, kb, ef_search=200, k_neighbors=5, knn_similarity_threshold=0.90, verbose=False
    ):
        self.nlp = nlp
        self.kb = kb
        self.ef_search = ef_search
        self.k = k_neighbors
        self.threshold = knn_similarity_threshold
        self.verbose = verbose

    @property
    def initialized(self):
        return self.ann_index != None

    def _initialize(self, aliases, ann_index, vectorizer, alias_tfidfs):
        self.aliases = aliases
        self.ann_index = ann_index
        self.vectorizer = vectorizer
        self.alias_tfidfs = alias_tfidfs

    @classmethod
    def create_with_defaults(cls, nlp, kb) -> Tuple[List[str], TfidfVectorizer, FloatIndex]:
        """
        Build tfidf vectorizer and ann index.
        Warning: Running this function can take a lot of memory
        Parameters
        ----------
        path: str, required.
            The path where the various model pieces will be saved and loaded from
        kb: KnowledgeBase, required.
            The spaCy KnowledgeBase instance with alias and entity info
        """
        cg = CandidateGenerator(nlp, kb)
        kb_aliases = kb.get_alias_strings()

        # nmslib hyperparameters (very important)
        # guide: https://github.com/nmslib/nmslib/blob/master/python_bindings/parameters.md
        # Default values resulted in very low recall.

        # set to the maximum recommended value. Improves recall at the expense of longer indexing time.
        # TODO: This variable name is so hot because I don't actually know what this parameter does.
        m_parameter = 100
        # `C` for Construction. Set to the maximum recommended value
        # Improves recall at the expense of longer indexing time
        construction = 2000
        num_threads = 60  # set based on the machine
        index_params = {
            "M": m_parameter,
            "indexThreadQty": num_threads,
            "efConstruction": construction,
            "post": 0,
        }

        # NOTE: here we are creating the tf-idf vectorizer with float32 type, but we can serialize the
        # resulting vectors using float16, meaning they take up half the memory on disk. Unfortunately
        # we can't use the float16 format to actually run the vectorizer, because of this bug in sparse
        # matrix representations in scipy: https://github.com/scipy/scipy/issues/7408
        print(f"Fitting tfidf vectorizer on {len(kb_aliases)} aliases")
        tfidf_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 3), min_df=2, dtype=np.float32
        )
        start_time = datetime.datetime.now()
        alias_tfidfs = tfidf_vectorizer.fit_transform(kb_aliases)
        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        print(f"Fitting and saving vectorizer took {total_time.total_seconds()} seconds")

        print(f"Finding empty (all zeros) tfidf vectors")
        empty_tfidfs_boolean_flags = np.array(alias_tfidfs.sum(axis=1) != 0).reshape(-1,)
        number_of_non_empty_tfidfs = sum(
            empty_tfidfs_boolean_flags == False
        )  # pylint: disable=singleton-comparison
        total_number_of_tfidfs = np.size(alias_tfidfs, 0)

        print(
            f"Deleting {number_of_non_empty_tfidfs}/{total_number_of_tfidfs} aliases because their tfidf is empty"
        )
        # remove empty tfidf vectors, otherwise nmslib will crash
        aliases = [alias for alias, flag in zip(kb_aliases, empty_tfidfs_boolean_flags) if flag]
        alias_tfidfs = alias_tfidfs[empty_tfidfs_boolean_flags]
        assert len(aliases) == np.size(alias_tfidfs, 0)

        print(f"Fitting ann index on {len(aliases)} aliases")
        start_time = datetime.datetime.now()
        ann_index = nmslib.init(
            method="hnsw", space="cosinesimil_sparse", data_type=nmslib.DataType.SPARSE_VECTOR
        )
        ann_index.addDataPointBatch(alias_tfidfs)
        ann_index.createIndex(index_params, print_progress=True)
        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        print(f"Fitting ann index took {elapsed_time.total_seconds()} seconds")

        cg._initialize(aliases, ann_index, tfidf_vectorizer, alias_tfidfs)
        return cg

    def nmslib_knn_with_zero_vectors(
        self, vectors: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        ann_index.knnQueryBatch crashes if any of the vectors is all zeros.
        This function is a wrapper around `ann_index.knnQueryBatch` that solves this problem. It works as follows:
        - remove empty vectors from `vectors`.
        - call `ann_index.knnQueryBatch` with the non-empty vectors only. This returns `neighbors`,
        a list of list of neighbors. `len(neighbors)` equals the length of the non-empty vectors.
        - extend the list `neighbors` with `None`s in place of empty vectors.
        - return the extended list of neighbors and distances.
        """
        empty_vectors_boolean_flags = np.array(vectors.sum(axis=1) != 0).reshape(-1,)
        empty_vectors_count = vectors.shape[0] - sum(empty_vectors_boolean_flags)
        if self.verbose:
            print(f"Number of empty vectors: {empty_vectors_count}")

        # init extended_neighbors with a list of Nones
        extended_neighbors = np.empty((len(empty_vectors_boolean_flags),), dtype=object)
        extended_distances = np.empty((len(empty_vectors_boolean_flags),), dtype=object)

        if vectors.shape[0] - empty_vectors_count == 0:
            return extended_neighbors, extended_distances

        # remove empty vectors before calling `ann_index.knnQueryBatch`
        vectors = vectors[empty_vectors_boolean_flags]

        # call `knnQueryBatch` to get neighbors
        original_neighbours = self.ann_index.knnQueryBatch(vectors, k=k)

        neighbors, distances = zip(*[(x[0].tolist(), x[1].tolist()) for x in original_neighbours])
        neighbors = list(neighbors)
        distances = list(distances)

        # neighbors need to be converted to an np.array of objects instead of ndarray of dimensions len(vectors)xk
        # Solution: add a row to `neighbors` with any length other than k. This way, calling np.array(neighbors)
        # returns an np.array of objects
        neighbors.append([])
        distances.append([])
        # interleave `neighbors` and Nones in `extended_neighbors`
        extended_neighbors[empty_vectors_boolean_flags] = np.array(neighbors)[:-1]
        extended_distances[empty_vectors_boolean_flags] = np.array(distances)[:-1]

        return extended_neighbors, extended_distances
    
    def __call__(
        self, mention_texts: List[str], k: int
    ) -> List[List[AliasCandidate]]:
        if not self.initialized:
            raise Exception(
                "Not initialized. Run create_tfidf_ann_index or load a pretrained ann_index using from_disk"
            )
        if self.verbose:
            print(f"Generating candidates for {len(mention_texts)} mentions")

        # tfidf vectorizer crashes on an empty array, so we return early here
        if mention_texts == []:
            return []

        tfidfs = self.vectorizer.transform(mention_texts)
        start_time = timer()

        # `ann_index.knnQueryBatch` crashes if one of the vectors is all zeros.
        # `nmslib_knn_with_zero_vectors` is a wrapper around `ann_index.knnQueryBatch` that addresses this issue.
        batch_neighbors, batch_distances = self.nmslib_knn_with_zero_vectors(tfidfs, k)
        end_time = timer()
        total_time = end_time - start_time
        if self.verbose:
            print(f"Finding neighbors took {total_time} seconds")

        short_alias_strings = set([a for a in self.kb.get_alias_strings() if len(a) < 3])

        batch_candidates = []
        for mention, neighbors, distances in zip(
            mention_texts, batch_neighbors, batch_distances
        ):
            if mention in short_alias_strings:
                batch_candidates.append(self.kb.get_candidates(mention))
                continue
            if neighbors is None:
                neighbors = []
            if distances is None:
                distances = []

            alias_candidates = []
            for neighbor_index, distance in zip(neighbors, distances):
                alias = self.aliases[neighbor_index]
                similarity = 1.0 - distance
                if similarity > self.threshold:
                    alias_candidates.append(AliasCandidate(alias, similarity))

            batch_candidates.append(alias_candidates)

        return batch_candidates

    def batch_candidates(
        self, batch_mention_texts: List[List[str]], k: int
    ) -> List[List[List[Candidate]]]:
        if not self.initialized:
            raise Exception(
                "Not initialized. Run create_tfidf_ann_index or load a pretrained ann_index using from_disk"
            )
        if self.verbose:
            print(f"Generating candidates for {len(batch_mention_texts)} mentions")

        batch_res = []
        offsets = []

        mention_texts = []
        for doc_mentions in batch_mention_texts:
            start = offsets[-1][1] if offsets else 0
            offsets.append((start, start + len(doc_mentions)))
            mention_texts += doc_mentions

        # tfidf vectorizer crashes on an empty array, so we return early here
        if mention_texts == []:
            return []

        tfidfs = self.vectorizer.transform(mention_texts)
        start_time = timer()

        # `ann_index.knnQueryBatch` crashes if one of the vectors is all zeros.
        # `nmslib_knn_with_zero_vectors` is a wrapper around `ann_index.knnQueryBatch` that addresses this issue.
        batch_neighbors, batch_distances = self.nmslib_knn_with_zero_vectors(tfidfs, k)
        # print(len(mention_texts), len(batch_neighbors), len(batch_distances))
        end_time = timer()
        total_time = end_time - start_time
        if self.verbose:
            print(f"Finding neighbors took {total_time} seconds")

        short_alias_strings = set([a for a in self.kb.get_alias_strings() if len(a) < 3])

        batch_candidates_by_doc = []
        for start, end in offsets:
            batch_candidates = []
            for mention, neighbors, distances in zip(
                mention_texts[start:end], batch_neighbors[start:end], batch_distances[start:end]
            ):

                if mention in short_alias_strings:
                    batch_candidates.append(self.kb.get_candidates(mention))
                    continue
                if neighbors is None:
                    neighbors = []
                if distances is None:
                    distances = []

                alias_candidates = []
                for neighbor_index, distance in zip(neighbors, distances):
                    alias = self.aliases[neighbor_index]
                    similarity = 1.0 - distance
                    if similarity > self.threshold:
                        alias_candidates.append(AliasCandidate(alias, similarity))

                if not alias_candidates:
                    candidates = []
                else:
                    candidates = []
                    for ac in alias_candidates:
                        candidates += self.kb.get_candidates(ac.alias)

                batch_candidates.append(candidates)
            batch_candidates_by_doc.append(batch_candidates)

        return batch_candidates_by_doc

    def from_disk(self, path, **kwargs):
        """Load data from disk"""

        aliases_path = f"{path}/aliases.json"
        ann_index_path = f"{path}/ann_index.bin"
        tfidf_vectorizer_path = f"{path}/tfidf_vectorizer.joblib"
        tfidf_vectors_path = f"{path}/tfidf_vectors_sparse.npz"

        cfg = {}
        deserializers = {"cg_cfg": lambda p: cfg.update(srsly.read_json(p))}

        from_disk(path, deserializers, {})
        self.ef_search = cfg.get("ef_search", 200)
        self.k_neighbors = cfg.get("k_neighbors", 5)
        self.threshold = cfg.get("knn_similarity_threshold", 0.75)
        self.verbose = cfg.get("verbose", False)

        aliases = srsly.read_json(aliases_path)
        tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
        alias_tfidfs = scipy.sparse.load_npz(tfidf_vectors_path).astype(np.float32)
        ann_index = nmslib.init(
            method="hnsw", space="cosinesimil_sparse", data_type=nmslib.DataType.SPARSE_VECTOR
        )
        ann_index.addDataPointBatch(alias_tfidfs)
        ann_index.loadIndex(str(ann_index_path))
        query_time_params = {"efSearch": self.ef_search}
        ann_index.setQueryTimeParams(query_time_params)

        self._initialize(aliases, ann_index, tfidf_vectorizer, alias_tfidfs)

        return self

    def to_disk(self, path, **kwargs):
        """Save data to disk"""
        cfg = {
            "ef_search": self.ef_search,
            "k_neighbors": self.k,
            "knn_similarity_threshold": self.threshold,
            "verbose": self.verbose,
        }
        serializers = {
            "cg_cfg": lambda p: srsly.write_json(p, cfg),
            "aliases": lambda p: srsly.write_json(p.with_suffix(".json"), self.aliases),
            "ann_index": lambda p: self.ann_index.saveIndex(str(p.with_suffix(".bin"))),
            "tfidf_vectorizer": lambda p: joblib.dump(self.vectorizer, p.with_suffix(".joblib")),
            "tfidf_vectors_sparse": lambda p: scipy.sparse.save_npz(
                p.with_suffix(".npz"), self.alias_tfidfs.astype(np.float16)
            ),
        }

        to_disk(path, serializers, {})
