from timeit import default_timer as timer
from spacy.kb import KnowledgeBase
from spacy.vocab import Vocab
from spacy.util import ensure_path, to_disk, from_disk
from .candidate_generator import CandidateGenerator



class ANNKnowledgeBase(KnowledgeBase):
     def __init__(self, 
        vocab: Vocab,
        entity_vector_length: int = 64,
        k: int = 1,
        m_parameter: int = 100,
        ef_search: int = 200,
        ef_construction: int = 2000,
        n_threads: int = 60
    ):
        """Initialize a CandidateGenerator
        
        k (int): Number of neighbors to query
        m_parameter (int): M parameter value for nmslib hnsw algorithm
        ef_search (int): Set to the maximum recommended value. 
            Improves recall at the expense of longer **inference** time
        ef_construction (int): Set to the maximum recommended value. 
            Improves recall at the expense of longer **indexing** time
        n_threads (int): Number of threads to use when creating the index. 
            Change based on your machine.
        """
        super().__init__(vocab, entity_vector_length)
        self.k = k
        self.m_parameter = m_parameter
        self.ef_search = ef_search
        self.ef_construction = ef_construction
        self.n_threads = n_threads
        self.ann_index = None

    def _initialize(self,
                    aliases: List[str],
                    short_aliases: Set[str],
                    ann_index: FloatIndex,
                    vectorizer: TfidfVectorizer,
                    alias_tfidfs: scipy.sparse.csr_matrix):
        """Used in `fit` and `from_disk` to initialize the CandidateGenerator with computed
        # TF-IDF Vectorizer and ANN Index
        
        aliases (List[str]): Aliases with vectors contained in the ANN Index
        short_aliases (Set[str]): Aliases too short for a TF-IDF representation
        ann_index (FloatIndex): Computed ANN Index of TF-IDF representations for aliases
        vectorizer (TfidfVectorizer): TF-IDF Vectorizer to get vector representation of aliases
        alias_tfidfs (scipy.sparse.csr_matrix): Computed TF-IDF Sparse Vectors for aliases
        """
        self.aliases = aliases
        self.short_aliases = short_aliases
        self.ann_index = ann_index
        self.vectorizer = vectorizer
        self.alias_tfidfs = alias_tfidfs

    def fit_index(self, verbose: bool = True):
        msg = Printer(no_print=verbose)

        kb_aliases = self.get_alias_strings()
        short_aliases = set([a for a in kb_aliases if len(a) < 4])

        # nmslib hyperparameters (very important)
        # guide: https://github.com/nmslib/nmslib/blob/master/python_bindings/parameters.md
        # m_parameter = 100
        # # `C` for Construction. Set to the maximum recommended value
        # # Improves recall at the expense of longer indexing time
        # construction = 2000
        # num_threads = 60  # set based on the machine
        index_params = {
            "M": self.m_parameter,
            "indexThreadQty": self.n_threads,
            "efConstruction": self.ef_construction,
            "post": 0,
        }

        # NOTE: here we are creating the tf-idf vectorizer with float32 type, but we can serialize the
        # resulting vectors using float16, meaning they take up half the memory on disk. Unfortunately
        # we can't use the float16 format to actually run the vectorizer, because of this bug in sparse
        # matrix representations in scipy: https://github.com/scipy/scipy/issues/7408
        
        msg.text(f"Fitting tfidf vectorizer on {len(kb_aliases)} aliases")
        tfidf_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 3), min_df=2, dtype=np.float32
        )
        start_time = timer()
        alias_tfidfs = tfidf_vectorizer.fit_transform(kb_aliases)
        end_time = timer()
        total_time = end_time - start_time
        msg.text(f"Fitting and saving vectorizer took {round(total_time)} seconds")

        msg.text(f"Finding empty (all zeros) tfidf vectors")
        empty_tfidfs_boolean_flags = np.array(alias_tfidfs.sum(axis=1) != 0).reshape(-1,)
        number_of_non_empty_tfidfs = sum(
            empty_tfidfs_boolean_flags == False
        )  # pylint: disable=singleton-comparison
        total_number_of_tfidfs = np.size(alias_tfidfs, 0)

        msg.text(
            f"Deleting {number_of_non_empty_tfidfs}/{total_number_of_tfidfs} aliases because their tfidf is empty"
        )
        # remove empty tfidf vectors, otherwise nmslib will crash
        aliases = [alias for alias, flag in zip(kb_aliases, empty_tfidfs_boolean_flags) if flag]
        alias_tfidfs = alias_tfidfs[empty_tfidfs_boolean_flags]
        assert len(aliases) == np.size(alias_tfidfs, 0)

        msg.text(f"Fitting ann index on {len(aliases)} aliases")
        start_time = timer()
        ann_index = nmslib.init(
            method="hnsw", space="cosinesimil_sparse", data_type=nmslib.DataType.SPARSE_VECTOR
        )
        ann_index.addDataPointBatch(alias_tfidfs)
        ann_index.createIndex(index_params, print_progress=verbose)
        query_time_params = {"efSearch": self.ef_search}
        ann_index.setQueryTimeParams(query_time_params)
        end_time = timer()
        total_time = end_time - start_time
        msg.text(f"Fitting ann index took {round(total_time)} seconds")

        self._initialize(aliases, short_aliases, ann_index, tfidf_vectorizer, alias_tfidfs)
        return self

    def require_ann_index(self):
        """Raise an error if the ann_index is not initialized
        
        RAISES:
            ValueError: ann_index not initialized
        """
        if self.ann_index is None:
            raise ValueError(f"ann_index not initialized. Have you run `cg.train` yet?")

    def get_alias_candidates(self, mention_texts: List[str]):
        self.require_ann_index()

        tfidfs = self.vectorizer.transform(mention_texts)
        start_time = timer()

        # `ann_index.knnQueryBatch` crashes if one of the vectors is all zeros.
        # `nmslib_knn_with_zero_vectors` is a wrapper around `ann_index.knnQueryBatch`
        # that addresses this issue.
        batch_neighbors, batch_distances = self._nmslib_knn_with_zero_vectors(tfidfs, self.k)
        end_time = timer()
        total_time = end_time - start_time

        batch_candidates = []
        for mention, neighbors, distances in zip(
            mention_texts, batch_neighbors, batch_distances
        ):
            if mention in self.short_aliases:
                batch_candidates.append([AliasCandidate(alias=mention, similarity=1.0)])
                continue
            if neighbors is None:
                neighbors = []
            if distances is None:
                distances = []

            alias_candidates = []
            for neighbor_index, distance in zip(neighbors, distances):
                alias = self.aliases[neighbor_index]
                similarity = 1.0 - distance
                alias_candidates.append(AliasCandidate(alias=alias, similarity=similarity))

            batch_candidates.append(alias_candidates)

        return batch_candidates

    def get_candidates(self, alias: str):
        """
        Return candidate entities for an alias. Each candidate defines the entity, the original alias,
        and the prior probability of that alias resolving to that entity.
        If the alias is not known in the KB, and empty list is returned.
        """
        alias_hash = self.vocab.strings[alias]
        if not alias_hash in self._alias_index:
            return []
        alias_index = self._alias_index.get(alias_hash)
        if not alias_index:
            # If we can't find the alias then search for the closest alias 
            # in the kb using the ann_index
            alias_candidates = self.get_alias_candidates([alias])[0]
            nearest_alias = alias_candidates[0].alias
            alias_hash = self.vocab.strings[nearest_alias]
            alias_index = self._alias_index.get(alias_hash)

        alias_entry = self._aliases_table[alias_index]

        return [Candidate(kb=self,
                          entity_hash=self._entries[entry_index].entity_hash,
                          entity_freq=self._entries[entry_index].freq,
                          entity_vector=self._vectors_table[self._entries[entry_index].vector_index],
                          alias_hash=alias_hash,
                          prior_prob=prior_prob)
                for (entry_index, prior_prob) in zip(alias_entry.entry_indices, alias_entry.probs)
                if entry_index != 0]

    def dump(self, path: Path):
        path = ensure_path(path)

        super().dump(path)

        cfg = {
            "k": self.k,
            "m_parameter": self.m_parameter,
            "ef_search": self.ef_search,
            "ef_construction": self.ef_construction,
            "n_threads": self.n_threads
        }
        serializers = {
            "cg_cfg": lambda p: srsly.write_json(p, cfg),
            "aliases": lambda p: srsly.write_json(p.with_suffix(".json"), self.aliases),
            "short_aliases": lambda p: srsly.write_json(p.with_suffix(".json"), self.short_aliases),
            "ann_index": lambda p: self.ann_index.saveIndex(str(p.with_suffix(".bin"))),
            "tfidf_vectorizer": lambda p: joblib.dump(self.vectorizer, p.with_suffix(".joblib")),
            "tfidf_vectors_sparse": lambda p: scipy.sparse.save_npz(
                p.with_suffix(".npz"), self.alias_tfidfs.astype(np.float16)
            ),
        }

        to_disk(path, serializers, {})

    def load_bulk(self, path: Path):
        path = ensure_path(path)
        
        super().load_bulk(path)

        aliases_path = path / "aliases.json"
        short_aliases_path = path / "short_aliases.json"
        ann_index_path = path / "ann_index.bin"
        tfidf_vectorizer_path = path / "tfidf_vectorizer.joblib"
        tfidf_vectors_path = path / "tfidf_vectors_sparse.npz"

        cfg = {}
        deserializers = {"cg_cfg": lambda p: cfg.update(srsly.read_json(p))}
        from_disk(path, deserializers, {})

        self.k = cfg.get("k", 5)
        self.m_parameter = cfg.get("m_parameter", 100)
        self.ef_search = cfg.get("ef_search", 200)
        self.ef_construction = cfg.get("ef_construction", 2000)
        self.n_threads = cfg.get("n_threads", 60)

        aliases = srsly.read_json(aliases_path)
        short_aliases = srsly.read_json(short_aliases_path)
        tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
        alias_tfidfs = scipy.sparse.load_npz(tfidf_vectors_path).astype(np.float32)
        ann_index = nmslib.init(
            method="hnsw", space="cosinesimil_sparse", data_type=nmslib.DataType.SPARSE_VECTOR
        )
        ann_index.addDataPointBatch(alias_tfidfs)
        ann_index.loadIndex(str(ann_index_path))
        query_time_params = {"efSearch": self.ef_search}
        ann_index.setQueryTimeParams(query_time_params)

        self._initialize(aliases, short_aliases, ann_index, tfidf_vectorizer, alias_tfidfs)

        return self