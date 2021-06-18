# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import re
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import srsly
from spacy import util
from spacy.pipeline import Pipe
from spacy.kb import KnowledgeBase
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy_ann.candidate_generator import CandidateGenerator
from spacy_ann.types import KnowledgeBaseCandidate
from spacy_ann.consts import stopwords


@Language.factory(
    "ann_linker",
    assigns=["span._.kb_alias"],
    default_config={
        'threshold': 0.7,
        'no_description_threshold': 0.5,
        'disambiguate': True
    },
    default_score_weights={
        "ents_f": 1.0,
        "ents_p": 0.0,
        "ents_r": 0.0,
        "ents_per_type": None,
    },
)
def make_ann_linker(
    nlp: Language,
    name: str,
    threshold: float,
    no_description_threshold: float,
    disambiguate: bool
):
    return AnnLinker(
        nlp,
        name,
        threshold,
        no_description_threshold,
        disambiguate
    )


class AnnLinker(Pipe):
    """The AnnLinker adds Entity Linking capabilities
    to map NER mentions to KnowledgeBase Aliases or directly to KnowledgeBase Ids
    """

    @classmethod
    def from_nlp(cls, nlp, **cfg):
        """Used in spacy.language.Language when constructing this pipeline.
        Tells spaCy that this pipe requires the nlp object

        nlp (Language): spaCy Language object

        RETURNS (AnnLinker): Initialized AnnLinker.
        """
        return cls(nlp, **cfg)

    def __init__(self, nlp, name="entity_linker", threshold=0.7, no_description_threshold=0.95, disambiguate=True):
        """Initialize the AnnLinker

        nlp (Language): spaCy Language object
        """
        Span.set_extension("alias_candidates", default=[], force=True)
        Span.set_extension("kb_candidates", default=[], force=True)

        self.nlp = nlp
        self.name = name
        self.kb = None
        self.cg = None
        self.threshold = threshold
        self.no_description_threshold = no_description_threshold
        self.disambiguate = disambiguate
        if not self.nlp.vocab.lookups.has_table("mentions_to_alias_cand"):
            self.nlp.vocab.lookups.add_table("mentions_to_alias_cand")

    @property
    def aliases(self) -> List[str]:
        """Get all aliases

        RETURNS (List[str]): List of aliases
        """
        return self.kb.get_alias_strings()

    def _get_span_text(self, span):
        """ transform span text by delete redundency words

        Args:
            span ([type]): entity

        Returns:
            span text
        """
        text = span.text

        if len(text) > 3:
            if span.label_ == 'ingredient':
                exc_words = stopwords
                text = re.sub('|'.join(exc_words), '', text)
            elif span.label_ == 'brand' and '/' in text:
                text = text.split('/')[0]
            # replace GPE
            doc = self.nlp.get_pipe('ner')(self.nlp.make_doc(span.text))
            loc_ents = [ent for ent in doc.ents if ent.label_ == 'GPE']
            for ent in loc_ents:
                text = text.replace(ent.text, '')
        return text.strip() or span.text

    def __call__(self, doc: Doc) -> Doc:
        """Annotate spaCy doc.ents with candidate info.
        If disambiguate is True, use entity vectors and doc context
        to pick the most likely Candidate

        doc (Doc): spaCy Doc

        RETURNS (Doc): spaCy Doc with updated annotations
        """

        self.require_kb()
        self.require_cg()

        mentions = doc.ents
        mention_strings = [self._get_span_text(e) for e in mentions]
        batch_candidates = self.cg(mention_strings)

        for ent, alias_candidates in zip(doc.ents, batch_candidates):
            alias_candidates = [
                ac for ac in alias_candidates if ac.similarity > self.threshold
            ]
            [
                ac
                for ac in alias_candidates
                if ac.similarity > self.no_description_threshold
            ]
            ent._.alias_candidates = alias_candidates

            if len(alias_candidates) == 0:
                continue
            else:
                mentions_table = self.nlp.vocab.lookups.get_table(
                    "mentions_to_alias_cand"
                )
                mentions_table.set(ent.text, alias_candidates[0].alias)

                if self.disambiguate:
                    kb_candidates = self.kb.get_alias_candidates(
                        alias_candidates[0].alias)

                    # create candidate matrix
                    entity_encodings = np.asarray(
                        [c.entity_vector for c in kb_candidates]
                    )
                    candidate_norm = np.linalg.norm(entity_encodings, axis=1)

                    sims = np.dot(entity_encodings, doc.vector.T) / (
                        (candidate_norm * doc.vector_norm) + 1e-8
                    )
                    ent._.kb_candidates = [
                        KnowledgeBaseCandidate(
                            entity=cand.entity_, context_similarity=sim
                        )
                        for cand, sim in zip(kb_candidates, sims)
                    ]

                    # TODO: Add thresholding here
                    best_candidate = kb_candidates[np.argmax(sims)]
                    for t in ent:
                        t.ent_kb_id = best_candidate.entity

        return doc

    def set_kb(self, kb: KnowledgeBase):
        """Set the KnowledgeBase

        kb (KnowledgeBase): spaCy KnowledgeBase
        """
        self.kb = kb

    def set_cg(self, cg: CandidateGenerator):
        """Set the CandidateGenerator

        cg (CandidateGenerator): Initialized CandidateGenerator 
        """
        self.cg = cg

    def require_kb(self):
        """Raise an error if the kb is not set.

        RAISES:
            ValueError: kb required
        """
        if getattr(self, "kb", None) in (None, True, False):
            raise ValueError(f"KnowledgeBase `kb` required for {self.name}")

    def require_cg(self):
        """Raise an error if the cg is not set.

        RAISES:
            ValueError: cg required
        """
        if getattr(self, "cg", None) in (None, True, False):
            raise ValueError(
                f"CandidateGenerator `cg` required for {self.name}")

    def from_disk(self, path: Path, **kwargs):
        """Deserialize saved AnnLinker from disk.

        path (Path): directory to deserialize from

        RETURNS (AnnLinker): Initialized AnnLinker
        """
        path = util.ensure_path(path)

        kb = KnowledgeBase(self.nlp.vocab, 300)
        kb.from_disk(path / "kb")
        self.set_kb(kb)

        cg = CandidateGenerator().from_disk(path)
        self.set_cg(cg)

        cfg = srsly.read_json(path / "cfg")

        self.threshold = cfg.get("threshold", 0.7)
        self.no_description_threshold = cfg.get(
            "no_description_threshold", 0.95)
        self.disambiguate = cfg.get("disambiguate", True)

        return self

    def to_disk(self, path: Path, exclude: Tuple = tuple(), **kwargs):
        """Serialize AnnLinker to disk.

        path (Path): directory to serialize to
        exclude (Tuple, optional): config to exclude. Defaults to tuple().
        """
        path = util.ensure_path(path)
        if not path.exists():
            path.mkdir()

        cfg = {
            "threshold": self.threshold,
            "no_description_threshold": self.no_description_threshold,
            "disambiguate": self.disambiguate,
        }
        srsly.write_json(path / "cfg", cfg)

        self.kb.to_disk(path / "kb")
        self.cg.to_disk(path)
