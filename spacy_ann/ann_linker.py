# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from collections import defaultdict
import os
import json
from typing import List
import numpy as np
import spacy
from spacy.errors import Errors
from spacy.compat import basestring_
from spacy.language import component
from spacy.kb import Candidate, KnowledgeBase
from spacy.tokens import Doc, Span
from spacy import util
import srsly
from bin.wiki_entity_linking.train_descriptions import EntityEncoder
from spacy_ann.candidate_generator import CandidateGenerator


@component(
    "ann_linker",
    requires=["doc.ents", "doc.sents", "token.ent_iob", "token.ent_type"],
    assigns=["span._.kb_alias"],
)
class ApproxNearestNeighborsLinker:
    """The ApproxNearestNeighborsLinker adds Entity Linking capabilities
    to map NER mentions to KnowledgeBase Aliases
    """

    @classmethod
    def from_nlp(cls, nlp, **cfg):
        return cls(nlp, **cfg)

    def __init__(self, nlp, **cfg):
        """Initialize the ApproxNearestNeighborsLinker."""
        Span.set_extension("kb_alias", default="", force=True)

        self.nlp = nlp
        self.kb = None
        self.cg = None
        self.disambiguate = cfg.get("disambiguate", True)

    @property
    def aliases(self) -> List[str]:
        """Return List of aliases in KB"""
        return self.kb.get_alias_strings()

    def __call__(self, doc: Doc) -> Doc:
        """Resolve the ent_id_ attribute of an ent span to the skills data store."""
        self.require_kb()
        self.require_cg()

        mentions = doc.ents
        mention_strings = [e.text for e in mentions]
        batch_candidates = self.cg(mention_strings)
        
        for ent, alias_candidates in zip(doc.ents, batch_candidates):
            if len(alias_candidates) == 0:
                continue
            else:
                if self.disambiguate:
                    kb_candidates = self.kb.get_candidates(alias_candidates[0].alias)

                    # create candidate matrix
                    entity_encodings = np.asarray([c.entity_vector for c in kb_candidates])
                    candidate_norm = np.linalg.norm(entity_encodings, axis=1)

                    sims = np.dot(entity_encodings, doc.vector.T) / (
                        candidate_norm * doc.vector_norm
                    )

                    # TODO: Add thresholding here
                    likely = kb_candidates[np.argmax(sims)]
                    for t in ent:
                        t.ent_kb_id = likely.entity
                else:
                    # Set aliases for a later pipeline component
                    ent._.kb_alias = alias_candidates[0]

        return doc

    def set_kb(self, kb: KnowledgeBase):
        self.kb = kb

    def set_cg(self, cg: CandidateGenerator):
        self.cg = cg

    def require_kb(self):
        # Raise an error if the knowledge base is not initialized.
        if getattr(self, "kb", None) in (None, True, False):
            raise ValueError(f"Knowledge Base Required for {self.name}")

    def require_cg(self):
        # Raise an error if the knowledge base is not initialized.
        if getattr(self, "cg", None) in (None, True, False):
            raise ValueError(f"Candidate Generator Required for {self.name}")

    def from_disk(self, path, **kwargs):
        """Load data from disk"""
        path = util.ensure_path(path)

        kb = KnowledgeBase(self.nlp.vocab, 300)
        kb.load_bulk(path / "kb")
        self.set_kb(kb)

        cg = CandidateGenerator().from_disk(path)
        self.set_cg(cg)

        return self

    def to_disk(self, path, exclude=tuple(), **kwargs):
        """Save data to disk"""
        path = util.ensure_path(path)
        if not path.exists():
            path.mkdir()
        self.kb.dump(path / "kb")
        self.cg.to_disk(path)
