# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from collections import defaultdict
import os
import json
import numpy as np
import spacy
from spacy.errors import Errors
from spacy.compat import basestring_
from spacy.language import component
from spacy.kb import KnowledgeBase
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
class ApproxNearestNeighborsLinker(object):
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
        self.k = cfg.get("k_neighbors", 5)

    def __call__(self, doc: Doc):
        """Resolve the ent_id_ attribute of an ent span to the skills data store."""
        self.require_kb()
        self.require_cg()

        mentions = doc.ents
        mention_strings = [x.text for x in mentions]
        batch_candidates = self.cg(mention_strings, self.k)

        for ent, candidates in zip(doc.ents, batch_candidates):
            ent._.kb_alias = batch_candidates[0]

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
        cfg = {}
        deserializers = {
            "cfg": lambda p: cfg.update(srsly.read_json(p)),
        }
        util.from_disk(path, deserializers, {})

        kb = KnowledgeBase(self.nlp.vocab, 300)
        kb.load_bulk(path / "kb")
        self.set_kb(kb)

        cg = CandidateGenerator(self.nlp, kb).from_disk(path)
        self.set_cg(cg)

        return self

    def to_disk(self, path, exclude=tuple(), **kwargs):
        """Save data to disk"""
        path = util.ensure_path(path)
        if not path.exists():
            path.mkdir()
        srsly.write_json(path / "cfg", {"k_neighbors": self.k})
        self.kb.dump(path / "kb")
        self.cg.to_disk(path)
