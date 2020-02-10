# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""spaCy ANN Linker, a pipeline component for generating spaCy KnowledgeBase Alias Candidates for Entity Linking."""


__version__ = '0.0.3'

from .ann_linker import ApproxNearestNeighborsLinker

from spacy.kb import KnowledgeBase
from spacy.tokens import Span
from spacy.util import registry


@registry.kb.register("get_candidates")
def get_candidates(kb: KnowledgeBase, ent: Span):
    alias = ent._.kb_alias if ent._.kb_alias else ent.text
    return kb.get_candidates(alias)
