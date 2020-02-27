# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import List, Optional
from pydantic import BaseModel
# from spacy.kb import Candidate
from spacy_ann.types import AliasCandidate, KnowledgeBaseCandidate

# class ApiAliasCandidate(BaseModel):
#     alias: str
#     similarity: float


# class ApiKBCandidate(BaseModel):
#     entity: str
#     context_similarity: float


class LinkingSpan(BaseModel):
    text: str
    start: int
    end: int
    label: str
    id: Optional[str] = None
    alias_candidates: Optional[List[AliasCandidate]] = None
    kb_candidates: Optional[List[KnowledgeBaseCandidate]] = None


class LinkingRecord(BaseModel):
    spans: List[LinkingSpan]
    context: str


class LinkingRequest(BaseModel):
    documents: List[LinkingRecord]


class LinkingResponse(BaseModel):
    documents: List[LinkingRecord]
