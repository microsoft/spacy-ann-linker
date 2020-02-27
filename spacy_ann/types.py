# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pydantic import BaseModel
from pydantic.dataclasses import dataclass


class AliasCandidate(BaseModel):
    """A data class representing a candidate alias
    that a NER mention may be linked to.
    """
    alias: str
    similarity: float


class KnowledgeBaseCandidate(BaseModel):
    entity: str
    context_similarity: float
