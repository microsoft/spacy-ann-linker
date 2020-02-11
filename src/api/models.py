from typing import List
from pydantic import BaseModel


class LinkingSpan(BaseModel):
    text: str
    start: int
    end: int
    label: str
    id: str = None


class LinkingRecord(BaseModel):
    spans: List[LinkingSpan]
    context: str


class LinkingRequest(BaseModel):
    documents: List[LinkingRecord]


class LinkingResponse(BaseModel):
    documents: List[LinkingRecord]
