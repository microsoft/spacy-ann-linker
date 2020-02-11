# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, Body
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
import spacy
import uvicorn

from models import LinkingRequest, LinkingResponse, LinkingRecord


load_dotenv(find_dotenv())
prefix = os.getenv("CLUSTER_ROUTE_PREFIX")
if not prefix:
    prefix = ""
prefix = prefix.rstrip("/")


app = FastAPI(
    title="spacy-ann-linker",
    version="1.0",
    description="Entity Linking with Approximate Nearest Neighbors index lookup for Aliases",
    openapi_prefix=prefix,
)

example_request = list(srsly.read_json('./example_request.json'))

nlp = spacy.load("../tutorial/models/ann_linker")


@app.get("/", include_in_schema=False)
def docs_redirect():
    return RedirectResponse(f"{prefix}/docs")


@app.post("/link", response_model=LinkingResponse)
async def link(body: LinkingRequest = Body(..., example=example_request)):
    """Link batch of Spans to their canonical KnowledgeBase Id."""

    res = LinkingResponse(documents=[])
    for doc in body.documents:
        spacy_doc = nlp.make_doc(doc.context)
        spans = [
            spacy_doc.char_span(s.start, s.end, label=s.label)
            for s in doc.spans
        ]
        spacy_doc.ents = [s for s in spans if s]
        spacy_doc = nlp.get_pipe('ann_linker')(spacy_doc)

        for i, ent in enumerate(spacy_doc.ents):
            doc.spans[i].id = ent.kb_id_

        res.documents.append(
            LinkingRecord(spans=doc.spans, context=doc.context)
        )
            
    return res
