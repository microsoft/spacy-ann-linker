# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from fastapi import Body, FastAPI, HTTPException
from starlette.requests import Request
from starlette.responses import RedirectResponse
import spacy
import srsly
import uvicorn

from spacy_ann import __version__
from spacy_ann.api.models import LinkingRequest, LinkingResponse, LinkingRecord


load_dotenv(find_dotenv())
openapi_prefix = os.getenv("CLUSTER_ROUTE_PREFIX", "").rstrip("/")


app = FastAPI(
    title = "spacy-ann-linker",
    version = __version__,
    description = "Remote Entity Linking with Approximate Nearest Neighbors index lookup for Aliases",
    openapi_prefix = openapi_prefix
)
example_request = srsly.read_json(Path(__file__).parent / 'example_request.json')


@app.get("/", include_in_schema=False)
def docs_redirect():
    return RedirectResponse(f"{openapi_prefix}/docs")


@app.post("/link", response_model=LinkingResponse)
async def link(request: Request, body: LinkingRequest = Body(..., example=example_request)):
    """Link batch of Spans to their canonical KnowledgeBase Id."""

    try:
        nlp = request.state.nlp
    except AttributeError as e:
        error_msg = "`nlp` does not exist in the request state." \
        "nlp is set using middleware defined in the `spacy_ann serve` command." \
        "Are you running this app outside of the `spacy_ann serve` command?"
        raise HTTPException(status_code=501, detail=error_msg)

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
