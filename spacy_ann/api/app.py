# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from fastapi import Body, Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader
from starlette.requests import Request
from starlette.responses import RedirectResponse
from starlette.status import HTTP_401_UNAUTHORIZED
import spacy
import srsly
import uvicorn

from spacy_ann import __version__
from spacy_ann.api.constants import NO_API_KEY
from spacy_ann.api.types import LinkingRequest, LinkingResponse, LinkingRecord


load_dotenv(find_dotenv())
openapi_prefix = os.getenv("CLUSTER_ROUTE_PREFIX", "").rstrip("/")


app = FastAPI(
    title = "spacy-ann-linker",
    version = __version__,
    description = "Remote Entity Linking with Approximate Nearest Neighbors index lookup for Aliases",
    openapi_prefix = openapi_prefix
)
example_request = srsly.read_json(Path(__file__).parent / 'example_request.json')


security = APIKeyHeader(name="api-key")


@app.get("/", include_in_schema=False)
def docs_redirect():
    return RedirectResponse(f"{openapi_prefix}/docs")


@app.post("/link", response_model=LinkingResponse)
async def link(request: Request,
            #    api_key = Depends(security),
               similarity_threshold: float = 0.65,
               body: LinkingRequest = Body(..., example=example_request)):
    """Link batch of Spans to their canonical KnowledgeBase Id."""

    try:
        nlp = request.state.nlp
        # app_api_key = request.state.api_key
    except AttributeError as e:
        error_msg = "`nlp` does not exist in the request state." \
        "nlp is set using middleware defined in the `spacy_ann serve` command." \
        "Are you running this app outside of the `spacy_ann serve` command?"
        raise HTTPException(status_code=501, detail=error_msg)

    # if app_api_key != NO_API_KEY and api_key != app_api_key:
    #     raise HTTPException(
    #         status_code=HTTP_401_UNAUTHORIZED, detail="Unauthorized auth api-key passed in header"
    #     )

    res = LinkingResponse(documents=[])
    for doc in body.documents:
        spacy_doc = nlp.make_doc(doc.context)
        spans = [
            spacy_doc.char_span(s.start, s.end, label=s.label)
            for s in doc.spans
        ]
        spacy_doc.ents = [s for s in spans if s]
        ann_linker = nlp.get_pipe('ann_linker')
        ann_linker.cg.threshold = similarity_threshold
        spacy_doc = ann_linker(spacy_doc)

        for i, ent in enumerate(spacy_doc.ents):
            doc.spans[i].id = ent.kb_id_
            doc.spans[i].alias_candidates = ent._.alias_candidates
            doc.spans[i].kb_candidates = ent._.kb_candidates


        print(doc)
        res.documents.append(
            LinkingRecord(spans=doc.spans, context=doc.context)
        )
            
    return res
