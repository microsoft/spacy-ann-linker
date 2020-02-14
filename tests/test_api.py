# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path
from starlette.requests import Request
from starlette.responses import RedirectResponse
from starlette.testclient import TestClient
import srsly
from spacy_ann.api.app import app


def test_docs_redirect():
    client = TestClient(app)
    response = client.get('/')    
    assert response.status_code == 200
    assert response.url.split('/')[-1] == "docs"


def test_link(trained_linker):

    @app.middleware("http")
    async def add_nlp_to_state(request: Request, call_next):
        request.state.nlp = trained_linker
        response = await call_next(request)
        return response

    client = TestClient(app)

    example_request = srsly.read_json(
        Path(__file__).parent.parent / "spacy_ann/api/example_request.json"
    )

    res = client.post('/link', json=example_request)
    assert res.status_code == 200

    data = res.json()

    for doc in data['documents']:
        for span in doc['spans']:
            assert 'id' in span

