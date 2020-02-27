# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from collections import defaultdict
import multiprocessing
import os
from pathlib import Path
from typing import Callable

import spacy
import typer
from wasabi import Printer
from spacy_ann.api.constants import NO_API_KEY


def serve(model: str,
        #   api_key: str = typer.Option(NO_API_KEY, prompt=True, hide_input=True, show_default=True, confirmation_prompt=True),
          host: str = "127.0.0.1",
          port: int = 8080,
          use_gunicorn: bool = False,
          n_workers: int = (multiprocessing.cpu_count() * 2) + 1):
    
    from fastapi import FastAPI, Depends
    from fastapi.security import APIKeyHeader
    from starlette.exceptions import HTTPException
    from starlette.requests import Request
    from starlette.status import HTTP_401_UNAUTHORIZED
    import uvicorn
    from spacy_ann.api.app import app


    nlp = spacy.load(model)


    @app.middleware("http")
    async def update_request_state(request: Request, call_next):
        request.state.nlp = nlp
        # request.state.api_key = api_key
        response = await call_next(request)
        return response
    
    if use_gunicorn:
        import gunicorn
        from gunicorn.app.wsgiapp import WSGIApplication
        from uvicorn.workers import UvicornWorker

        class FastAPIApplication(WSGIApplication):

            def __init__(self, app, options=None):
                self.options = options or {}
                self.application = app
                super().__init__()

            def load_config(self):
                config = {key: value for key, value in self.options.items()
                        if key in self.cfg.settings and value is not None}
                for key, value in config.items():
                    self.cfg.set(key.lower(), value)

            def load(self):
                return self.application
        options = {
            'bind': f"{host}:{port}",
            'workers': n_workers,
            'worker_class': 'uvicorn.workers.UvicornWorker'
        }
        FastAPIApplication(app, options).run()
    else:
        uvicorn.run(app, host=host, port=port)



