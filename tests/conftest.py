# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import subprocess
from pathlib import Path

import pytest
import spacy
import srsly


@pytest.fixture
def entities():
    return list(srsly.read_jsonl("examples/tutorial/data/entities.jsonl"))


@pytest.fixture
def aliases():
    return list(srsly.read_jsonl("examples/tutorial/data/aliases.jsonl"))


@pytest.fixture
def nlp():
    return spacy.load("en_core_web_md")


@pytest.fixture()
def trained_linker():
    model_path = Path("examples/tutorial/models/ann_linker")
    if not model_path.exists():
        subprocess.run(
            [
                "spacy_ann",
                "create_index",
                "en_core_web_md",
                "examples/tutorial/data",
                "examples/tutorial/models",
            ]
        )
    # if not TRAINED_LINKER:
    TRAINED_LINKER = spacy.load("examples/tutorial/models/ann_linker")
    return TRAINED_LINKER
