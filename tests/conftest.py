# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path
import os
import subprocess

import pytest
import spacy
from spacy_ann import AnnLinker


@pytest.fixture()
def nlp():
    return spacy.blank('en')


@pytest.fixture()
def trained_linker():
    model_path = Path("examples/tutorial/models/ann_linker")
    if not model_path.exists():
        subprocess.run([
            "spacy_ann", "create_index", "en_core_web_md", "examples/tutorial/data", "examples/tutorial/models"
        ])
    # if not TRAINED_LINKER:
    TRAINED_LINKER = spacy.load("examples/tutorial/models/ann_linker")
    return TRAINED_LINKER
