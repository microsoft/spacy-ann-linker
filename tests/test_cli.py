# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path
import os
import subprocess
import spacy


def test_main():
    process = subprocess.Popen([
        "spacy_ann"
    ], stdout=subprocess.PIPE)

    process.wait(timeout=10)
    
    assert "Available commands" in str(process.stdout.read())

def test_create_index():
    model_path = Path("examples/tutorial/models/ann_linker")
    subprocess.run([
        "spacy_ann", "create_index", "en_core_web_md", "examples/tutorial/data", "examples/tutorial/models"
    ])

    nlp = spacy.load(model_path)
    assert "ann_linker" in nlp.pipe_names
