from pathlib import Path
import os
import subprocess
import spacy
from spacy_ann import AnnLinker


def test_create_index():
    model_path = Path("examples/tutorial/models/ann_linker")
    subprocess.run([
        "spacy_ann", "create_index", "en_core_web_md", "examples/tutorial/data", "examples/tutorial/models"
    ])

    nlp = spacy.load(model_path)
    assert "ann_linker" in nlp.pipe_names
