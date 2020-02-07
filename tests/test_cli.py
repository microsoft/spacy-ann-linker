import os
import subprocess


def test_create_index():
    subprocess.run([
        "spacy_ann", "create_index", "en_core_web_md", "examples/tutorial/data", "examples/tutorial/models"
    ])