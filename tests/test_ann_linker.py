# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
from spacy_ann.ann_linker import AnnLinker


def test_ann_linker(trained_linker):
    nlp = trained_linker
    ruler = nlp.create_pipe('entity_ruler')
    patterns = [
        {"label": "SKILL", "pattern": alias}
        for alias in ['NLP', 'researched', 'machine learning']
    ]
    ruler.add_patterns(patterns)
    nlp.add_pipe(ruler, before="ann_linker")

    doc = nlp("NLP is a highly researched subset of machine learning.")

    ents = list(doc.ents)
    assert ents[0].kb_id_ == "a3"
    assert ents[1].kb_id_ == "a15"
    assert ents[2].kb_id_ == "a1"
