# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


def test_ann_linker(trained_linker):
    nlp = trained_linker
    ann_linker = nlp.get_pipe('ann_linker')
    ann_linker.enable_context_similarity = True
    ruler = nlp.add_pipe("entity_ruler", before="ann_linker")
    patterns = [
        {"label": "SKILL", "pattern": alias}
        for alias in ["NLP", "researched", "machine learning"]
    ]
    ruler.add_patterns(patterns)

    doc = nlp("NLP is a highly researched subset of machine learning.")

    ents = list(doc.ents)
    assert ents[0].kb_id_ == "a3"
    assert ents[1].kb_id_ == "a15"
    assert ents[2].kb_id_ == "a1"
