# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


def test_remote_ann_linker(nlp):
    linker = nlp.create_pipe('remote_ann_linker')

    assert linker.base_url == None
    assert linker.headers == {}

    linker = nlp.create_pipe('remote_ann_linker', {"base_url": "http://linkingurl/link"})
    assert linker.base_url == "http://linkingurl/link"


def test_doc_ents_to_json(nlp):
    ruler = nlp.create_pipe('entity_ruler')
    patterns = [
        {"label": "SKILL", "pattern": alias}
        for alias in ['NLP', 'researched', 'Machine learning']
    ]
    ruler.add_patterns(patterns)
    nlp.add_pipe(ruler)

    doc = nlp("NLP is a highly researched subset of Machine learning.")

    linker = nlp.create_pipe('remote_ann_linker')
    assert linker._ents_to_json(doc.ents) == [
        {
            "text": "NLP",
            "start": 0,
            "end": 3,
            "label": "SKILL"
        },
        {
            "text": "researched",
            "start": 16,
            "end": 26,
            "label": "SKILL"
        },
        {
            "text": "Machine learning",
            "start": 37,
            "end": 53,
            "label": "SKILL"
        }
    ]


def test_remote_ann_linker_disk(nlp):

    linker = nlp.create_pipe('remote_ann_linker', {"base_url": "http://linkingurl/link"})
    assert linker.base_url == "http://linkingurl/link"
    old_base_url = linker.base_url

    linker.to_disk('/tmp/spacy_ann')
    linker = nlp.create_pipe('remote_ann_linker').from_disk('/tmp/spacy_ann')

    assert linker.base_url == old_base_url
