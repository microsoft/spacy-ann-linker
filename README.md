<!-- <p align="center">
  <a href="https://microsoft.github.io/spacy-ann-linker"><img src="https://typer.tiangolo.com/img/logo-margin/logo-margin-vectoar.svg" alt="spaCy ANN Linker"></a>
</p> -->
<p>
    <em>spaCy ANN Linker, a pipeline component for generating spaCy KnowledgeBase Alias Candidates for Entity Linking based on an Approximate Nearest Neighbors (ANN) index computed on the Character N-Gram TF-IDF representation of all aliases in your KnowledgeBase.</em>
</p>
<p align="center">
<a href="https://dev.azure.com/kakh/spacy-ann-linker/_apis/build/status/microsoft.spacy-ann-linker?branchName=master" target="_blank">
    <img src="https://dev.azure.com/kakh/spacy-ann-linker/_apis/build/status/microsoft.spacy-ann-linker?branchName=master" alt="Build Status">
</a>
<a href="https://pypi.org/project/spacy-ann-linker" target="_blank">
    <img src="https://badge.fury.io/py/spacy-ann-linker.svg" alt="Package version">
</a>
<a href="https://codecov.io/gh/microsoft/spacy-ann-linker" target="_blank">
  <img src="https://codecov.io/gh/microsoft/spacy-ann-linker/branch/master/graph/badge.svg" alt="Code Coverage"/>
</a>
</p>

---

**Documentation**: <a href="https://microsoft.github.io/spacy-ann-linker" target="_blank">https://microsoft.github.io/spacy-ann-linker</a>

**Source Code**: <a href="https://github.com/microsoft/spacy-ann-linker" target="_blank">https://github.com/microsoft/spacy-ann-linker</a>

---

spaCy ANN Linker is a <a href="https://github.com/explosion/spaCy" target="_blank">spaCy</a> a pipeline component for generating alias candidates for spaCy entities in `doc.ents`. It provides an optional interface for linking ambiguous aliases based on descriptions for each entity.

The key features are:

* **Easy spaCy Integration**: spaCy ANN Linker provides completely serializable spaCy pipeline components that integrate directly into your existing spaCy model.
* **CLI for simple Index Creation**: Simply run `spacy_ann create_index` with your data to create an Approximate Nearest Neighbors index from your data, make an `ann_linker` pipeline component and save a spaCy model.

* **Built in Web API** for easy deployment and Batch Entity Linking queries

## Requirements

Python 3.6+

spaCy ANN Linker is convenient wrapper built on a few comprehensive, high-performing packages.

* <a href="https://spacy.io" class="external-link" target="_blank">spaCy</a>
* <a href="https://github.com/nmslib/nmslib" class="external-link" target="_blank">nmslib (ANN Index)</a>
* <a href="https://scikit-learn.org/stable/" class="external-link" target="_blank">scikit-learn (TF-IDF)</a>.
* <a href="https://fastapi.tiangolo.com" class="external-link" target="_blank">FastAPI (Web Service)</a>.

## Installation

<div class="termy">

```console
$ pip install spacy-ann-linker
---> 100%
Successfully installed spacy-ann-linker
```

</div>

## Data Prerequisites

To use this spaCy ANN Linker you need pre-existing Knowledge Base data.
spaCy ANN Linker expects data to exist in 2 JSONL files together in a directory

```
kb_dir
│   aliases.jsonl
│   entities.jsonl
```

For testing the package, you can use the example data in `examples/tutorial/data`

```
examples/tutorial/data
│   aliases.jsonl
│   entities.jsonl
```

### **entities.jsonl Record Format**

```json
{"id": "Canonical Entity Id", "description": "Entity Description used for Disambiguation"}
```

**Example data**
```json
{"id": "a1", "description": "Machine learning (ML) is the scientific study of algorithms and statistical models..."}
{"id": "a2", "description": "ML (\"Meta Language\") is a general-purpose functional programming language. It has roots in Lisp, and has been characterized as \"Lisp with types\"."}
{"id": "a3", "description": "Natural language processing (NLP) is a subfield of linguistics, computer science, information engineering, and artificial intelligence concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data."}
{"id": "a4", "description": "Neuro-linguistic programming (NLP) is a pseudoscientific approach to communication, personal development, and psychotherapy created by Richard Bandler and John Grinder in California, United States in the 1970s."}
...
```

### **aliases.jsonl Record Format**

```json
{"alias": "alias string", "entities": ["list", "of", "entity", "ids"], "probabilities": [0.5, 0.5]}
```

**Example data**
```json
{"alias": "ML", "entities": ["a1", "a2"], "probabilities": [0.5, 0.5]}
{"alias": "Machine learning", "entities": ["a1"], "probabilities": [1.0]}
{"alias": "Meta Language", "entities": ["a2"], "probabilities": [1.0]}
{"alias": "NLP", "entities": ["a3", "a4"], "probabilities": [0.5, 0.5]}
{"alias": "Natural language processing", "entities": ["a3"], "probabilities": [1.0]}
{"alias": "Neuro-linguistic programming", "entities": ["a4"], "probabilities": [1.0]}
...
```
  
## Example Data

`spacy-ann-linker` comes with some example data to get you started.

!!! important
    If this is your first time using `spacy-ann-linker` start out with the example data using the `spacy_ann example_data` command. Just pass an output_dir to write the example data to.

<div class="termy">

```console
$ spacy_ann example_data ./kb

=============== Example Data ================
Writing Example data to test/kb
✔ Done.
```

</div>

This should leave you with a folder called ./kb_dir that has a structure like

```
kb_dir
│   aliases.jsonl
│   entities.jsonl
```

## spaCy prerequisites

If you don't have a pretrained spaCy model, download one now. The model needs to have vectors
so download a model bigger than `en_core_web_sm`


<div class="termy">

```console
$ spacy download en_core_web_md
---> 100%
Successfully installed en_core_web_md
```

</div>

## Next Steps

Once you have the Data and spaCy prerequisites completed follow along with the [Tutorial](https://microsoft.github.io/spacy-ann-linker/tutorial/create_index/) to for a step-by-step guide for using the `spacy_ann` package.

!!! important
    These are just the prerequisites. Follow the full tutorial linked above for a step-by-step guide to working with `spacy-ann-linker`.

## License

This project is licensed under the terms of the MIT license.
