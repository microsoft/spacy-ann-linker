<!-- <p align="center">
  <a href="https://microsoft.github.io/spacy-ann-linker"><img src="https://typer.tiangolo.com/img/logo-margin/logo-margin-vectoar.svg" alt="spaCy ANN Linker"></a>
</p> -->
<p>
    <em>spaCy ANN Linker, a pipeline component for generating spaCy KnowledgeBase Alias Candidates for Entity Linking.</em>
</p>
<p align="center">
<a href="https://dev.azure.com/kakh/spacy-ann-linker/_apis/build/status/microsoft.spacy-ann-linker?branchName=master" target="_blank">
    <img src="https://dev.azure.com/kakh/spacy-ann-linker/_apis/build/status/microsoft.spacy-ann-linker?branchName=master" alt="Build Status">
</a>
<a href="https://pypi.org/project/spacy-ann-linker" target="_blank">
    <img src="https://badge.fury.io/py/spacy-ann-linker.svg" alt="Package version">
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
* <a href="https://github.com/nmslib/nmslib" class="external-link" target="_blank">nmslib (ANN Index)</a>.
* <a href="https://github.com/nmslib/nmslib" class="external-link" target="_blank">nmslib (ANN Index)</a>.
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

## Usage

Once you have your data, and a spaCy model with vectors, compute the nearest neighbors index for your Aliases.

Run the `create_index` help command to understand the required arguments.

<div class="termy">

```console
$ spacy_ann create_index --help 
spacy_ann create_index --help
Usage: spacy_ann create_index [OPTIONS] MODEL KB_DIR OUTPUT_DIR

  Create an ApproxNearestNeighborsLinker based on the Character N-Gram TF-
  IDF vectors for aliases in a KnowledgeBase

  model (str): spaCy language model directory or name to load kb_dir (Path):
  path to the directory with kb entities.jsonl and aliases.jsonl files
  output_dir (Path): path to output_dir for spaCy model with ann_linker pipe

  kb File Formats

  e.g. entities.jsonl

  {"id": "a1", "description": "Machine learning (ML) is the scientific study
  of algorithms and statistical models..."} {"id": "a2", "description": "ML
  ("Meta Language") is a general-purpose functional programming language. It
  has roots in Lisp, and has been characterized as "Lisp with types"."}

  e.g. aliases.jsonl {"alias": "ML", "entities": ["a1", "a2"],
  "probabilities": [0.5, 0.5]}

Options:
  --new-model-name TEXT
  --cg-threshold FLOAT
  --n-iter INTEGER
  --verbose / --no-verbose
  --install-completion      Install completion for the current shell.
  --show-completion         Show completion for the current shell, to copy it
                            or customize the installation.
  --help                    Show this message and exit.
```

</div>

Now provide the required arguments. I'm using the example data but at this step use your own.
the `create_index` command will run a few steps and you should see an output like the one below.

<div class="termy">

```console
spacy_ann create_index en_core_web_md examples/tutorial/data examples/tutorial/models

// The create_index command runs a few steps

// Load the model passed as the first positional argument (en_core_web_md)
===================== Load Model ======================
⠹ Loading model en_core_web_md✔ Done.
ℹ 0 entities without a description

// Train an EntityEncoder on the descriptions of each Entity
================= Train EntityEncoder =================
⠸ Starting training EntityEncoder✔ Done Training

// Apply the EntityEncoder to get the final vectors for each entity
================= Apply EntityEncoder =================
⠙ Applying EntityEncoder to descriptions✔ Finished, embeddings created
✔ Done adding entities and aliases to kb

// Create Nearest Neighbors index from the Aliases in kb_dir/aliases.jsonl
================== Create ANN Index ===================
Fitting tfidf vectorizer on 6 aliases
Fitting and saving vectorizer took 0.012949 seconds
Finding empty (all zeros) tfidf vectors
Deleting 2/6 aliases because their tfidf is empty
Fitting ann index on 4 aliases

0%   10   20   30   40   50   60   70   80   90   100%
|----|----|----|----|----|----|----|----|----|----|
***************************************************
Fitting ann index took 0.030826 seconds

```
</div>


### Using the saved model

Now that you have a trained spaCy ANN Linker component you can load the saved model from `output_dir` and run
it just like you would any normal spaCy model.

```Python
import spacy
from spacy.tokens import Span

# Load the spaCy model from the output_dir you used
# from the create_index command
model_dir = "examples/tutorial/models/ann_linker"
nlp = spacy.load(model_dir)

# The NER component of the en_core_web_md model doesn't actually
# recognize the aliases as entities so we'll add a 
# spaCy EntityRuler component for now to extract them.
ruler = nlp.create_pipe('entity_ruler')
patterns = [
    {"label": "SKILL", "pattern": alias}
    for alias in nlp.get_pipe('ann_linker').kb.get_alias_strings() + ['machine learn']
]
ruler.add_patterns(patterns)
nlp.add_pipe(ruler, before="ann_linker")

doc = nlp("NLP is a subset of machine learn.")

print([(e.text, e.label_, e.kb_id_) for e in doc.ents])

# Outputs:
# [('NLP', 'SKILL', 'a3'), ('Machine learning', 'SKILL', 'a1')]
#
# In our entities.jsonl file
# a3 => Natural Language Processing
# a1 => Machine learning
```


<!-- 
### Recap

In summary, you declare **once** the types of parameters (*arguments* and *options*) as function parameters.

You do that with standard modern Python types.

You don't have to learn a new syntax, the methods or classes of a specific library, etc.

Just standard **Python 3.6+**.

For example, for an `int`:

```Python
total: int
```

or for a `bool` flag:

```Python
force: bool
```

And similarly for **files**, **paths**, **enums** (choices), etc. And there are tools to create **groups of subcommands**, add metadata, extra **validation**, etc.

**You get**: great editor support, including **completion** and **type checks** everywhere.

**Your users get**: automatic **`--help`**, (optional) **autocompletion** in their terminal (Bash, Zsh, Fish, PowerShell).

For a more complete example including more features, see the <a href="https://typer.tiangolo.com/tutorial/">Tutorial - User Guide</a>.

## Optional Dependencies

Typer uses <a href="https://click.palletsprojects.com/" class="external-link" target="_blank">Click</a> internally. That's the only dependency.

But you can also install extras:

* <a href="https://pypi.org/project/colorama/" class="external-link" target="_blank"><code>colorama</code></a>: and Click will automatically use it to make sure your terminal's colors always work correctly, even in Windows.
    * Then you can use any tool you want to output your terminal's colors in all the systems, including the integrated `typer.style()` and `typer.secho()` (provided by Click).
    * Or any other tool, e.g. <a href="https://pypi.org/project/wasabi/" class="external-link" target="_blank"><code>wasabi</code></a>, <a href="https://github.com/erikrose/blessings" class="external-link" target="_blank"><code>blessings</code></a>.
* <a href="https://github.com/click-contrib/click-completion" class="external-link" target="_blank"><code>click-completion</code></a>: and Typer will automatically configure it to provide completion for all the shells, including installation commands.

You can install `typer` with `colorama` and `click-completion` with `pip install typer[all]`.

## Other tools and plug-ins

Click has many plug-ins available that you can use. And there are many tools that help with command line applications that you can use as well, even if they are not related to Typer or Click.

For example:

* <a href="https://github.com/click-contrib/click-spinner" class="external-link" target="_blank"><code>click-spinner</code></a>: to show the user that you are loading data. A Click plug-in.
    * There are several other Click plug-ins at <a href="https://github.com/click-contrib" class="external-link" target="_blank">click-contrib</a> that you can explore.
* <a href="https://pypi.org/project/tabulate/" class="external-link" target="_blank"><code>tabulate</code></a>: to automatically display tabular data nicely. Independent of Click or typer.
* etc... you can re-use many of the great available tools for building CLIs. -->

## License

This project is licensed under the terms of the MIT license.
