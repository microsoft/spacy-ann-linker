# Tutorial - Create an ANN Index

Once you have your data in the supported format, and a spaCy model with vectors you can use the `spacy_ann` CLI to compute the nearest neighbors index for your Aliases and tran an Encoder for disambiguating entity spans to their canonical Id

Run the `create_index` help command to understand the required arguments.

<div class="termy">

```console
$ spacy_ann create_index --help 
spacy_ann create_index --help
Usage: spacy_ann create_index [OPTIONS] MODEL KB_DIR OUTPUT_DIR

  Create an AnnLinker based on the Character N-Gram TF-
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
