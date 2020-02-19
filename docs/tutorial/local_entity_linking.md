# Tutorial - Local Entity Linking

In the previous step, you ran the `spacy_ann create_index` CLI command. The output of this
command is a loadable spaCy model with an `ann_linker` capable of Entity Linking against your KnowledgeBase data.
You can load the saved model from `output_dir` in the previous step just like you would any normal spaCy model.

## Load `ann_linker` model

First load the model created by `spacy_ann create_index`

```Python hl_lines="8 9"
{!./src/local_ann_linker.py!}
```

## Load Extraction Model

This is a bit of misnomar for the provided example code.
You likely want a trained NER model but the purpose of this example we'll just arbitrarily extract entities using the spaCy `EntityRuler` component by just add a few terms to it that are close to those in our KnowledgeBase.

```Python hl_lines="14 15 16 17 18 19 20"
{!./src/local_ann_linker.py!}
```

## Test the trained `ann_linker` component

Run the pipeline on some sample text and ensure that you have `e.kb_id_` set properly for each entity. You should get id `a3` for "NLP" and id `a1` for "machine learn

```Python hl_lines="22 23 24"
{!./src/local_ann_linker.py!}
```

## Next Steps

This works great when you can afford to fit your KnowledgeBase in memory and have full access to your KnowledgeBase. In the next step of this tutorial, we'll talk about hosting the KnowledgeBase and ANN Index remotely and making batch calls to the endpoint so you can keep the KnowledgeBase and model code separate.
