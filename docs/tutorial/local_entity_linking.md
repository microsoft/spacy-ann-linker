# Tutorial - Local Entity Linking

In the previous step, you ran the `spacy_ann create_index` CLI command. The output of this
command is a loadable spaCy model with an `ann_linker` capable of Entity Linking against your KnowledgeBase data.
You can load the saved model from `output_dir` in the previous step just like you would any normal spaCy model.

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
