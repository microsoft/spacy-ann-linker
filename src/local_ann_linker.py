import spacy
from spacy.tokens import Span

if __name__ == "__main__":

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

    print([(e.text, e.label_, e.kb_id_, e._.alias_candidates) for e in doc.ents])

    # Outputs:
    # [('NLP', 'SKILL', 'a3'), ('machine learn', 'SKILL', 'a1')]
    #
    # In our entities.jsonl file
    # a3 => Natural Language Processing
    # a1 => Machine learning
