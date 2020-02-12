import spacy

if __name__ == "__main__":
    nlp = spacy.blank("en")
    aliases = ['machine learning', 'ML', 'NLP', 'researched']
    ruler = nlp.create_pipe('entity_ruler', {"overwrite_ents": True})
    patterns = [{"label": "SKILL", "pattern": alias} for alias in aliases]
    ruler.add_patterns(patterns)

    remote_ann_linker = nlp.create_pipe('remote_ann_linker', {
        'base_url': "http://localhost:8080/link"
    })
    nlp.add_pipe(remote_ann_linker)

    doc = nlp("NLP is a highly researched area of machine learning")
    print([(e.text, e.label_, e.kb_id_) for e in doc.ents])
    
    # Outputs:
    # [('NLP', 'SKILL', 'a3'), ('Machine learning', 'SKILL', 'a1')]
    #
    # In our entities.jsonl file
    # a3 => Natural Language Processing
    # a1 => Machine learning