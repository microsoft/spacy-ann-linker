import spacy

if __name__ == "__main__":
    nlp = spacy.blank("en")
    aliases = ["machine learning", "ML", "NLP", "researchers"]
    ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})
    patterns = [{"label": "SKILL", "pattern": alias} for alias in aliases]
    ruler.add_patterns(patterns)

    remote_ann_linker = nlp.add_pipe(
        "remote_ann_linker", config={"base_url": "http://localhost:8080/link"}
    )

    #  researchers 是一个新的mention，不在alias数据库里，但是通过与Research的高度近似，链接到a15实体
    doc = nlp("NLP is a highly researchers focuses area of machine learning")
    print([(e.text, e.label_, e.kb_id_) for e in doc.ents])
    
    # Outputs:
    # [('NLP', 'SKILL', 'a3'), ('Machine learning', 'SKILL', 'a1')]
    #
    # In our entities.jsonl file
    # a3 => Natural Language Processing
    # a1 => Machine learning
