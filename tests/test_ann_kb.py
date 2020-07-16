from spacy_ann.ann_kb import AnnKnowledgeBase
from wasabi import msg


def test_fit_index(nlp, entities, aliases):
    kb = AnnKnowledgeBase(nlp.vocab, entity_vector_length=300)
    print(vars(kb))

    entity_ids = []
    descriptions = []
    freqs = []
    for e in entities:
        entity_ids.append(e["id"])
        descriptions.append(e.get("description", ""))
        freqs.append(100)

    msg.divider("Apply EntityEncoder")

    with msg.loading("Applying EntityEncoder to descriptions"):
        # get the pretrained entity vectors
        embeddings = [nlp.make_doc(desc).vector for desc in descriptions]
        msg.good("Finished, embeddings created")

    with msg.loading("Setting kb entities and aliases"):
        # set the entities, can also be done by calling `kb.add_entity` for each entity
        for i in range(len(entity_ids)):
            entity = entity_ids[i]
            if not kb.contains_entity(entity):
                kb.add_entity(entity, freqs[i], embeddings[i])

        for a in aliases:
            ents = [e for e in a["entities"] if kb.contains_entity(e)]
            n_ents = len(ents)
            if n_ents > 0:
                prior_prob = [1.0 / n_ents] * n_ents
                kb.add_alias(alias=a["alias"], entities=ents, probabilities=prior_prob)

    kb.fit_index(verbose=True)

    assert kb.get_candidates("research")[0].entity_ == "a15"

    assert kb.get_candidates("researched")[0].alias_ == "Research"
    assert kb.get_candidates("researched")[0].entity_ == "a15"
