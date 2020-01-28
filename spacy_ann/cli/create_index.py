# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from collections import defaultdict
from pathlib import Path
from typing import Callable
import spacy
from spacy.kb import KnowledgeBase
from spacy.language import Language
from spacy.util import ensure_path
import srsly
import typer
from wasabi import Printer
from bin.wiki_entity_linking.train_descriptions import EntityEncoder

from spacy_ann import ApproxNearestNeighborsLinker
from spacy_ann.candidate_generator import CandidateGenerator

INPUT_DIM = 300  # dimension of pretrained input vectors
DESC_WIDTH = 300  # dimension of output entity vectors


def create_index(model: str,
                 kb_dir: Path,
                 output_dir: Path,
                 new_model_name: str = "ann_linker",
                 cg_threshold: float = 0.8,
                 n_iter: int = 5,
                 verbose: bool = True):

    """Create an ApproxNearestNeighborsLinker based on the Character N-Gram
    TF-IDF vectors for aliases in a KnowledgeBase

    model (str): spaCy language model directory or name to load
    kb_dir (Path): path to the directory with kb entities.jsonl and aliases.jsonl files
    output_dir (Path): path to output_dir for spaCy model with ann_linker pipe


    kb File Formats
    
    e.g. entities.jsonl

    {"id": "a1", "description": "Machine learning (ML) is the scientific study of algorithms and statistical models..."}
    {"id": "a2", "description": "ML (\"Meta Language\") is a general-purpose functional programming language. It has roots in Lisp, and has been characterized as \"Lisp with types\"."}

    e.g. aliases.jsonl
    {"alias": "ML", "entities": ["a1", "a2"], "probabilities": [0.5, 0.5]}
    """
    msg = Printer(hide_animation = not verbose)

    msg.divider("Load Model")
    with msg.loading(f"Loading model {model}"):
        nlp = spacy.load(model)
        msg.good("Done.")

    if output_dir is not None:
        output_dir = Path(output_dir / new_model_name)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

    entities = list(srsly.read_jsonl(kb_dir / "entities.jsonl"))
    aliases = list(srsly.read_jsonl(kb_dir / "aliases.jsonl"))
    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=INPUT_DIM)

    # set up the data
    entity_ids = []
    descriptions = []
    freqs = []
    n_no_desc = 0
    for e in entities:
        if "description" in e:
            entity_ids.append(e["id"])
            descriptions.append(e["description"])
            freqs.append(100)
        else:
            n_no_desc += 1

    msg.info(f"{n_no_desc} entities without a description")

    msg.divider("Train EntityEncoder")

    with msg.loading("Starting training EntityEncoder"):
        # training entity description encodings
        # this part can easily be replaced with a custom entity encoder
        encoder = EntityEncoder(nlp=nlp, input_dim=INPUT_DIM, desc_width=DESC_WIDTH, epochs=n_iter)
        encoder.train(description_list=descriptions, to_print=True)
        msg.good("Done Training")

    msg.divider("Apply EntityEncoder")

    with msg.loading("Applying EntityEncoder to descriptions"):
        # get the pretrained entity vectors
        embeddings = encoder.apply_encoder(descriptions)
        msg.good("Finished, embeddings created")

    with msg.loading("Setting kb entities and aliases"):
        # set the entities, can also be done by calling `kb.add_entity` for each entity
        for i in range(len(entity_ids)):
            entity = entity_ids[i]
            if not kb.contains_entity(entity):
                kb.add_entity(entity, freqs[i], embeddings[i])

        for a in aliases:
            if len(a["entities"]) > 0:
                prior_prob = [1.0 / len(a["entities"])] * len(a["entities"])
                kb.add_alias(alias=a["alias"], entities=a["entities"], probabilities=prior_prob)

        msg.good("Done adding entities and aliases to kb")

    cg = CandidateGenerator.create_with_defaults(nlp, kb)

    ann_linker = nlp.create_pipe("ann_linker")
    ann_linker.set_kb(kb)
    ann_linker.set_cg(cg)

    nlp.add_pipe(ann_linker, last=True)

    nlp.meta["name"] = new_model_name
    with nlp.disable_pipes("tagger", "parser", "ner"):
        nlp.to_disk(output_dir)
        nlp_loaded = nlp.from_disk(output_dir)


def _print_kb(kb):
    print(kb.get_size_entities(), "kb entities")
    print(kb.get_size_aliases(), "kb aliases")


if __name__ == "__main__":
    typer.run(create_index)