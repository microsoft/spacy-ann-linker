# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import requests
import srsly
from requests import HTTPError
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.util import ensure_path, from_disk, minibatch, to_disk
from typing import List, Tuple, Dict, Any, Optional


@Language.factory(
    "remote_ann_linker",
    requires=["doc.ents", "doc.sents", "token.ent_iob", "token.ent_type"],
    assigns=["span._.kb_alias"],
)
def make_remote_ann_linker(
    nlp: Language,
    name: str = 'remote_ann_linker', 
    cfg: Optional[Dict[str, Any]] = dict(),
):
    """Construct an ANNLinker component.
    """
    return RemoteAnnLinker(
        nlp, name, cfg
    )


class RemoteAnnLinker:
    """The RemoteAnnLinker interfaces with a Remote Server to handle 
    Entity Linking when the KnowledgeBase and ANN Index cannot be in memory.
    """

    @classmethod
    def from_nlp(cls, nlp, **cfg):
        """Used in spacy.language.Language when constructing this pipeline.
        Tells spaCy that this pipe requires the nlp object
        
        nlp (Language): spaCy Language object
        
        RETURNS (RemoteAnnLinker): Initialized RemoteAnnLinker.
        """
        return cls(nlp, **cfg)

    def __init__(self, 
                 nlp: Language, 
                 name: str = 'remote_ann_linker', 
                 cfg: Optional[Dict[str, Any]] = dict()):
        """Initialize the RemoteAnnLinker
        
        nlp (Language): spaCy Language object
        """
        Span.set_extension("kb_alias", default="", force=True)

        self.nlp = nlp
        self.name = name
        self.cfg = dict(cfg)
        self.base_url = cfg.get("base_url")
        self.headers = cfg.get("headers", {})

    @property
    def aliases(self) -> List[str]:
        """Get all aliases
        
        RETURNS (List[str]): List of aliases
        """
        return self.kb.get_alias_strings()

    def _ents_to_json(self, ents: List[Span]) -> List[Dict[str, Any]]:
        """Convert spaCy ents to JSON
        
        ents (List[Span]): List of spaCy ents from `doc.ents`
        
        RETURNS (List[Dict[str, Any]]): JSON spans
        """
        return [
            {
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_,
            }
            for ent in ents
        ]

    def __call__(self, doc: Doc) -> Doc:
        """Annotate spaCy doc.ents with candidate info.
        to pick the most likely Candidate
        
        doc (Doc): spaCy Doc
        
        RETURNS (Doc): spaCy Doc with updated annotations
        """

        documents = [{"spans": self._ents_to_json(doc.ents), "context": doc.text}]

        data = self._make_request(documents)
        for ent, span in zip(doc.ents, data["documents"][0]["spans"]):
            if span["id"]:
                for t in ent:
                    t.ent_kb_id_ = span["id"]

        return doc

    def pipe(
        self,
        stream: Generator[Doc, None, None],
        batch_size: int = 32,
        n_threads: int = -1,
    ) -> Generator[Doc, None, None]:
        """Annotate a stream of spaCy docs ents with candidate info.
        to pick the most likely Candidate
        
        docs (Generator[Doc]): Stream of spaCy Docs
        
        RETURNS (Generator[Doc]): Stream of spaCy Docs with updated annotations
        """

        for docs in minibatch(stream, size=batch_size):
            documents = [
                {"spans": self._ents_to_json(doc.ents), "context": doc.text}
                for doc in docs
            ]

            data = self._make_request(documents)

            for spacy_doc, res_doc in zip(docs, data["documents"]):
                for ent, span in zip(spacy_doc.ents, res_doc["spans"]):
                    if span["id"]:
                        for t in ent:
                            t.ent_kb_id_ = span["id"]

            yield from docs

    def _make_request(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make request to remote Web Service with batch
        of Documents
        
        documents (Dict[str, Any]): Batch of documents to link
            entity spans
        
        RAISES:
            ValueError: If there is a server error, raise and exit
        
        RETURNS (Dict[str, Any]): List of Documents with id prop set on each span
        """

        res = requests.post(
            self.base_url, headers=self.headers, json={"documents": documents}
        )
        try:
            res.raise_for_status()
        except HTTPError as e:
            raise ValueError("Error in making request to the server.", e)
        data = res.json()
        return data

    def from_disk(self, path: Path, **kwargs):
        """Deserialize saved RemoteAnnLinker from disk.
        
        path (Path): directory to deserialize from
        
        RETURNS (RemoteAnnLinker): Initialized RemoteAnnLinker
        """
        path = ensure_path(path)
        cfg = {}
        deserializers = {"cfg": lambda p: cfg.update(srsly.read_json(p))}
        from_disk(path, deserializers, {})
        self.cfg.update(cfg)
        self.base_url = cfg.get("base_url")
        self.headers = cfg.get("headers", {})

        return self

    def to_disk(self, path: Path, exclude: Tuple = tuple(), **kwargs):
        """Serialize RemoteAnnLinker to disk.
        
        path (Path): directory to serialize to
        exclude (Tuple, optional): config to exclude. Defaults to tuple().
        """
        path = ensure_path(path)
        serializers = {"cfg": lambda p: srsly.write_json(p, self.cfg)}

        to_disk(path, serializers, {})
