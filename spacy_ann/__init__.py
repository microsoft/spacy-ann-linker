# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""spaCy ANN Linker, a pipeline component for generating spaCy KnowledgeBase Alias Candidates for Entity Linking."""


__version__ = "0.3.3"

from .ann_linker import AnnLinker
from .remote_ann_linker import RemoteAnnLinker
