# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import NamedTuple


class AliasCandidate(NamedTuple):
    """
    A data class representing a candidate alias that a NER mention may be linked to.
    Parameters
    ----------
    alias : str, required.
        The alias close to the NER mention
    similarity: float, required.
        The similarity from the mention text to the alias in tf-idf space.
    """

    alias: str
    similarity: float
