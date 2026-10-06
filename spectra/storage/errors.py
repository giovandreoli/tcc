"""Exception hierarchy shared by the storage layer.

Security and persistence failures share a root so a caller can guard a whole operation
with one ``except`` without having to know which layer refused.
"""

from __future__ import annotations


class StorageError(Exception):
    """Base class for every storage and security failure."""
