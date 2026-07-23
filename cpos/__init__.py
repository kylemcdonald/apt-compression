"""Public API for the CPOS preview codec."""

from .codec import (
    ALGORITHM_VERSION,
    CONTAINER_VERSION,
    CposHeader,
    CposVersionError,
    decode,
    encode,
    inspect,
)

__all__ = [
    "ALGORITHM_VERSION",
    "CONTAINER_VERSION",
    "CposHeader",
    "CposVersionError",
    "decode",
    "encode",
    "inspect",
]
__version__ = "1.0.0"
