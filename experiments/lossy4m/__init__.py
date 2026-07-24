"""Experimental mass-aware four-million-point CPOS successor."""

from .codec import (
    CODEC_VERSION,
    CONTAINER_VERSION,
    DEFAULT_ALLOCATION_EXPONENT,
    DEFAULT_BIN_WIDTH_DA,
    DEFAULT_TARGET_POINTS,
    DecodedCloud,
    Lossy4mHeader,
    decode,
    decode_retained,
    encode,
    inspect,
)

__all__ = [
    "CODEC_VERSION",
    "CONTAINER_VERSION",
    "DEFAULT_ALLOCATION_EXPONENT",
    "DEFAULT_BIN_WIDTH_DA",
    "DEFAULT_TARGET_POINTS",
    "DecodedCloud",
    "Lossy4mHeader",
    "decode",
    "decode_retained",
    "encode",
    "inspect",
]
