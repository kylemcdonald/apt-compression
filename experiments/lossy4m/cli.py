"""Command-line interface for the experimental CP4M codec."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cpos.io import read_pos, write_pos

from .codec import (
    DEFAULT_ALLOCATION_EXPONENT,
    DEFAULT_BIN_WIDTH_DA,
    DEFAULT_TARGET_POINTS,
    decode,
    decode_retained,
    encode,
    inspect,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.lossy4m.cli",
        description="Experimental mass-aware four-million-seed CPOS codec",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    encode_parser = commands.add_parser("encode", help="encode POS as CP4M")
    encode_parser.add_argument("input", type=Path)
    encode_parser.add_argument("output", type=Path)
    encode_parser.add_argument(
        "--target-points",
        type=int,
        default=DEFAULT_TARGET_POINTS,
    )
    encode_parser.add_argument(
        "--bin-width-da",
        type=float,
        default=DEFAULT_BIN_WIDTH_DA,
    )
    encode_parser.add_argument(
        "--allocation-exponent",
        type=float,
        default=DEFAULT_ALLOCATION_EXPONENT,
    )

    decode_parser = commands.add_parser("decode", help="decode CP4M as POS")
    decode_parser.add_argument("input", type=Path)
    decode_parser.add_argument("output", type=Path)
    decode_parser.add_argument(
        "--noise",
        choices=("none", "uniform", "gaussian"),
        default="uniform",
    )
    decode_parser.add_argument(
        "--retained-only",
        action="store_true",
        help="write only exact retained 12-bit seeds",
    )

    inspect_parser = commands.add_parser(
        "inspect",
        help="print versioned CP4M metadata",
    )
    inspect_parser.add_argument("input", type=Path)

    args = parser.parse_args()
    if args.command == "encode":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(encode(
            read_pos(args.input),
            target_points=args.target_points,
            bin_width_da=args.bin_width_da,
            allocation_exponent=args.allocation_exponent,
        ))
        return 0
    if args.command == "decode":
        payload = args.input.read_bytes()
        points = (
            decode_retained(payload).points
            if args.retained_only
            else decode(payload, noise=args.noise).points
        )
        write_pos(args.output, points)
        return 0

    print(json.dumps(inspect(args.input.read_bytes()).to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
