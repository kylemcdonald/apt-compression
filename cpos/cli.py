"""Command-line interface for CPOS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .codec import DEFAULT_MAX_POINTS, decode, encode, inspect
from .io import read_pos, write_pos


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="cpos",
        description="Lossy APT .POS codec for quick web-based previews",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    encode_parser = subparsers.add_parser("encode", help="encode POS as CPOS")
    encode_parser.add_argument("input", type=Path)
    encode_parser.add_argument("output", type=Path)
    encode_parser.add_argument(
        "--max-points", type=int, default=DEFAULT_MAX_POINTS
    )

    decode_parser = subparsers.add_parser("decode", help="decode CPOS as POS")
    decode_parser.add_argument("input", type=Path)
    decode_parser.add_argument("output", type=Path)

    inspect_parser = subparsers.add_parser(
        "inspect", help="print versioned CPOS metadata"
    )
    inspect_parser.add_argument("input", type=Path)

    args = parser.parse_args()
    if args.command == "encode":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(
            encode(read_pos(args.input), max_points=args.max_points)
        )
        return 0
    if args.command == "decode":
        write_pos(args.output, decode(args.input.read_bytes()))
        return 0

    print(json.dumps(inspect(args.input.read_bytes()).to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
