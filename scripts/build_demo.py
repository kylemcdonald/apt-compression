"""Build the compact public CPOS example used by the browser demo."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from cpos import decode, encode, inspect
from cpos.codec import DEFAULT_MAX_POINTS
from cpos.io import read_pos
from download_example import OUTPUT_NAME, download

EXAMPLE_NAME = "example.cpos"
METADATA_NAME = "example.json"
ZENODO_RECORD = "https://zenodo.org/records/7979668"
SOURCE_URL = (
    "https://zenodo.org/api/records/7979668/files/"
    "ger_erlangen_felfer_ck10.zip/content"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pos",
        type=Path,
        help="existing POS source; otherwise download the public control",
    )
    parser.add_argument(
        "--download-to",
        type=Path,
        default=Path("data") / OUTPUT_NAME,
    )
    parser.add_argument("--output", type=Path, default=Path("demo/data"))
    parser.add_argument(
        "--max-points", type=int, default=DEFAULT_MAX_POINTS
    )
    args = parser.parse_args()

    source = args.pos.resolve() if args.pos else download(args.download_to)
    points = read_pos(source)
    payload = encode(points, max_points=args.max_points)
    header = inspect(payload)
    decoded = decode(payload)

    args.output.mkdir(parents=True, exist_ok=True)
    cpos_path = args.output / EXAMPLE_NAME
    cpos_path.write_bytes(payload)
    metadata = {
        "title": "Ck10 steel",
        "source_file": "R56_01769-v01.pos",
        "source_archive": "ger_erlangen_felfer_ck10.zip",
        "source_url": SOURCE_URL,
        "zenodo_record": ZENODO_RECORD,
        "license": "CC-BY-4.0",
        "creator": "Peter Felfer / Martina Heller",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "original_size_bytes": source.stat().st_size,
        "cpos_size_bytes": len(payload),
        "original_point_count": len(points),
        "stored_point_count": len(decoded),
        "compression_ratio": round(source.stat().st_size / len(payload), 3),
        "container_version": ".".join(map(str, header.container_version)),
        "algorithm_version": ".".join(map(str, header.algorithm_version)),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    (args.output / METADATA_NAME).write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
