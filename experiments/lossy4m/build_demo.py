"""Build the public Ck10 CP4M example and standalone static visualizer."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from cpos.io import read_pos

from .codec import decode_retained, encode

DEFAULT_SOURCE = Path(
    "/Users/kyle/Documents/GitHub/uap/rangefinder/controls/"
    "control_Ck10_steel_felfer_R56_01769.pos"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--demo",
        type=Path,
        default=Path(__file__).parent / "demo",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path(__file__).parent / "results" / "full.json",
    )
    parser.add_argument("--site", type=Path)
    args = parser.parse_args()
    points = read_pos(args.pos)
    payload = encode(points)
    retained = decode_retained(payload)
    data = args.demo / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "example.cp4m").write_bytes(payload)
    metadata = {
        "title": "Ck10 steel",
        "source_points": len(points),
        "stored_points": retained.header.stored_point_count,
        "synthesized_points": (
            retained.header.original_point_count
            - retained.header.stored_point_count
        ),
        "file_bytes": len(payload),
        "bin_width_da": retained.header.spectrum_bin_da,
        "allocation_exponent": retained.header.allocation_exponent,
    }
    (data / "example.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))

    if args.benchmark.is_file():
        result = json.loads(args.benchmark.read_text())
        public_result = {
            "target_points": result["target_points"],
            "bin_width_da": result["bin_width_da"],
            "allocation_exponent": result["allocation_exponent"],
            "files": [
                {
                    "name": Path(record["label"]).name,
                    "points": record["point_count"],
                    "raw_bytes": record["raw_bytes"],
                    "cpos_bytes": record["cpos_1_0"]["file_bytes"],
                    "cp4m_bytes": record["cp4m"]["file_bytes"],
                    "cp4m_exact_fraction": record["cp4m"]["exact_fraction"],
                    "cpos_spatial_js": (
                        record["cpos_1_0"]["quality"]["spatial_js_bits"]
                    ),
                    "cp4m_spatial_js": (
                        record["cp4m"]["quality"]["spatial_js_bits"]
                    ),
                }
                for record in result["files"]
            ],
        }
        (data / "benchmark.json").write_text(
            json.dumps(public_result, indent=2) + "\n",
        )

    if args.site:
        root = Path(__file__).parents[2]
        args.site.mkdir(parents=True, exist_ok=True)
        for name in (
            "index.html",
            "style.css",
            "app.js",
            "renderer.js",
            "spectrum.js",
            "encoder-worker.js",
        ):
            shutil.copy2(args.demo / name, args.site / name)
        shutil.copy2(
            Path(__file__).parent / "javascript" / "cp4m.js",
            args.site / "cp4m.js",
        )
        shutil.copy2(
            Path(__file__).parent / "javascript" / "cp4m-encode.js",
            args.site / "cp4m-encode.js",
        )
        shutil.copytree(
            Path(__file__).parent / "javascript" / "vendor",
            args.site / "vendor",
            dirs_exist_ok=True,
        )
        shutil.copy2(root / "javascript" / "cpos.js", args.site / "cpos.js")
        shutil.copytree(data, args.site / "data", dirs_exist_ok=True)
        shutil.copy2(
            root / "demo" / "data" / "example.cpos",
            args.site / "data" / "example.cpos",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
