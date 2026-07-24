"""Render a reproducible Markdown summary of the CP4M benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


QUALITY_FIELDS = (
    "spatial_js_bits",
    "spatial_total_variation",
    "occupied_voxel_recall",
    "mean_axis_emd",
    "mass_js_bits",
    "mass_total_variation",
)


def _weighted(files: list[dict], codec: str, field: str) -> float:
    points = sum(record["point_count"] for record in files)
    return sum(
        record["point_count"] * record[codec]["quality"][field]
        for record in files
    ) / points


def _mib(value: int) -> str:
    return f"{value / 2**20:,.2f}"


def _short_name(label: str) -> str:
    return Path(label).name.replace("|", r"\|")


def render(result: dict) -> str:
    files = result["files"]
    if not files:
        raise ValueError("benchmark contains no completed files")
    total_points = sum(record["point_count"] for record in files)
    raw_bytes = sum(record["raw_bytes"] for record in files)
    cpos_bytes = sum(record["cpos_1_0"]["file_bytes"] for record in files)
    cp4m_bytes = sum(record["cp4m"]["file_bytes"] for record in files)
    cpos_seeds = sum(record["cpos_1_0"]["stored_points"] for record in files)
    cp4m_seeds = sum(record["cp4m"]["stored_points"] for record in files)
    cpos_encode = sum(record["cpos_1_0"]["encode_seconds"] for record in files)
    cp4m_encode = sum(record["cp4m"]["encode_seconds"] for record in files)
    cpos_decode = sum(record["cpos_1_0"]["decode_seconds"] for record in files)
    cp4m_decode = sum(
        record["cp4m"]["decode_expand_seconds"] for record in files
    )
    cpos_quality = {
        field: _weighted(files, "cpos_1_0", field)
        for field in QUALITY_FIELDS
    }
    cp4m_quality = {
        field: _weighted(files, "cp4m", field)
        for field in QUALITY_FIELDS
    }

    lines = [
        "# CPOS 4M experiment: 18-file results",
        "",
        "## Bottom line",
        "",
        (
            f"Across {len(files)} files containing {total_points:,} points, "
            f"CP4M stores {cp4m_seeds:,} exact 12-bit seeds and expands back "
            f"to all {total_points:,} records."
        ),
        "",
        (
            f"CP4M occupies **{cp4m_bytes:,} bytes ({_mib(cp4m_bytes)} MiB)** "
            f"versus **{cpos_bytes:,} bytes ({_mib(cpos_bytes)} MiB)** for "
            "CPOS 1.0."
        ),
        "",
        (
            f"That is {cp4m_bytes / cpos_bytes:.3f}× the bytes for "
            f"{cp4m_seeds / cpos_seeds:.3f}× as many exact retained points. "
            f"CP4M's point-weighted spatial JS divergence is "
            f"{cpos_quality['spatial_js_bits'] / cp4m_quality['spatial_js_bits']:.1f}× "
            "lower."
        ),
        "",
        "| Aggregate | CPOS 1.0 | CP4M |",
        "| --- | ---: | ---: |",
        f"| File bytes | {cpos_bytes:,} | {cp4m_bytes:,} |",
        f"| MiB | {_mib(cpos_bytes)} | {_mib(cp4m_bytes)} |",
        f"| Exact retained seeds | {cpos_seeds:,} | {cp4m_seeds:,} |",
        (
            f"| Exact fraction of corpus | {cpos_seeds / total_points:.2%} | "
            f"{cp4m_seeds / total_points:.2%} |"
        ),
        f"| Decoded/expanded points | {cpos_seeds:,} | {total_points:,} |",
        f"| Ratio vs float32 `.POS` | {raw_bytes / cpos_bytes:.2f}× | {raw_bytes / cp4m_bytes:.2f}× |",
        f"| Encode time (sum) | {cpos_encode:.2f} s | {cp4m_encode:.2f} s |",
        f"| Decode time (sum) | {cpos_decode:.2f} s | {cp4m_decode:.2f} s |",
        f"| Spatial JS divergence | {cpos_quality['spatial_js_bits']:.9f} | {cp4m_quality['spatial_js_bits']:.9f} |",
        f"| Spatial total variation | {cpos_quality['spatial_total_variation']:.9f} | {cp4m_quality['spatial_total_variation']:.9f} |",
        f"| Occupied-voxel recall | {cpos_quality['occupied_voxel_recall']:.6%} | {cp4m_quality['occupied_voxel_recall']:.6%} |",
        f"| Mean axis EMD | {cpos_quality['mean_axis_emd']:.9f} | {cp4m_quality['mean_axis_emd']:.9f} |",
        f"| Mass JS divergence (0.1 Da) | {cpos_quality['mass_js_bits']:.9f} | {cp4m_quality['mass_js_bits']:.9f} |",
        f"| Mass total variation (0.1 Da) | {cpos_quality['mass_total_variation']:.9f} | {cp4m_quality['mass_total_variation']:.9f} |",
        "",
        "## Per-file comparison",
        "",
        "| File | Points | `.POS` MiB | CPOS MiB | CP4M MiB | CP4M exact | CPOS spatial JS | CP4M spatial JS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in files:
        cpos = record["cpos_1_0"]
        cp4m = record["cp4m"]
        lines.append(
            f"| `{_short_name(record['label'])}` "
            f"| {record['point_count']:,} "
            f"| {_mib(record['raw_bytes'])} "
            f"| {_mib(cpos['file_bytes'])} "
            f"| {_mib(cp4m['file_bytes'])} "
            f"| {cp4m['exact_fraction']:.1%} "
            f"| {cpos['quality']['spatial_js_bits']:.7f} "
            f"| {cp4m['quality']['spatial_js_bits']:.7f} |"
        )
    lines.extend([
        "",
        "## What the metrics mean",
        "",
        (
            "Spatial metrics use a 32³ normalized voxel grid plus 256-bin "
            "axis marginals. CPOS points are weighted by the source/retained "
            "count ratio of their native mass bin. CP4M metrics use the actual "
            "fully expanded, uniformly dithered output."
        ),
        "",
        (
            "Mass metrics use 0.1 Da bins. CP4M stores the complete structural "
            "histogram and therefore restores it exactly. This does not mean "
            "that discarded within-bin mass positions were recovered."
        ),
        "",
        (
            "An exact CP4M seed means its stored four-field 12-bit tuple was "
            "recovered losslessly. Synthesized records are deterministic "
            "children of those seeds and are explicitly marked as synthesized."
        ),
        "",
        "## Reproduction",
        "",
        "```bash",
        "python3 -m experiments.lossy4m.benchmark \\",
        "  --output experiments/lossy4m/results/full.json",
        "python3 -m experiments.lossy4m.report \\",
        "  experiments/lossy4m/results/full.json \\",
        "  --output experiments/lossy4m/RESULTS.md",
        "```",
        "",
        (
            f"Parameters: target `{result['target_points']:,}`, histogram "
            f"`{result['bin_width_da']}` Da, allocation exponent "
            f"`{result['allocation_exponent']}`."
        ),
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.benchmark.read_text())
    args.output.write_text(render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
