from __future__ import annotations

import copy
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / "artifacts" / "manifest.json"
PUBLIC_DIR = ROOT / "public"
PUBLIC_ARTIFACTS = PUBLIC_DIR / "artifacts"
COMPRESSED_METHOD = "cdf_v2_linear_64_10mb"
DATASET_PREFIX = "499"


def select_dataset(manifest: dict) -> dict:
    datasets = manifest.get("datasets", [])
    for dataset in datasets:
        name = str(dataset.get("name", "")).lower()
        if name.startswith(DATASET_PREFIX) and name.endswith(".pos"):
            return dataset
    for dataset in datasets:
        name = str(dataset.get("name", "")).lower()
        if name.startswith(DATASET_PREFIX):
            return dataset
    raise RuntimeError(f"No dataset starting with {DATASET_PREFIX!r} found in {SOURCE_MANIFEST}")


def build_manifest(source: dict, dataset: dict, artifact_rel: str) -> dict:
    method = copy.deepcopy(dataset.get("methods", {}).get(COMPRESSED_METHOD))
    if not method or method.get("available") is False:
        raise RuntimeError(f"{dataset.get('name')} is missing available {COMPRESSED_METHOD}")

    method["artifact"] = artifact_rel
    method["artifact_endpoint"] = artifact_rel
    method["available"] = True
    method["frontend_generated"] = True
    method["cdf_v2"] = True

    public_dataset = copy.deepcopy(dataset)
    public_dataset.pop("raw_path", None)
    public_dataset.pop("source_mtime", None)
    public_dataset["methods"] = {COMPRESSED_METHOD: method}

    return {
        "version": source.get("version", 1),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pages_demo": True,
        "methods": {
            COMPRESSED_METHOD: source.get("methods", {}).get(COMPRESSED_METHOD, "CDF grid v2")
        },
        "datasets": [public_dataset],
    }


def main() -> int:
    if not SOURCE_MANIFEST.exists():
        raise FileNotFoundError(f"{SOURCE_MANIFEST} does not exist. Run preprocess first.")
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    dataset = select_dataset(source)
    method = dataset["methods"][COMPRESSED_METHOD]
    artifact_src = ROOT / method["artifact"]
    if not artifact_src.exists():
        raise FileNotFoundError(f"Missing CDF v2 artifact: {artifact_src}")

    artifact_rel = f"artifacts/datasets/{dataset['id']}/cdf_v2/{artifact_src.name}"
    artifact_dst = PUBLIC_DIR / artifact_rel

    if PUBLIC_ARTIFACTS.exists():
        shutil.rmtree(PUBLIC_ARTIFACTS)
    artifact_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(artifact_src, artifact_dst)

    manifest = build_manifest(source, dataset, artifact_rel)
    (PUBLIC_ARTIFACTS / "manifest.json").write_text(
        json.dumps(manifest, indent=2, separators=(",", ": ")) + "\n",
        encoding="utf-8",
    )
    (PUBLIC_DIR / ".nojekyll").write_text("", encoding="utf-8")

    print(f"Wrote Pages demo manifest for {dataset['name']}")
    print(f"Copied {artifact_src} -> {artifact_dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
