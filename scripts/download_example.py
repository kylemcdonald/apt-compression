"""Download the public Ck10 steel POS control used by the CPOS demo."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

ARCHIVE_URL = (
    "https://zenodo.org/api/records/7979668/files/"
    "ger_erlangen_felfer_ck10.zip/content"
)
ARCHIVE_MD5 = "a00725b05df2094922a87585fa4c79f4"
ARCHIVE_NAME = "ger_erlangen_felfer_ck10.zip"
MEMBER_SUFFIX = "R56_01769-v01.pos"
OUTPUT_NAME = "control_Ck10_steel_felfer_R56_01769.pos"
OUTPUT_SIZE = 88_405_776


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(destination: Path, force: bool = False) -> Path:
    destination = destination.resolve()
    if destination.exists() and destination.stat().st_size == OUTPUT_SIZE:
        return destination
    if destination.exists() and not force:
        raise RuntimeError(
            f"{destination} exists with the wrong size; pass --force to replace it"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    archive = destination.parent / ARCHIVE_NAME
    if force or not archive.exists() or file_md5(archive) != ARCHIVE_MD5:
        partial = archive.with_suffix(archive.suffix + ".part")
        request = urllib.request.Request(
            ARCHIVE_URL,
            headers={"User-Agent": "apt-cpos/1.0"},
        )
        print(f"Downloading {ARCHIVE_URL}")
        with urllib.request.urlopen(request) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        if file_md5(partial) != ARCHIVE_MD5:
            partial.unlink(missing_ok=True)
            raise RuntimeError("downloaded Zenodo archive failed its MD5 check")
        partial.replace(archive)

    partial_output = destination.with_suffix(destination.suffix + ".part")
    with zipfile.ZipFile(archive) as source:
        matches = [
            name for name in source.namelist() if name.endswith(MEMBER_SUFFIX)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one {MEMBER_SUFFIX} member, found {matches}"
            )
        with source.open(matches[0]) as compressed, partial_output.open("wb") as output:
            shutil.copyfileobj(compressed, output)
    if partial_output.stat().st_size != OUTPUT_SIZE:
        partial_output.unlink(missing_ok=True)
        raise RuntimeError("extracted POS file has an unexpected size")
    partial_output.replace(destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / OUTPUT_NAME,
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(download(args.output, force=args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
