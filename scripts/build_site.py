"""Assemble the dependency-free GitHub Pages tree in ``site/``."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DEMO_FILES = ("index.html", "style.css", "app.js", "renderer.js")
DATA_FILES = ("example.cpos", "example.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("site"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output == root or root not in output.parents:
        raise RuntimeError("site output must be a dedicated directory inside the repo")
    if output.exists():
        shutil.rmtree(output)
    (output / "data").mkdir(parents=True)
    (output / "javascript").mkdir()

    for name in DEMO_FILES:
        shutil.copy2(root / "demo" / name, output / name)
    for name in DATA_FILES:
        shutil.copy2(root / "demo" / "data" / name, output / "data" / name)
    shutil.copy2(root / "javascript" / "cpos.js", output / "javascript" / "cpos.js")
    (output / ".nojekyll").write_text("")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
