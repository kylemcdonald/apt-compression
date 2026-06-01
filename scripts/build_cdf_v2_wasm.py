from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    src = ROOT / "src" / "wasm" / "cdf_v2_sampler.ts"
    out = ROOT / "public" / "cdf_v2_sampler.wasm"
    out.parent.mkdir(parents=True, exist_ok=True)
    asc = ROOT / "node_modules" / ".bin" / "asc"
    cmd = [
        str(asc),
        str(src),
        "--outFile",
        str(out),
        "--optimize",
        "--runtime",
        "stub",
        "--initialMemory",
        "16",
        "--maximumMemory",
        "32768",
        "--exportRuntime",
    ]
    subprocess.run(cmd, check=True)
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
