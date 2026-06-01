from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def find_port(preferred: int) -> int:
    for port in range(preferred, preferred + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found near {preferred}")


def run_checked(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def ensure_node_modules() -> None:
    if (ROOT / "node_modules").exists():
        return
    run_checked(["npm", "install"])


def print_manifest_summary(manifest_path: Path) -> None:
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for dataset in manifest.get("datasets", []):
        for key, method in dataset.get("methods", {}).items():
            if key in {"full", "cdf_v2_linear_64_10mb"}:
                raw_mb = method.get("raw_size_bytes", dataset["raw_size_bytes"]) / (1024 * 1024)
                rows.append(
                    (
                        dataset["name"],
                        method.get("label", key),
                        raw_mb,
                        method.get("compressed_size_bytes", 0) / (1024 * 1024),
                        method.get("compression_ratio", 0),
                    )
                )
    if not rows:
        return
    print("\nCompression summary:")
    print("dataset | method | raw MB | compressed MB | ratio")
    print("--------+--------+--------+---------------+------")
    for dataset, method, raw_mb, compressed_mb, ratio in rows:
        print(f"{dataset} | {method} | {raw_mb:.2f} | {compressed_mb:.2f} | {ratio:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local APT compression viewer.")
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing before starting servers.")
    parser.add_argument("--quick", action="store_true", help="Use quick preprocessing settings.")
    parser.add_argument("--all", action="store_true", help="In quick mode, process every dataset.")
    parser.add_argument("--synthetic", action="store_true", help="Generate/process a synthetic dataset.")
    parser.add_argument("--port", type=int, default=5173)
    parser.add_argument("--api-port", type=int, default=8765)
    parser.add_argument("--skip-npm-install", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = ROOT / "artifacts" / "manifest.json"

    if args.preprocess or not manifest_path.exists():
        cmd = [sys.executable, "preprocess.py"]
        if args.quick:
            cmd.append("--quick")
        if args.all:
            cmd.append("--all")
        if args.synthetic:
            cmd.append("--synthetic")
        run_checked(cmd)

    if not args.skip_npm_install:
        ensure_node_modules()

    print_manifest_summary(manifest_path)

    api_port = find_port(args.api_port)
    web_port = find_port(args.port)
    env = os.environ.copy()
    env["VITE_API_BASE"] = f"http://127.0.0.1:{api_port}"

    api_proc = subprocess.Popen(
        [sys.executable, "server.py", "--host", "127.0.0.1", "--port", str(api_port)],
        cwd=ROOT,
    )
    web_proc = subprocess.Popen(
        ["npm", "run", "dev", "--", "--host", "127.0.0.1", "--port", str(web_port)],
        cwd=ROOT,
        env=env,
    )

    url = f"http://localhost:{web_port}"
    print(f"\nViewer running at {url}", flush=True)
    print("Press Ctrl-C to stop the API and Vite servers.", flush=True)

    procs = [api_proc, web_proc]
    try:
        while all(proc.poll() is None for proc in procs):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping servers...", flush=True)
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
