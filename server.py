from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run preprocess.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def decode_point_pack(path: Path) -> np.ndarray:
    with np.load(path) as data:
        xyz_q = data["xyz"].astype(np.float32)
        mass_q = data["mass"].astype(np.float32)
        bounds = data["bounds"].astype(np.float32)
        mass_range = data["mass_range"].astype(np.float32)
    xyz = bounds[0] + (xyz_q / 65535.0) * (bounds[1] - bounds[0])
    mass = mass_range[0] + (mass_q / 65535.0) * (mass_range[1] - mass_range[0])
    return np.column_stack([xyz, mass]).astype(np.float32)


def downsample(points: np.ndarray, budget: int) -> np.ndarray:
    if budget <= 0 or len(points) == 0:
        return np.empty((0, 4), dtype=np.float32)
    if len(points) <= budget:
        return points.astype(np.float32, copy=False)
    idx = np.linspace(0, len(points) - 1, budget, dtype=np.int64)
    return points[idx].astype(np.float32, copy=False)


def read_raw_points(path: Path, budget: int) -> np.ndarray:
    size = path.stat().st_size
    if size % 16 != 0:
        raise ValueError(f"{path} is not a valid 16-byte-record POS file")
    count = size // 16
    budget = min(max(int(budget), 0), count)
    if budget == 0:
        return np.empty((0, 4), dtype=np.float32)
    mm = np.memmap(path, dtype=">f4", mode="r", shape=(count, 4))
    if budget >= count:
        return np.asarray(mm, dtype=np.float32)
    idx = np.linspace(0, count - 1, budget, dtype=np.int64)
    return np.asarray(mm[idx], dtype=np.float32)


def read_raw_range_points(path: Path, budget: int, mass_min: float, mass_max: float, chunk_atoms: int = 2_000_000) -> np.ndarray:
    size = path.stat().st_size
    if size % 16 != 0:
        raise ValueError(f"{path} is not a valid 16-byte-record POS file")
    count = size // 16
    if budget <= 0:
        return np.empty((0, 4), dtype=np.float32)
    mm = np.memmap(path, dtype=">f4", mode="r", shape=(count, 4))
    parts: list[np.ndarray] = []
    remaining = int(budget)
    for start in range(0, count, chunk_atoms):
        stop = min(count, start + chunk_atoms)
        chunk = np.asarray(mm[start:stop], dtype=np.float32)
        finite = np.isfinite(chunk).all(axis=1)
        if not finite.any():
            continue
        c = chunk[finite]
        keep = (c[:, 3] >= mass_min) & (c[:, 3] <= mass_max)
        if keep.any():
            selected = c[keep]
            if len(selected) > remaining:
                selected = selected[:remaining]
            parts.append(selected.astype(np.float32, copy=True))
            remaining -= len(selected)
            if remaining <= 0:
                break
    if not parts:
        return np.empty((0, 4), dtype=np.float32)
    return np.concatenate(parts, axis=0)


class Handler(SimpleHTTPRequestHandler):
    server_version = "APTCompressionHTTP/0.1"

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/manifest":
                self.send_json(self.server.manifest)
                return
            if path.startswith("/api/hypernetwork/"):
                self.handle_artifact(path, "/api/hypernetwork/<dataset>/<method>")
                return
            if path.startswith("/api/artifact/"):
                self.handle_artifact(path, "/api/artifact/<dataset>/<method>")
                return
            if path.startswith("/api/points/"):
                self.handle_points(path, urllib.parse.parse_qs(parsed.query))
                return
            self.serve_static(path)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            try:
                self.send_error_json(500, str(exc))
            except (BrokenPipeError, ConnectionResetError):
                return

    def send_json(self, data: Any) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status: int, message: str) -> None:
        body = json.dumps({"error": message}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_artifact(self, path: str, expected: str) -> None:
        parts = path.strip("/").split("/")
        if len(parts) != 4:
            self.send_error_json(404, f"Expected {expected}")
            return
        _, _, dataset_id, method = parts
        dataset = self.server.datasets.get(dataset_id)
        if dataset is None:
            self.send_error_json(404, f"Unknown dataset {dataset_id}")
            return
        method_info = dataset.get("methods", {}).get(method)
        if not method_info or not method_info.get("available"):
            self.send_error_json(404, f"Unknown or unavailable artifact method {method}")
            return
        artifact = method_info.get("artifact")
        if not artifact:
            self.send_error_json(404, f"{method} has no artifact")
            return
        target = Path(artifact)
        if not target.exists() or not target.is_file():
            self.send_error_json(404, f"Missing artifact {artifact}")
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_points(self, path: str, query: dict[str, list[str]]) -> None:
        parts = path.strip("/").split("/")
        if len(parts) != 4:
            self.send_error_json(404, "Expected /api/points/<dataset>/<method>")
            return
        _, _, dataset_id, method = parts
        budget = int(query.get("budget", ["1000000"])[0])
        dataset = self.server.datasets.get(dataset_id)
        if dataset is None:
            self.send_error_json(404, f"Unknown dataset {dataset_id}")
            return

        method_info = dataset.get("methods", {}).get(method, {})

        if method == "full":
            points = read_raw_points(Path(dataset["raw_path"]), budget)
        elif method_info.get("raw_range"):
            mass_min, mass_max = (float(value) for value in method_info["raw_range"])
            points = read_raw_range_points(Path(dataset["raw_path"]), budget, mass_min, mass_max)
        elif method_info.get("display_artifact"):
            if not method_info.get("available"):
                raise ValueError(method_info.get("notes", f"{method} unavailable"))
            points = decode_point_pack(Path(method_info["display_artifact"]))
            points = downsample(points, budget)
        else:
            self.send_error_json(404, f"Unknown method {method}")
            return

        arr = np.ascontiguousarray(points.astype("<f4", copy=False))
        body = memoryview(arr).cast("B")
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(body.nbytes))
        self.send_header("X-Point-Count", str(len(points)))
        self.end_headers()
        self.wfile.write(body)

    def serve_static(self, request_path: str) -> None:
        if request_path == "/":
            request_path = "/index.html"
        root = self.server.static_dir.resolve()
        target = (root / request_path.lstrip("/")).resolve()
        if not str(target).startswith(str(root)) or not target.exists() or not target.is_file():
            self.send_error(404)
            return
        body = target.read_bytes()
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class APTServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], handler, manifest: dict[str, Any], static_dir: Path):
        super().__init__(address, handler)
        self.manifest = manifest
        self.static_dir = static_dir
        self.datasets = {dataset["id"]: dataset for dataset in manifest.get("datasets", [])}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve APT compression artifacts.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--static-dir", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.artifacts / "manifest.json")
    server = APTServer((args.host, args.port), Handler, manifest, args.static_dir)
    print(f"API server running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
