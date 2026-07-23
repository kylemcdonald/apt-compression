"""Sanity checks: spectrum around Ga peaks, Ga wire projections, sizes."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common


def main():
    pts = common.load_pos()
    print(f"atoms: {len(pts):,}")
    print("bounds:", pts[:, :3].min(axis=0), pts[:, :3].max(axis=0))
    mass = pts[:, 3]

    # check candidate Ga peaks: 69Ga+ 68.9256, 71Ga+ 70.9247, 69Ga2+ 34.4628, 71Ga2+ 35.4624
    for center in (68.9256, 70.9247, 34.4628, 35.4624):
        w = 0.15
        n = int(((mass > center - w) & (mass < center + w)).sum())
        bg_lo = int(((mass > center - 3 * w) & (mass < center - w)).sum()) / 2
        print(f"peak {center:8.4f}: counts in +-{w} = {n:7d}   sideband/2 = {bg_lo:9.1f}")

    rng_table = common.load_ranging(extra=[(70.75, 71.10, "Ga+")])
    print("species:", list(zip(rng_table.labels, rng_table.elements)))
    species = rng_table.assign(mass)
    counts = np.bincount(species, minlength=len(rng_table.labels))
    for lab, el, c in zip(rng_table.labels, rng_table.elements, counts):
        print(f"  {lab:10s} {el:3s} {c:9,d}  ({100*c/len(pts):.3f}%)")

    # Ga projections to confirm the wire
    ga_mask = np.zeros(len(pts), dtype=bool)
    for i, el in enumerate(rng_table.elements):
        if el == "Ga":
            ga_mask |= species == i
    ga = pts[ga_mask]
    print(f"Ga atoms: {len(ga):,}")
    bounds = np.stack([pts[:, :3].min(axis=0), pts[:, :3].max(axis=0)])
    spec = common.GridSpec.for_bounds(bounds, 1.0)
    g = common.density_grid(ga[:, :3], spec)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, (name, axis) in zip(axes, [("along z (xy)", 0), ("along y (xz)", 1), ("along x (yz)", 2)]):
        proj = common.gaussian_smooth(g.sum(axis=axis), 1.5)
        ax.imshow(proj, origin="lower", cmap="magma", aspect="auto")
        ax.set_title(f"Ga projection {name}")
    fig.tight_layout()
    out = Path(__file__).parent / "artifacts" / "explore_ga_projections.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print("wrote", out)

    # total density projection for comparison
    gall = common.density_grid(pts[:, :3], spec)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, (name, axis) in zip(axes, [("along z", 0), ("along y", 1), ("along x", 2)]):
        proj = gall.sum(axis=axis)
        ax.imshow(np.log1p(proj), origin="lower", cmap="viridis", aspect="auto")
        ax.set_title(f"total log-density {name}")
    fig.tight_layout()
    out2 = Path(__file__).parent / "artifacts" / "explore_total_projections.png"
    fig.savefig(out2, dpi=110)
    print("wrote", out2)


if __name__ == "__main__":
    main()
