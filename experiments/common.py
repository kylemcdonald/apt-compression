"""Shared infrastructure for APT compression experiments.

Data model
----------
A dataset is an (N, 4) float32 array: x, y, z (nm), mass (Da), read from a
big-endian .POS file. A ranging table maps mass windows to species labels
(from apt-analysis custom ranging). Species index 0 is always "unranged".

Codec contract
--------------
Every codec writes one artifact directory containing arbitrary files and a
manifest.json, and can decode that directory back into an (M, 4) float32
point cloud with M approximately equal to the original atom count. Metrics
are computed uniformly on decoded points so density-based and point-based
codecs are directly comparable.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import zstandard

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = Path("/Users/kyle/Documents/GitHub/uap/uap-materials-article/src/data")
POS_PATH = DATA_DIR / "86a2fa56-8593-4856-bd42-b73716197abf.POS"
APT_OUTPUTS = Path("/Users/kyle/Documents/GitHub/uap/apt-analysis/outputs")
EXP_DIR = REPO / "experiments"
ART_DIR = EXP_DIR / "artifacts"

SPECTRUM_BIN_DA = 0.01
SPECTRUM_MAX_DA = 120.0


def dataset_slug(pos_path: Path) -> str:
    """POS filename -> apt-analysis output slug ('Sample 1- POS file' ->
    'sample-1-pos-file')."""
    import re

    stem = pos_path.name
    if stem.upper().endswith(".POS"):
        stem = stem[:-4]
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", stem).strip("-").lower()
    return slug


def ranging_csv_for(pos_path: Path) -> Path:
    return APT_OUTPUTS / dataset_slug(pos_path) / "custom" / "peaks_summary.csv"


def load_pos(path: Path = POS_PATH) -> np.ndarray:
    size = path.stat().st_size
    assert size % 16 == 0
    mm = np.memmap(path, dtype=">f4", mode="r", shape=(size // 16, 4))
    return np.asarray(mm, dtype=np.float32)


@dataclass
class Ranging:
    labels: list[str]            # labels[0] == "unranged"
    elements: list[str]          # primary element per species ("" for unranged)
    lo: np.ndarray               # (S,) window starts, Da
    hi: np.ndarray               # (S,) window ends, Da
    species_of_window: np.ndarray  # (S,) species index per window

    def assign(self, mass: np.ndarray) -> np.ndarray:
        """Return int16 species index per atom (0 = unranged)."""
        out = np.zeros(len(mass), dtype=np.int16)
        for w in range(len(self.lo)):
            sel = (mass >= self.lo[w]) & (mass < self.hi[w])
            out[sel] = self.species_of_window[w]
        return out


ATOMIC_WEIGHTS = {
    "H": 1, "He": 4, "Li": 7, "Be": 9, "B": 11, "C": 12, "N": 14, "O": 16,
    "F": 19, "Na": 23, "Mg": 24, "Al": 27, "Si": 28, "P": 31, "S": 32,
    "Cl": 35, "K": 39, "Ca": 40, "Sc": 45, "Ti": 48, "V": 51, "Cr": 52,
    "Mn": 55, "Fe": 56, "Co": 59, "Ni": 59, "Cu": 64, "Zn": 65, "Ga": 70,
    "Ge": 73, "As": 75, "Se": 79, "Br": 80, "Zr": 91, "Nb": 93, "Mo": 96,
    "Pd": 106, "Ag": 108, "Cd": 112, "Sn": 119, "Sb": 122, "Te": 128,
    "I": 127, "Ba": 137, "W": 184, "Pt": 195, "Au": 197, "Pb": 207, "Bi": 209,
}


def primary_element(species_label: str) -> str:
    """Heaviest element mentioned in a species label like 'AlH2+' -> 'Al'."""
    import re

    toks = re.findall(r"[A-Z][a-z]?", species_label)
    toks = [t for t in toks if t in ATOMIC_WEIGHTS]
    if not toks:
        return ""
    return max(toks, key=lambda t: ATOMIC_WEIGHTS[t])


def load_ranging(csv_path: Path | None = None, extra: list | None = None,
                 pos_path: Path | None = None) -> Ranging:
    """Build a ranging table from apt-analysis peaks_summary.csv.

    Windows with the same species label share one species index. `extra`
    allows appending windows missing from the CSV, e.g. the 71Ga+ peak:
    [(lo, hi, label), ...].
    """
    if csv_path is None:
        csv_path = ranging_csv_for(pos_path or POS_PATH)
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(
                (float(r["integration_left_da"]), float(r["integration_right_da"]), r["top_species"])
            )
    for e in extra or []:
        rows.append(e)
    rows.sort()
    labels = ["unranged"]
    elements = [""]
    lo, hi, sow = [], [], []
    for left, right, label in rows:
        if label not in labels:
            labels.append(label)
            elements.append(primary_element(label))
        lo.append(left)
        hi.append(right)
        sow.append(labels.index(label))
    return Ranging(
        labels=labels,
        elements=elements,
        lo=np.array(lo, dtype=np.float64),
        hi=np.array(hi, dtype=np.float64),
        species_of_window=np.array(sow, dtype=np.int16),
    )


def ranging_from_windows(windows: list) -> Ranging:
    """Build a chemistry-free Ranging from [(lo, hi), ...] mass windows."""
    lo = np.array([w[0] for w in windows], dtype=np.float64)
    hi = np.array([w[1] for w in windows], dtype=np.float64)
    labels = ["unranged"] + [
        f"peak_{0.5 * (a + b):.2f}Da" for a, b in zip(lo, hi)]
    return Ranging(
        labels=labels,
        elements=[""] * len(labels),
        lo=lo,
        hi=hi,
        species_of_window=np.arange(1, len(windows) + 1, dtype=np.int16),
    )


def split_runs_at_valleys(hist: np.ndarray, runs, z_split: float = 10.0,
                          smooth_sigma: float = 2.0) -> list:
    """Split histogram runs at statistically significant internal valleys.

    A huge peak's elevated tail can chain over and swallow a small
    neighboring peak, costing it its own window/category and exact storage.
    A valley splits when its dominance depth — min(max-left, max-right) -
    valley — exceeds z_split Poisson sigmas of the (smoothed) valley count.
    """
    from scipy.ndimage import gaussian_filter1d

    sm = gaussian_filter1d(np.asarray(hist, dtype=np.float64), smooth_sigma)
    out = []
    stack = [(int(a), int(b)) for a, b in runs]
    while stack:
        a, b = stack.pop()
        seg = sm[a:b]
        if len(seg) >= 8:
            lmax = np.maximum.accumulate(seg)
            rmax = np.maximum.accumulate(seg[::-1])[::-1]
            depth = np.minimum(lmax, rmax) - seg
            signif = depth / np.sqrt(seg + 1.0)
            i = int(np.argmax(signif))
            if signif[i] > z_split and 0 < i < len(seg) - 1:
                stack.append((a, a + i))
                stack.append((a + i, b))
                continue
        out.append((a, b))
    return sorted(out)


def auto_ranging(
    mass: np.ndarray,
    z_detect: float = 8.0,
    z_extend: float = 0.5,
    bg_halfwidth_da: float = 0.5,
    merge_gap_da: float = 0.05,
    pad_da: float = 0.08,
    z_split: float = 10.0,
    min_peak_atoms: int = 200,
    max_windows: int = 400,
) -> Ranging:
    """Chemistry-agnostic ranging: peak windows detected from the spectrum.

    A window is a contiguous m/z interval whose counts rise z_extend sigma
    above a rolling-percentile background and which contains at least one bin
    z_detect sigma above it, padded outward by pad_da. Windows are cut
    generously on purpose: background atoms mixed into a peak category are
    harmless, but peak shoulders left in the background category decouple
    those atoms' mass from their position. No species identification is
    attempted — a peak is a peak; everything else is species 0 (background).
    """
    from scipy.ndimage import percentile_filter

    hist = fine_spectrum(mass)
    nb = len(hist)
    hw = max(int(round(bg_halfwidth_da / SPECTRUM_BIN_DA)), 1)
    bg = percentile_filter(hist, 20, size=2 * hw + 1, mode="nearest")
    noise = np.sqrt(bg + 1.0)
    sig = hist - bg
    extend = sig > z_extend * noise
    detect = sig > z_detect * noise

    # contiguous extend-runs that contain at least one detect bin
    edges = np.diff(extend.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)  # exclusive
    pad = max(int(round(pad_da / SPECTRUM_BIN_DA)), 0)
    gap_bins = max(int(round(merge_gap_da / SPECTRUM_BIN_DA)), 1)
    runs = []
    for a, b in zip(starts, ends):
        if not detect[a:b].any():
            continue
        if runs and a - runs[-1][1] <= gap_bins:
            runs[-1][1] = b
        else:
            runs.append([a, b])
    runs = split_runs_at_valleys(hist, runs, z_split)
    # Pad outward, but never across the count valley between neighboring
    # runs: merging two distinct peaks into one window would let an abundant
    # peak absorb a rare one, costing the rare peak its exact storage and its
    # own spatial field.
    wins = []
    for i, (a, b) in enumerate(runs):
        left = 0
        if i > 0:
            pb = runs[i - 1][1]
            left = pb + int(np.argmin(hist[pb:a + 1])) if a > pb else pb
        right = nb
        if i + 1 < len(runs):
            na = runs[i + 1][0]
            right = b + int(np.argmin(hist[b:na + 1])) if na > b else na
        wins.append((max(a - pad, left), min(b + pad, right)))
    wins = [(a, b) for a, b in wins if b > a and hist[a:b].sum() >= min_peak_atoms]
    if len(wins) > max_windows:
        wins = sorted(wins, key=lambda w: -hist[w[0]:w[1]].sum())[:max_windows]
        wins.sort()
    return ranging_from_windows(
        [(a * SPECTRUM_BIN_DA, b * SPECTRUM_BIN_DA) for a, b in wins])


def write_artifact_dataset(outdir: Path, pos_path: Path) -> None:
    (outdir / "dataset.json").write_text(json.dumps({
        "pos": str(pos_path), "ranging_csv": str(ranging_csv_for(pos_path)),
    }))


def load_ranging_for_artifact(outdir: Path) -> Ranging:
    p = outdir / "dataset.json"
    if p.exists():
        d = json.loads(p.read_text())
        return load_ranging(Path(d["ranging_csv"]))
    return load_ranging()


# --- container helpers -------------------------------------------------------

_cctx = zstandard.ZstdCompressor(level=19)
_dctx = zstandard.ZstdDecompressor()


def zsave(path: Path, **arrays: np.ndarray) -> int:
    """Save arrays as a zstd-compressed npz-like blob. Returns bytes written."""
    import io

    buf = io.BytesIO()
    np.savez(buf, **arrays)
    payload = _cctx.compress(buf.getvalue())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return len(payload)


def zload(path: Path) -> dict[str, np.ndarray]:
    import io

    raw = _dctx.decompress(path.read_bytes())
    with np.load(io.BytesIO(raw), allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def zbytes(data: bytes) -> bytes:
    return _cctx.compress(data)


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


# --- density / spectrum helpers ---------------------------------------------


@dataclass
class GridSpec:
    origin: np.ndarray  # (3,)
    voxel: float
    shape: tuple[int, int, int]

    @classmethod
    def for_bounds(cls, bounds: np.ndarray, voxel: float) -> "GridSpec":
        extent = bounds[1] - bounds[0]
        shape = tuple(int(np.ceil(e / voxel)) + 1 for e in extent)
        return cls(origin=bounds[0].copy(), voxel=voxel, shape=shape)

    def indices(self, xyz: np.ndarray) -> np.ndarray:
        idx = np.floor((xyz - self.origin) / self.voxel).astype(np.int32)
        return np.clip(idx, 0, np.array(self.shape) - 1)

    def flat(self, xyz: np.ndarray) -> np.ndarray:
        i = self.indices(xyz)
        nx, ny, nz = self.shape
        return (i[:, 2].astype(np.int64) * ny + i[:, 1]) * nx + i[:, 0]

    @property
    def ncells(self) -> int:
        nx, ny, nz = self.shape
        return nx * ny * nz


def density_grid(xyz: np.ndarray, spec: GridSpec) -> np.ndarray:
    flat = spec.flat(xyz)
    counts = np.bincount(flat, minlength=spec.ncells)
    nx, ny, nz = spec.shape
    return counts.reshape(nz, ny, nx).astype(np.float32)


def gaussian_smooth(vol: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(vol.astype(np.float32), sigma=sigma)


def fine_spectrum(mass: np.ndarray) -> np.ndarray:
    nbins = int(round(SPECTRUM_MAX_DA / SPECTRUM_BIN_DA))
    h, _ = np.histogram(mass, bins=nbins, range=(0.0, SPECTRUM_MAX_DA))
    return h.astype(np.float64)


# --- metrics -----------------------------------------------------------------


def tv_error(a: np.ndarray, b: np.ndarray) -> float:
    """Total-variation distance between two nonnegative profiles (0..1)."""
    sa, sb = float(a.sum()), float(b.sum())
    if sa <= 0 and sb <= 0:
        return 0.0
    if sa <= 0 or sb <= 0:
        return 1.0
    return float(0.5 * np.abs(a / sa - b / sb).sum())


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom <= 0:
        return 0.0
    return float((a * b).sum() / denom)


@dataclass
class Reference:
    points: np.ndarray
    species: np.ndarray
    ranging: Ranging
    bounds: np.ndarray
    spectrum: np.ndarray
    elements: list = field(default_factory=list)        # elements with atoms
    rare_elements: list = field(default_factory=list)   # share < 1%
    voxel: dict = field(default_factory=dict)
    sigma: dict = field(default_factory=dict)
    total_grid2: np.ndarray = field(default=None)       # 2 nm total density
    total_grid1: np.ndarray = field(default=None)       # 1 nm, lightly smoothed
    grids: dict = field(default_factory=dict)           # element -> smoothed grid
    rare_proj: dict = field(default_factory=dict)       # el -> axis -> projection
    mass_band_edges: np.ndarray = field(default=None)   # viewer-palette bands, Da
    mass_band_proj: dict = field(default_factory=dict)  # band -> axis -> projection

    RARE_SHARE = 0.01
    RARE_MIN_COUNT = 3000

    @classmethod
    def build(cls, points: np.ndarray, ranging: Ranging) -> "Reference":
        species = ranging.assign(points[:, 3])
        bounds = np.stack([points[:, :3].min(axis=0), points[:, :3].max(axis=0)])
        ref = cls(
            points=points,
            species=species,
            ranging=ranging,
            bounds=bounds,
            spectrum=fine_spectrum(points[:, 3]),
        )
        n = len(points)
        counts = {}
        for el in sorted(set(e for e in ranging.elements if e)):
            c = int(ref.element_mask(species, el).sum())
            if c > 0:
                counts[el] = c
        ref.elements = sorted(counts, key=lambda e: -counts[e])
        ref.rare_elements = [
            el for el in ref.elements
            if counts[el] / n < cls.RARE_SHARE and counts[el] >= cls.RARE_MIN_COUNT
        ]
        for el in ref.elements:
            share = counts[el] / n
            ref.voxel[el] = 2.0 if share >= 0.05 else 3.0
            ref.sigma[el] = 0.5 if share >= 0.05 else 1.0
        ref.total_grid2 = density_grid(points[:, :3], GridSpec.for_bounds(bounds, 2.0))
        # 1 nm view with light smoothing: the 2 nm metric saturates near 0.996
        # for every grid codec while codecs differ visibly below 2 nm (edge
        # sharpness, striations, poles). Exact-point codecs land near 1 here;
        # distribution codecs are capped by the Poisson noise they resample.
        ref.total_grid1 = gaussian_smooth(
            density_grid(points[:, :3], GridSpec.for_bounds(bounds, 1.0)), 0.5)
        for el in ref.elements:
            ref.grids[el] = ref._element_grid(points, species, el)
        for el in ref.rare_elements:
            ref.rare_proj[el] = ref._projections(points, species, el)
        # The viewer colors mass with six equal palette segments.  Species-only
        # metrics cannot detect mass/position decorrelation inside the broad
        # "unranged" population, which is exactly what makes a localized red
        # inclusion disappear while all elemental scores still look good.
        display_max = min(
            SPECTRUM_MAX_DA,
            float(np.ceil(ranging.hi.max() + 12.0)) if len(ranging.hi)
            else SPECTRUM_MAX_DA,
        )
        ref.mass_band_edges = np.linspace(0.0, display_max, 7)
        for band in range(6):
            mask = ref._mass_band_mask(points[:, 3], band)
            if int(mask.sum()) >= cls.RARE_MIN_COUNT:
                ref.mass_band_proj[band] = ref._mass_projections(points[mask, :3])
        return ref

    def element_mask(self, species: np.ndarray, el: str) -> np.ndarray:
        idxs = [i for i, e in enumerate(self.ranging.elements) if e == el]
        return np.isin(species, idxs)

    def _element_grid(self, points, species, el) -> np.ndarray:
        mask = self.element_mask(species, el)
        spec = GridSpec.for_bounds(self.bounds, self.voxel[el])
        g = density_grid(points[mask, :3], spec)
        return gaussian_smooth(g, self.sigma[el])

    def _projections(self, points, species, el) -> dict:
        mask = self.element_mask(species, el)
        xyz = points[mask, :3]
        spec = GridSpec.for_bounds(self.bounds, 1.0)
        g = density_grid(xyz, spec)  # (nz, ny, nx)
        return {
            "z": gaussian_smooth(g.sum(axis=0), 1.5),
            "y": gaussian_smooth(g.sum(axis=1), 1.5),
            "x": gaussian_smooth(g.sum(axis=2), 1.5),
        }

    def _mass_band_mask(self, mass: np.ndarray, band: int) -> np.ndarray:
        lo = self.mass_band_edges[band]
        if band == 5:
            # The shader clamps all masses beyond display_max to the final
            # (red) palette segment, so the metric must do the same.
            return mass >= lo
        return (mass >= lo) & (mass < self.mass_band_edges[band + 1])

    def _mass_projections(self, xyz: np.ndarray) -> dict:
        spec = GridSpec.for_bounds(self.bounds, 1.0)
        g = density_grid(xyz, spec)
        return {
            "z": gaussian_smooth(g.sum(axis=0), 1.5),
            "y": gaussian_smooth(g.sum(axis=1), 1.5),
            "x": gaussian_smooth(g.sum(axis=2), 1.5),
        }

    def evaluate(self, decoded: np.ndarray) -> dict:
        """Metrics for a decoded (M,4) point cloud against this reference."""
        t0 = time.time()
        dspecies = self.ranging.assign(decoded[:, 3])
        out = {}
        out["spectrum_tv"] = tv_error(self.spectrum, fine_spectrum(decoded[:, 3]))
        g2 = density_grid(decoded[:, :3], GridSpec.for_bounds(self.bounds, 2.0))
        scale = len(self.points) / max(len(decoded), 1)
        out["density_tv_2nm"] = tv_error(self.total_grid2, g2)
        out["density_corr_2nm"] = pearson(self.total_grid2, g2 * scale)
        g1 = gaussian_smooth(
            density_grid(decoded[:, :3], GridSpec.for_bounds(self.bounds, 1.0)), 0.5)
        out["density_corr_1nm"] = pearson(self.total_grid1, g1 * scale)
        out["density_tv_1nm"] = tv_error(self.total_grid1, g1)
        del g1
        for el in self.elements:
            mask = self.element_mask(dspecies, el)
            spec = GridSpec.for_bounds(self.bounds, self.voxel[el])
            g = gaussian_smooth(density_grid(decoded[mask, :3], spec), self.sigma[el])
            out[f"{el}_density_corr"] = pearson(self.grids[el], g)
            out[f"{el}_density_tv"] = tv_error(self.grids[el], g)
            out[f"{el}_count_ratio"] = float(mask.sum() * scale) / max(
                float(self.element_mask(self.species, el).sum()), 1.0
            )
        # rare-element structure: correlation of smoothed 2D projections
        rare_proj_min = None
        for el in self.rare_elements:
            mask = self.element_mask(dspecies, el)
            xyz = decoded[mask, :3]
            spec = GridSpec.for_bounds(self.bounds, 1.0)
            g = density_grid(xyz, spec)
            worst = None
            for ax_name, ax in [("z", 0), ("y", 1), ("x", 2)]:
                proj = gaussian_smooth(g.sum(axis=ax), 1.5)
                c = pearson(self.rare_proj[el][ax_name], proj)
                out[f"{el}_proj_corr_{ax_name}"] = c
                worst = c if worst is None else min(worst, c)
            if worst is not None:
                rare_proj_min = worst if rare_proj_min is None else min(rare_proj_min, worst)
        if self.rare_elements:
            out["rare_min_density_corr"] = min(
                out[f"{el}_density_corr"] for el in self.rare_elements)
            out["rare_min_proj_corr"] = rare_proj_min
        # Preserve the spatial structure carried by each mass/color band,
        # including mass variation within species 0 (unranged).  This catches
        # codecs that reproduce the spectrum and geometry independently but
        # destroy their correlation.
        mass_band_min = None
        for band, raw_proj in self.mass_band_proj.items():
            mask = self._mass_band_mask(decoded[:, 3], band)
            rec_proj = self._mass_projections(decoded[mask, :3])
            band_worst = None
            for ax_name in ("z", "y", "x"):
                c = pearson(raw_proj[ax_name], rec_proj[ax_name])
                out[f"mass_band_{band}_proj_corr_{ax_name}"] = c
                band_worst = c if band_worst is None else min(band_worst, c)
            out[f"mass_band_{band}_proj_corr"] = band_worst
            mass_band_min = (
                band_worst if mass_band_min is None
                else min(mass_band_min, band_worst)
            )
        if mass_band_min is not None:
            out["mass_band_min_proj_corr"] = mass_band_min
        out["spatial_species_error"] = self._spatial_species_error(decoded, dspecies)
        out["metric_seconds"] = time.time() - t0
        return out

    def _spatial_species_error(self, decoded: np.ndarray, dspecies: np.ndarray) -> float:
        s = 8
        nspecies = len(self.ranging.labels)
        extent = np.maximum(self.bounds[1] - self.bounds[0], 1e-6)

        def cell_hist(points, species):
            norm = np.clip((points[:, :3] - self.bounds[0]) / extent, 0.0, 0.999999)
            key = (
                (norm[:, 2] * s).astype(np.int64) * s + (norm[:, 1] * s).astype(np.int64)
            ) * s + (norm[:, 0] * s).astype(np.int64)
            joint = key * nspecies + species
            return np.bincount(joint, minlength=s * s * s * nspecies).reshape(
                s * s * s, nspecies
            ).astype(np.float64)

        raw = cell_hist(self.points, self.species)
        rec = cell_hist(decoded, dspecies)
        raw_tot = raw.sum(axis=1)
        rec_tot = rec.sum(axis=1)
        active = raw_tot > 0
        raw_d = raw[active] / raw_tot[active, None]
        rec_d = np.divide(
            rec[active], rec_tot[active, None], out=np.zeros_like(rec[active]),
            where=rec_tot[active, None] > 0,
        )
        w = raw_tot[active] / raw_tot[active].sum()
        return float((w * (0.5 * np.abs(raw_d - rec_d).sum(axis=1))).sum())


# --- decoded-point synthesis helpers ----------------------------------------


def sample_points_from_grid(
    grid: np.ndarray, spec: GridSpec, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw n points from a nonnegative density grid with linear interpolation.

    Voxels are drawn by multinomial; within-voxel offsets use a triangular
    (tent) kernel: uniform-in-voxel (box) convolved with a second box. The
    resulting samples follow the piecewise-LINEAR interpolation of the voxel
    densities rather than the blocky piecewise-constant histogram, so there
    are no jaggies at grid-cell boundaries.
    """
    flat = grid.ravel().astype(np.float64)
    total = flat.sum()
    if total <= 0 or n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    counts = rng.multinomial(n, flat / total)
    nz_idx = np.nonzero(counts)[0]
    reps = counts[nz_idx]
    cells = np.repeat(nz_idx, reps)
    nx, ny = spec.shape[0], spec.shape[1]
    iz, rem = np.divmod(cells, nx * ny)
    iy, ix = np.divmod(rem, nx)
    base = np.stack([ix, iy, iz], axis=1).astype(np.float32)
    tent = (rng.random((len(cells), 3), dtype=np.float32)
            + rng.random((len(cells), 3), dtype=np.float32) - 0.5)
    pos = spec.origin.astype(np.float32) + (base + tent) * spec.voxel
    hi = spec.origin.astype(np.float32) + (
        np.array(spec.shape, dtype=np.float32) * spec.voxel)
    return np.clip(pos, spec.origin.astype(np.float32), hi - 1e-4)


def sample_mass_from_hist(
    hist: np.ndarray, edges_lo: float, bin_w: float, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw n masses from a binned spectrum, uniform within bins."""
    p = hist.astype(np.float64)
    total = p.sum()
    if total <= 0 or n <= 0:
        return np.full(n, -1.0, dtype=np.float32)
    counts = rng.multinomial(n, p / total)
    nz = np.nonzero(counts)[0]
    bins = np.repeat(nz, counts[nz])
    vals = edges_lo + (bins + rng.random(len(bins))) * bin_w
    out = vals.astype(np.float32)
    rng.shuffle(out)
    return out


def result_row(name: str, size_bytes: int, enc_s: float, dec_s: float, metrics: dict) -> dict:
    return {
        "method": name,
        "size_bytes": int(size_bytes),
        "size_mb": round(size_bytes / 1e6, 3),
        "ratio_vs_raw": round(POS_PATH.stat().st_size / max(size_bytes, 1), 1),
        "encode_seconds": round(enc_s, 2),
        "decode_seconds": round(dec_s, 2),
        **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in metrics.items()},
    }
