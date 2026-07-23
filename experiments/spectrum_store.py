"""Global fine spectrum storage shared by density-based codecs.

Key observation: peaks never shift in m/z, so the global spectrum can be
stored once, finely and near-losslessly, for ~100 KB. Density codecs then
only need to model *where* each species is, and synthesize per-atom masses
from the global spectrum conditioned on species windows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import common
from common import Ranging

BIN = common.SPECTRUM_BIN_DA
MAX = common.SPECTRUM_MAX_DA
NBINS = int(round(MAX / BIN))
TAIL_SCALE = 40.0  # 0.025 Da steps for the >120 Da tail


def encode(mass: np.ndarray, path: Path) -> int:
    hist, _ = np.histogram(mass, bins=NBINS, range=(0.0, MAX))
    tail = mass[mass >= MAX]
    tail_q = np.round((np.sort(tail) - MAX) * TAIL_SCALE).astype(np.uint16)
    return common.zsave(path, hist=hist.astype(np.uint32), tail=tail_q)


class SpectrumStore:
    def __init__(self, path: Path, ranging: Ranging):
        d = common.zload(path)
        self.hist = d["hist"].astype(np.float64)
        self.tail = d["tail"].astype(np.float64) / TAIL_SCALE + MAX
        self.ranging = ranging
        centers = (np.arange(NBINS) + 0.5) * BIN
        # bin -> species window ownership (0 = unranged)
        self.bin_species = np.zeros(NBINS, dtype=np.int16)
        for w in range(len(ranging.lo)):
            sel = (centers >= ranging.lo[w]) & (centers < ranging.hi[w])
            self.bin_species[sel] = ranging.species_of_window[w]

    def species_hist(self, s: int) -> np.ndarray:
        h = np.where(self.bin_species == s, self.hist, 0.0)
        return h

    def sample_for_species(self, s: int, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n masses for species index s (0 = unranged incl. >MAX tail)."""
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        h = self.species_hist(s)
        if s == 0 and len(self.tail):
            frac_tail = len(self.tail) / max(len(self.tail) + h.sum(), 1.0)
            n_tail = int(round(n * frac_tail))
            idx = rng.integers(0, len(self.tail), n_tail)
            tail_masses = self.tail[idx].astype(np.float32)
            body = common.sample_mass_from_hist(h, 0.0, BIN, n - n_tail, rng)
            out = np.concatenate([body, tail_masses])
            rng.shuffle(out)
            return out
        return common.sample_mass_from_hist(h, 0.0, BIN, n, rng)

    def sample_for_species_band(
        self,
        s: int,
        band: int,
        band_count: int,
        display_mass_max: float,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw n masses for species s restricted to one viewer palette band.

        The final band absorbs everything >= its lower edge (the shader clamps
        high masses to the last palette segment), including the >MAX tail.
        """
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        lo = display_mass_max * band / band_count
        hi = display_mass_max * (band + 1) / band_count
        centers = (np.arange(len(self.hist)) + 0.5) * BIN
        allowed = centers >= lo
        if band < band_count - 1:
            allowed &= centers < hi
        h = np.where(allowed, self.species_hist(s), 0.0)
        tail = np.zeros(0, dtype=np.float64)
        if s == 0 and band == band_count - 1 and len(self.tail):
            tail = self.tail[self.tail >= lo]
        total = float(h.sum() + len(tail))
        if total <= 0:
            return self.sample_for_species(s, n, rng)
        n_tail = int(round(n * len(tail) / total))
        body = common.sample_mass_from_hist(h, 0.0, BIN, n - n_tail, rng)
        if n_tail:
            tail_mass = tail[rng.integers(0, len(tail), n_tail)].astype(np.float32)
            body = np.concatenate([body, tail_mass])
            rng.shuffle(body)
        return body.astype(np.float32)
