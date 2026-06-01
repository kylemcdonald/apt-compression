from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

import preprocess as pp


def load_reference(ds: dict[str, Any], ds_dir: Path) -> pp.ReferenceStats:
    with np.load(ds_dir / "reference_stats.npz") as data:
        return pp.ReferenceStats(
            atom_count=int(ds["atom_count"]),
            raw_size_bytes=int(ds["raw_size_bytes"]),
            bounds=data["bounds"].astype(np.float32),
            mass_range=data["mass_range"].astype(np.float32),
            spectrum_edges=data["spectrum_edges"].astype(np.float32),
            spectrum_counts=data["spectrum_counts"].astype(np.float64),
            z_counts=data["z_counts"].astype(np.float64),
            radial_counts=data["radial_counts"].astype(np.float64),
            local_counts=data["local_counts"].astype(np.float64),
            spatial_bins=int(data["spatial_bins"][0]),
            metric_mass_bins=int(data["metric_mass_bins"][0]),
        )


def train_hypernetwork(masses: np.ndarray, teacher_matrix: np.ndarray, args: Any, torch: Any, nn: Any, F: Any, device: str):
    mean = teacher_matrix.mean(axis=0).astype(np.float32)
    centered = teacher_matrix - mean[None, :]
    rank = min(args.hyper_rank, max(1, len(masses) - 1), centered.shape[0])
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:rank].astype(np.float32)
    coeff = centered @ basis.T
    coeff_mean = coeff.mean(axis=0).astype(np.float32)
    coeff_std = np.maximum(coeff.std(axis=0).astype(np.float32), 1e-6)
    coeff_norm = (coeff - coeff_mean[None, :]) / coeff_std[None, :]

    mass_min = float(masses.min())
    mass_max = float(masses.max())
    mass_norm = ((masses - mass_min) / max(mass_max - mass_min, 1e-6) * 2.0 - 1.0).astype(np.float32)

    class CoeffNet(nn.Module):
        def __init__(self, output_dim: int):
            super().__init__()
            self.register_buffer("freq", torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0, 32.0], dtype=torch.float32), persistent=False)
            in_dim = 1 + 2 * len(self.freq)
            self.net = nn.Sequential(nn.Linear(in_dim, 96), nn.SiLU(), nn.Linear(96, 96), nn.SiLU(), nn.Linear(96, output_dim))

        def encode(self, x):
            xb = x[:, None] * self.freq.to(x.device)[None, :]
            return torch.cat([x[:, None], torch.sin(math.pi * xb), torch.cos(math.pi * xb)], dim=1)

        def forward(self, x):
            return self.net(self.encode(x))

    model = CoeffNet(rank).to(device)
    x_t = torch.from_numpy(mass_norm).to(device)
    y_t = torch.from_numpy(coeff_norm.astype(np.float32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-6)
    for _ in range(args.hyper_epochs):
        pred = model(x_t)
        loss = F.mse_loss(pred, y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred_norm = model(x_t).cpu().numpy()
    predicted_coeff = pred_norm * coeff_std[None, :] + coeff_mean[None, :]
    predicted_weights = mean[None, :] + predicted_coeff @ basis
    rel_err = np.linalg.norm(predicted_weights - teacher_matrix) / max(np.linalg.norm(teacher_matrix - mean[None, :]), 1e-6)
    return model, mean, basis, coeff_mean, coeff_std, predicted_weights.astype(np.float32), {
        "rank": rank,
        "mass_min": mass_min,
        "mass_max": mass_max,
        "relative_teacher_weight_error": float(rel_err),
    }
