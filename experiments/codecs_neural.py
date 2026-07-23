"""Neural backbones.

inr:   multiresolution hash-grid encoder + small MLP predicting log1p(count)
       per voxel (instant-ngp style), weights stored fp16.
hyper: per-z-slice latent codes -> hypernetwork -> weights of a tiny 2D MLP
       that predicts the slice's log1p density field.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np

import common

PRIMES = (1, 2654435761, 805459861)


def _device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --- hash-grid INR -----------------------------------------------------------


def _build_inr(levels, log2T, feat, n_min, n_max, hidden):
    import torch
    import torch.nn as nn

    class HashGrid(nn.Module):
        def __init__(self):
            super().__init__()
            b = (n_max / n_min) ** (1.0 / max(levels - 1, 1))
            self.res = [int(round(n_min * b**l)) for l in range(levels)]
            self.T = 2**log2T
            self.tables = nn.Parameter(
                torch.empty(levels, self.T, feat).uniform_(-1e-4, 1e-4)
            )

        def forward(self, x):  # x (B,3) in [0,1]
            feats = []
            for l, res in enumerate(self.res):
                pos = x * (res - 1)
                c0 = pos.floor().long()
                w = pos - c0.float()
                fl = 0.0
                for dx in (0, 1):
                    for dy in (0, 1):
                        for dz in (0, 1):
                            corner = c0 + torch.tensor(
                                [dx, dy, dz], device=x.device, dtype=torch.long
                            )
                            corner = corner.clamp(max=res - 1)
                            h = (
                                corner[:, 0] * PRIMES[0]
                                ^ corner[:, 1] * PRIMES[1]
                                ^ corner[:, 2] * PRIMES[2]
                            ) % self.T
                            wt = (
                                (w[:, 0] if dx else 1 - w[:, 0])
                                * (w[:, 1] if dy else 1 - w[:, 1])
                                * (w[:, 2] if dz else 1 - w[:, 2])
                            )
                            fl = fl + wt[:, None] * self.tables[l][h]
                feats.append(fl)
            return torch.cat(feats, dim=1)

    class INR(nn.Module):
        def __init__(self):
            super().__init__()
            self.grid = HashGrid()
            self.mlp = nn.Sequential(
                nn.Linear(levels * feat, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):
            return self.mlp(self.grid(x)).squeeze(-1)

    return INR()


def _voxel_coords(shape):
    nz, ny, nx = shape
    z = (np.arange(nz) + 0.5) / nz
    y = (np.arange(ny) + 0.5) / ny
    x = (np.arange(nx) + 0.5) / nx
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)


def encode_inr(grid: np.ndarray, path: Path, levels=8, log2T=15, feat=2,
               n_min=8, n_max=256, hidden=64, steps=3000, batch=65536, seed=0):
    import torch

    torch.manual_seed(seed)
    dev = _device()
    model = _build_inr(levels, log2T, feat, n_min, n_max, hidden).to(dev)
    target = np.log1p(grid.astype(np.float32)).ravel()
    coords = _voxel_coords(grid.shape)
    t_t = torch.from_numpy(target).to(dev)
    c_t = torch.from_numpy(coords).to(dev)
    nz_idx = torch.from_numpy(np.nonzero(target > 0)[0]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-4)
    n = len(target)
    t0 = time.time()
    for step in range(steps):
        half = batch // 2
        i_u = torch.randint(0, n, (half,), device=dev)
        i_n = nz_idx[torch.randint(0, len(nz_idx), (half,), device=dev)]
        idx = torch.cat([i_u, i_n])
        pred = model(c_t[idx])
        loss = torch.nn.functional.mse_loss(pred, t_t[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(f"  inr step {step} loss {loss.item():.5f} ({time.time()-t0:.0f}s)")
    state = {k: v.detach().cpu().numpy().astype(np.float16)
             for k, v in model.state_dict().items()}
    cfg = np.array([levels, log2T, feat, n_min, n_max, hidden], dtype=np.int64)
    common.zsave(path, cfg=cfg, grid_shape=np.array(grid.shape, dtype=np.int32),
                 **{f"w_{k.replace('.', '_')}": v for k, v in state.items()})
    # keep original key names for reload
    names = list(state.keys())
    common.zsave(path.with_suffix(".names"),
                 names=np.array(names, dtype=np.bytes_))
    return {}


def decode_inr(path: Path, shape) -> np.ndarray:
    import torch

    d = common.zload(path)
    cfg = d["cfg"]
    levels, log2T, feat, n_min, n_max, hidden = (int(v) for v in cfg)
    model = _build_inr(levels, log2T, feat, n_min, n_max, hidden)
    names = [n.decode() for n in common.zload(path.with_suffix(".names"))["names"]]
    state = {}
    for k in names:
        state[k] = torch.from_numpy(
            d[f"w_{k.replace('.', '_')}"].astype(np.float32))
    model.load_state_dict(state)
    dev = _device()
    model = model.to(dev).eval()
    grid_shape = tuple(d["grid_shape"])
    coords = _voxel_coords(grid_shape)
    out = np.zeros(len(coords), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(coords), 262144):
            c = torch.from_numpy(coords[i : i + 262144]).to(dev)
            out[i : i + 262144] = model(c).cpu().numpy()
    return np.clip(np.expm1(out.reshape(grid_shape)), 0.0, None)


# --- hypernetwork ------------------------------------------------------------

H1 = 32


def _target_param_count():
    return (2 * H1 + H1) + (H1 * H1 + H1) + (H1 + 1)


def encode_hyper(grid: np.ndarray, path: Path, latent=16, hyper_hidden=128,
                 steps=4000, slices_per_step=16, px_per_slice=4096, seed=0):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    dev = _device()
    nz, ny, nx = grid.shape
    n_target = _target_param_count()
    latents = nn.Parameter(torch.randn(nz, latent, device=dev) * 0.1)
    hyper = nn.Sequential(
        nn.Linear(latent, hyper_hidden), nn.ReLU(),
        nn.Linear(hyper_hidden, n_target),
    ).to(dev)
    target = torch.from_numpy(np.log1p(grid.astype(np.float32))).to(dev)
    xs = (torch.arange(nx, device=dev, dtype=torch.float32) + 0.5) / nx
    ys = (torch.arange(ny, device=dev, dtype=torch.float32) + 0.5) / ny
    opt = torch.optim.Adam([{"params": hyper.parameters()}, {"params": [latents]}],
                           lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-4)

    def target_forward(w, pts):  # w (S,P), pts (S,B,2)
        o = 0
        W1 = w[:, o : o + 2 * H1].reshape(-1, 2, H1); o += 2 * H1
        b1 = w[:, o : o + H1]; o += H1
        W2 = w[:, o : o + H1 * H1].reshape(-1, H1, H1); o += H1 * H1
        b2 = w[:, o : o + H1]; o += H1
        W3 = w[:, o : o + H1].reshape(-1, H1, 1); o += H1
        b3 = w[:, o : o + 1]
        h = torch.relu(torch.bmm(pts, W1) + b1[:, None, :])
        h = torch.relu(torch.bmm(h, W2) + b2[:, None, :])
        return (torch.bmm(h, W3) + b3[:, None, :]).squeeze(-1)

    t0 = time.time()
    for step in range(steps):
        sl = torch.randint(0, nz, (slices_per_step,), device=dev)
        ix = torch.randint(0, nx, (slices_per_step, px_per_slice), device=dev)
        iy = torch.randint(0, ny, (slices_per_step, px_per_slice), device=dev)
        pts = torch.stack([xs[ix], ys[iy]], dim=-1)
        w = hyper(latents[sl])
        pred = target_forward(w, pts)
        tgt = target[sl[:, None], iy, ix]
        loss = torch.nn.functional.mse_loss(pred, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(f"  hyper step {step} loss {loss.item():.5f} ({time.time()-t0:.0f}s)")

    state = {"latents": latents.detach().cpu().numpy().astype(np.float16)}
    for i, (k, v) in enumerate(hyper.state_dict().items()):
        state[f"h{i}"] = v.cpu().numpy().astype(np.float16)
    common.zsave(path, grid_shape=np.array(grid.shape, dtype=np.int32),
                 cfg=np.array([latent, hyper_hidden], dtype=np.int64), **state)
    return {}


def decode_hyper(path: Path, shape) -> np.ndarray:
    import torch
    import torch.nn as nn

    d = common.zload(path)
    latent, hyper_hidden = (int(v) for v in d["cfg"])
    nz, ny, nx = (int(v) for v in d["grid_shape"])
    n_target = _target_param_count()
    hyper = nn.Sequential(
        nn.Linear(latent, hyper_hidden), nn.ReLU(),
        nn.Linear(hyper_hidden, n_target),
    )
    sd = hyper.state_dict()
    for i, k in enumerate(sd.keys()):
        sd[k] = torch.from_numpy(d[f"h{i}"].astype(np.float32))
    hyper.load_state_dict(sd)
    latents = torch.from_numpy(d["latents"].astype(np.float32))
    xs = (torch.arange(nx, dtype=torch.float32) + 0.5) / nx
    ys = (torch.arange(ny, dtype=torch.float32) + 0.5) / ny
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pts = torch.stack([xx.ravel(), yy.ravel()], dim=-1)

    out = np.zeros((nz, ny, nx), dtype=np.float32)
    with torch.no_grad():
        w_all = hyper(latents)
        for z in range(nz):
            w = w_all[z : z + 1]
            o = 0
            W1 = w[:, o : o + 2 * H1].reshape(2, H1); o += 2 * H1
            b1 = w[:, o : o + H1]; o += H1
            W2 = w[:, o : o + H1 * H1].reshape(H1, H1); o += H1 * H1
            b2 = w[:, o : o + H1]; o += H1
            W3 = w[:, o : o + H1].reshape(H1, 1); o += H1
            b3 = w[:, o : o + 1]
            h = torch.relu(pts @ W1 + b1)
            h = torch.relu(h @ W2 + b2)
            pred = (h @ W3 + b3).squeeze(-1)
            out[z] = pred.reshape(ny, nx).numpy()
    return np.clip(np.expm1(out), 0.0, None)
