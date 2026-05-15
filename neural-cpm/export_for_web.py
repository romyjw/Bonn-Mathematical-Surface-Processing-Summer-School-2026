"""export_for_web.py

Runs the full neural-CPM pipeline on the Apple example and dumps compact
JSON files that the HTML demo (web/) can replay without any Python runtime.

Outputs in precomputed/:
    apple_mesh.json     surface + band geometry (decimated)
    poisson_frames.json 4 stages of the Poisson pipeline
    heat_frames.json    sub-sampled Heat trajectory on the surface
    meta.json           bbox, default camera, captions

Run once:
    conda activate pde
    python export_for_web.py
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import torch
from scipy.spatial import KDTree
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from utils import (
    define_band_points,
    Laplacian_matrix,
    retrieve_neural_weights,
    neural_extension,
    precompute_rbf_data,
    interpolate_from_precomputed,
)
from model import SurfNO_weights_only

# ---------------------------------------------------------------------------
DEVICE = "cpu"
USE_CACHE = True
SURFACE_DECIMATE = 3   # keep every N-th surface point in the JSON
BAND_DECIMATE = 4      # keep every N-th band point in the JSON
HEAT_FRAMES = 41       # number of frames in the heat animation (incl. t=0)

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "precomputed"
OUT.mkdir(exist_ok=True)
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)


def fmt(arr, decimals=4):
    """Round + tolist for compact JSON."""
    return np.round(np.asarray(arr), decimals).tolist()


# ---------------------------------------------------------------------------
# 1. surface, band, partition
# ---------------------------------------------------------------------------
print("[1] surface + band")
surface_feature_TP = np.load(ROOT / "data/Apple_surface_feature.npy")
surface_points  = surface_feature_TP[:, :3]
surface_normals = surface_feature_TP[:, 3:6]
Tree_surface_points = KDTree(surface_points)

delta_x, dist_to_surface = 0.05, 0.2
threshold = 0.8 * dist_to_surface

band_points, mask_threshold = define_band_points(
    delta_x, surface_points, Tree_surface_points,
    dist_to_surface, threshold)
Tree_band_points = KDTree(band_points)
print(f"    surface={surface_points.shape[0]}  band={band_points.shape[0]}")


# ---------------------------------------------------------------------------
# 2. neural weights (cached)
# ---------------------------------------------------------------------------
print("[2] neural weights")
cache_path = CACHE / "neural_weights.pt"
if USE_CACHE and cache_path.exists():
    blob = torch.load(cache_path, map_location=DEVICE, weights_only=False)
    neural_weights = [w.to(DEVICE) for w in blob["neural_weights"]]
    all_distances_to_central = blob["all_distances_to_central"]
    all_local_band_indexes   = blob["all_local_band_indexes"]
    print("    loaded from cache")
else:
    model = SurfNO_weights_only().to(DEVICE)
    model.load_state_dict(torch.load(
        ROOT / "data/SurfNO_pretrained_weights.pth",
        map_location=DEVICE, weights_only=True))
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        (neural_weights,
         all_distances_to_central,
         all_local_band_indexes) = retrieve_neural_weights(
            surface_points, band_points, 400, model,
            Tree_band_points, mask_threshold,
            surface_feature_TP=surface_feature_TP)
    print(f"    retrieve_neural_weights took {time.time()-t0:.1f} s")
    torch.save({
        "neural_weights":           [w.detach().cpu() for w in neural_weights],
        "all_distances_to_central": all_distances_to_central,
        "all_local_band_indexes":   all_local_band_indexes,
    }, cache_path)


# ---------------------------------------------------------------------------
# 3. assemble Laplacian + RBF interpolant
# ---------------------------------------------------------------------------
print("[3] Laplacian + RBF cache")
Lap, denom = Laplacian_matrix(band_points, Tree_band_points, delta_x)
neighbors_indices, factors, phi_vecs = precompute_rbf_data(
    band_points, Tree_band_points, surface_points, k=8, epsilon=1.0)


# ---------------------------------------------------------------------------
# 4. Poisson pipeline — record 4 stages
# ---------------------------------------------------------------------------
print("[4] Poisson")
omega = 10.0
f_raw = (np.sin(omega*band_points[:,0])
         * np.sin(omega*band_points[:,1])
         * np.sin(omega*band_points[:,2]))
f_raw -= f_raw.mean()

f_smooth = neural_extension(neural_weights, f_raw,
                      all_local_band_indexes,
                      all_distances_to_central)[0]

rhs = f_smooth * denom
u_band = spsolve(Lap, rhs); u_band -= u_band.mean()

u_surface = interpolate_from_precomputed(
    u_band, neighbors_indices, factors, phi_vecs, clipping=True)

poisson_stages = [
    {
        "name":   "1. raw RHS f on band",
        "target": "band",
        "values": f_raw,
        "caption": "f(x) = sin(ωx) sin(ωy) sin(ωz) sampled on every band point. "
                   "Note it is NOT constant in the normal direction.",
    },
    {
        "name":   "2. f after rot_update (neural smoothing)",
        "target": "band",
        "values": f_smooth,
        "caption": "rot_update applies the per-region neural attention and a "
                   "distance-weighted ensemble. The result is approximately "
                   "constant along surface normals — i.e. living on S.",
    },
    {
        "name":   "3. solve  Lap · u = f · δx²  on band",
        "target": "band",
        "values": u_band,
        "caption": "One sparse linear solve. u_band is the Poisson solution on "
                   "the narrow band.",
    },
    {
        "name":   "4. RBF interpolation → surface",
        "target": "surface",
        "values": u_surface,
        "caption": "Project the band solution onto the actual surface points "
                   "with the cached local RBF interpolant.",
    },
]


# ---------------------------------------------------------------------------
# 5. Heat pipeline — sub-sample to HEAT_FRAMES frames
# ---------------------------------------------------------------------------
print("[5] Heat")
Lap_h = Lap / denom
dt    = 0.1 * delta_x**2
nsteps = 100

u0 = (np.sin(omega*band_points[:,0])
      * np.sin(omega*band_points[:,1])
      * np.sin(omega*band_points[:,2]))
u0 = neural_extension(neural_weights, u0,
                all_local_band_indexes,
                all_distances_to_central)[0]

U_over_t = np.zeros((nsteps + 1, u0.shape[0]))
U_over_t[0] = u0
u_band = u0.copy()
for t in tqdm(range(nsteps), desc="    heat"):
    u_band = u_band + dt * (Lap_h @ u_band)
    u_band = neural_extension(neural_weights, u_band,
                        all_local_band_indexes,
                        all_distances_to_central,
                        temperature=0.0423)[0]
    U_over_t[t + 1] = u_band

heat_surface = np.zeros((nsteps + 1, surface_points.shape[0]))
for t in tqdm(range(nsteps + 1), desc="    RBF"):
    heat_surface[t] = interpolate_from_precomputed(
        U_over_t[t], neighbors_indices, factors, phi_vecs, clipping=True)

heat_idx = np.linspace(0, nsteps, HEAT_FRAMES).astype(int)
heat_frames = heat_surface[heat_idx]


# ---------------------------------------------------------------------------
# 6. write JSON
# ---------------------------------------------------------------------------
print("[6] writing JSON")
S = slice(None, None, SURFACE_DECIMATE)
B = slice(None, None, BAND_DECIMATE)

mesh = {
    "surface_points":  fmt(surface_points[S], 4),
    "surface_normals": fmt(surface_normals[S], 4),
    "band_points":     fmt(band_points[B], 4),
}
(OUT / "apple_mesh.json").write_text(json.dumps(mesh))

def pack_stage(stage):
    if stage["target"] == "surface":
        vals = stage["values"][S]
    else:
        vals = stage["values"][B]
    vmax = float(np.max(np.abs(vals)))
    return {
        "name":    stage["name"],
        "target":  stage["target"],
        "caption": stage["caption"],
        "values":  fmt(vals, 5),
        "vmin":    -vmax, "vmax": vmax,
    }

(OUT / "poisson_frames.json").write_text(json.dumps(
    [pack_stage(s) for s in poisson_stages]))

heat_vmax = float(np.max(np.abs(heat_frames[:, S])))
(OUT / "heat_frames.json").write_text(json.dumps({
    "target":     "surface",
    "n_frames":   len(heat_frames),
    "vmin":       -heat_vmax,
    "vmax":        heat_vmax,
    "step_times": [int(i) for i in heat_idx],
    "values":      fmt(heat_frames[:, S], 4),
}))

bbox_min = surface_points.min(axis=0).tolist()
bbox_max = surface_points.max(axis=0).tolist()
center   = ((surface_points.min(0) + surface_points.max(0)) / 2).tolist()
radius   = float(np.linalg.norm(surface_points.max(0) - surface_points.min(0)) / 2)
(OUT / "meta.json").write_text(json.dumps({
    "bbox_min": fmt(bbox_min, 4),
    "bbox_max": fmt(bbox_max, 4),
    "center":   fmt(center, 4),
    "radius":   round(radius, 4),
    "delta_x":  delta_x,
    "omega":    omega,
    "n_steps":  nsteps,
    "captions": {
        "intro":   "Below: neural-CPM applied to the Apple. "
                   "Step through the Poisson pipeline or play the Heat animation.",
    },
}))

print("done. files in:", OUT)
for p in sorted(OUT.iterdir()):
    print(f"   {p.name}  {p.stat().st_size/1024:.1f} KB")
