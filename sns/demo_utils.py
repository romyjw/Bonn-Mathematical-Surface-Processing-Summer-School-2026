"""
demo_utils.py — helpers for the SNS demo notebook.

Provides:
  - o3d_mesh / launch_viewer / arrow_mesh  (Open3D visualisation)
  - SCALINGS / scalar_to_rgb / make_colorbar  (colour mapping)

Imported by sns_demo.ipynb after neural_surfaces-main has been added to sys.path.
"""

import os
import sys
import subprocess
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# ── Colour-scale functions ─────────────────────────────────────────────────────
# Imported from the library; if not yet on sys.path, add it automatically.
try:
    from visuals.helpers.colourmappings import (
        mapping12, mapping13,
        linear, linear2, linear3, linear5, linear6, linear7, linear8, linear9,
        linear10, linear100, linear500, linear1000, linear10000,
        quadratic, positive_only_linear1,
        scaled_normals_cmap, scaled_normals_cmap2, logmap,
    )
except ImportError:
    _ns = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'neural_surfaces-main')
    if os.path.isdir(_ns) and _ns not in sys.path:
        sys.path.insert(0, _ns)
    from visuals.helpers.colourmappings import (
        mapping12, mapping13,
        linear, linear2, linear3, linear5, linear6, linear7, linear8, linear9,
        linear10, linear100, linear500, linear1000, linear10000,
        quadratic, positive_only_linear1,
        scaled_normals_cmap, scaled_normals_cmap2, logmap,
    )

# Maps string keys (used in the notebook config) to scale functions.
SCALINGS = {
    'mapping12': mapping12, 'mapping13': mapping13,
    'linear': linear, 'linear2': linear2, 'linear3': linear3,
    'linear5': linear5, 'linear6': linear6, 'linear7': linear7,
    'linear8': linear8, 'linear9': linear9,
    'linear10': linear10, 'linear100': linear100,
    'linear500': linear500, 'linear1000': linear1000, 'linear10000': linear10000,
    'quadratic': quadratic, 'positive_only_linear1': positive_only_linear1,
    'scaled_normals_cmap': scaled_normals_cmap, 'scaled_normals_cmap2': scaled_normals_cmap2,
    'logmap': logmap,
}

# Default scaling and colormap per visualisation mode.
MODE_DEFAULTS = {
    'mean_curvature':     ('linear9',   'seismic'),
    'gaussian_curvature': ('linear5',   'seismic'),
    'max_curvature':      ('mapping12', 'seismic'),
    'area_distortion':    ('logmap',    'hot'),
    'laplace_beltrami':   ('linear5',   'viridis'),
}


# ── Colour mapping ─────────────────────────────────────────────────────────────

def scalar_to_rgb(values, cmap, scaling):
    """Map an array of scalar values to RGB using a named scaling function."""
    if scaling == 'percentile':
        lo, hi = np.percentile(values, [2, 98])
        scaled = Normalize(vmin=lo, vmax=hi, clip=True)(values)
    else:
        scaled = np.clip(SCALINGS[scaling](values), 0.0, 1.0)
    return plt.get_cmap(cmap)(scaled)[:, :3].astype(np.float64)


def make_colorbar(fig, ax, scale_fn, cmap_name, label, n_ticks=9):
    """Add a colorbar to ax with tick labels recovered via scale_fn(invert=True)."""
    sm = cm.ScalarMappable(cmap=cmap_name)
    sm.set_clim(0, 1)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax)
    tick_pos = np.linspace(0, 1, n_ticks)
    cb.set_ticks(tick_pos)
    cb.set_ticklabels([f'{v:.3g}' for v in scale_fn(tick_pos, invert=True)])
    cb.set_label(label, fontsize=9)


# ── Open3D helpers ─────────────────────────────────────────────────────────────

def o3d_mesh(V, F, colours):
    """Build an Open3D TriangleMesh from vertices, faces, and per-vertex colours."""
    import open3d as o3d
    m = o3d.geometry.TriangleMesh()
    m.vertices      = o3d.utility.Vector3dVector(np.asarray(V, dtype=np.float64))
    m.triangles     = o3d.utility.Vector3iVector(np.asarray(F))
    m.vertex_colors = o3d.utility.Vector3dVector(np.asarray(colours, dtype=np.float64))
    m.compute_vertex_normals()
    return m


def launch_colorbar(entries, cmap, title=''):
    """Render a colorbar to a temp PNG and open it in the system image viewer.

    entries: list of (label, scaling_key) pairs, e.g. [('mean curvature', 'linear9')]
    cmap:    colormap name string
    title:   figure suptitle
    """
    import json as _json

    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _ns_root  = os.path.join(_this_dir, 'neural_surfaces-main')

    script = f"""
import sys
sys.path.insert(0, {repr(_ns_root)})
sys.path.insert(0, {repr(_this_dir)})
import json, os, subprocess, tempfile
import matplotlib
matplotlib.use('Agg')          # non-interactive renderer; fine since we save to file
from demo_utils import SCALINGS, make_colorbar
import matplotlib.pyplot as plt

entries = json.loads({repr(_json.dumps(entries))})
n = len(entries)
fig, axes = plt.subplots(1, n, figsize=(max(1.5 * n, 2), 4))
if n == 1:
    axes = [axes]
for (label, scaling_key), ax in zip(entries, axes):
    make_colorbar(fig, ax, SCALINGS[scaling_key], {repr(cmap)}, label)
fig.suptitle({repr(title)}, fontsize=10, y=1.02)
plt.tight_layout()

tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
tmp.close()
fig.savefig(tmp.name, bbox_inches='tight', dpi=150)
import platform
_sys = platform.system()
if _sys == 'Darwin':
    subprocess.Popen(['open', tmp.name])
elif _sys == 'Windows':
    os.startfile(tmp.name)
else:
    subprocess.Popen(['xdg-open', tmp.name])
"""
    subprocess.Popen([sys.executable, '-c', script])


def launch_viewer(*meshes, window_name='SNS viewer', bg_color=(0.85, 0.85, 0.85)):
    """Open an interactive Open3D window in a subprocess (non-blocking)."""
    import open3d as o3d
    tmps = []
    for m in meshes:
        f = tempfile.NamedTemporaryFile(suffix='.ply', delete=False)
        f.close()
        o3d.io.write_triangle_mesh(f.name, m)
        tmps.append(f.name)

    load_lines   = '\n'.join(
        f'g{i} = o3d.io.read_triangle_mesh({repr(t)})' for i, t in enumerate(tmps))
    geom_list    = ', '.join(f'g{i}' for i in range(len(tmps)))
    unlink_lines = '\n'.join(f'os.unlink({repr(t)})' for t in tmps)

    script = f"""
import open3d as o3d, os, numpy as np
{load_lines}
vis = o3d.visualization.Visualizer()
vis.create_window(window_name={repr(window_name)}, width=1024, height=768)
for g in [{geom_list}]:
    vis.add_geometry(g)
vis.get_render_option().background_color = np.array({list(bg_color)})
while vis.poll_events():
    vis.update_renderer()
vis.destroy_window()
{unlink_lines}
"""
    subprocess.Popen([sys.executable, '-c', script])


def arrow_mesh(pts, dir_main, dir_perp, normals, al, offset, ratio, overlap=True):
    """
    Build arrowhead geometry for a crossfield visualisation.

    Returns (V, F) as numpy arrays suitable for o3d_mesh().
    Each point gets a small arrowhead in dir_main, with dir_perp setting the
    arrow width. overlap=True shifts the arrow so its tip sits at the base point.
    """
    V1    = pts + offset * normals
    N     = len(V1)
    shift = al * dir_main if overlap else np.zeros_like(dir_main)

    V = np.empty((N * 4, 3))
    V[0::4] = V1 - shift
    V[1::4] = V1 + al * dir_main + al * dir_perp - shift
    V[2::4] = V1 + al * dir_main - al * dir_perp - shift
    V[3::4] = V1 + ratio * al * dir_main          - shift

    idx = np.arange(N)
    F = np.empty((N * 2, 3), dtype=np.int32)
    F[0::2] = np.stack([4*idx+1, 4*idx,   4*idx+2], axis=1)
    F[1::2] = np.stack([4*idx+1, 4*idx+2, 4*idx+3], axis=1)

    return V.astype(np.float64), F
