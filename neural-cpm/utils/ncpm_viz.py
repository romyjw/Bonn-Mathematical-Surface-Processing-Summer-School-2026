"""ncpm_viz.py

Small matplotlib / Plotly helpers used by the teaching notebook.
Kept deliberately thin so students can read & modify them.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the 3D projection
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# §1 — 2-D cartoon of the closest-point method
# ---------------------------------------------------------------------------
def plot_cpm_cartoon(ax=None, n_curve_pts=200, n_normals=18, band_w=0.25, grid_h=0.07):
    """Schematic 2-D figure: closed curve, narrow band of grid cells, normals,
    and the 'closest-point' arrows from a few grid cells back to the curve.

    Purpose: gives the eye the same picture students see in the HTML demo,
    before jumping to the 3-D Apple example.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    # closed curve: r(theta) = 1 + 0.25 sin(3 theta)
    theta = np.linspace(0, 2 * np.pi, n_curve_pts, endpoint=False)
    r = 1.0 + 0.25 * np.sin(3 * theta) + 0.08 * np.cos(5 * theta)
    curve = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

    # outward normals via central difference
    dx = np.gradient(curve[:, 0])
    dy = np.gradient(curve[:, 1])
    tang = np.stack([dx, dy], axis=1)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12
    nrm = np.stack([tang[:, 1], -tang[:, 0]], axis=1)  # rotate -90 deg
    # ensure outward
    if np.mean(np.sum(nrm * curve, axis=1)) < 0:
        nrm = -nrm

    # grid cells in a band
    xs = np.arange(-1.7, 1.7 + grid_h, grid_h)
    ys = np.arange(-1.7, 1.7 + grid_h, grid_h)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1)
    # closest point distance via brute force
    d = np.min(np.linalg.norm(pts[:, None, :] - curve[None, :, :], axis=2), axis=1)
    band_mask = d < band_w

    ax.scatter(pts[band_mask, 0], pts[band_mask, 1], s=8, c="#7ec8e3",
               alpha=0.55, label="band cells")
    ax.plot(curve[:, 0], curve[:, 1], color="#222", lw=2, label="surface S")

    step = max(1, n_curve_pts // n_normals)
    ax.quiver(curve[::step, 0], curve[::step, 1], nrm[::step, 0], nrm[::step, 1],
              color="#444", scale=18, width=0.004, alpha=0.7,
              label="outward normals")

    # a few cp arrows
    sel = np.where(band_mask)[0]
    rng = np.random.default_rng(0)
    sel = rng.choice(sel, size=min(6, sel.size), replace=False)
    for k in sel:
        p = pts[k]
        nearest = curve[np.argmin(np.linalg.norm(curve - p, axis=1))]
        ax.annotate("", xy=nearest, xytext=p,
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2))

    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("CPM idea: extend u from S into the band along normals")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    return ax


# ---------------------------------------------------------------------------
# §2.0 — surface + normals
# ---------------------------------------------------------------------------
def plot_surface_with_normals(points, normals, subsample=40, title="Surface + normals"):
    """3-D scatter of the surface coloured by z, with a subset of normals as
    arrows. Returns the matplotlib figure."""
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2],
               cmap="viridis", s=2, alpha=0.5)
    idx = np.arange(0, points.shape[0], subsample)
    ax.quiver(points[idx, 0], points[idx, 1], points[idx, 2],
              normals[idx, 0], normals[idx, 1], normals[idx, 2],
              length=0.06, color="k", linewidth=0.5, alpha=0.6)
    ax.set_title(title)
    ax.set_box_aspect((1, 1, 1))
    return fig


# ---------------------------------------------------------------------------
# §2.1 — band overlay
# ---------------------------------------------------------------------------
def plot_band_overlay(surface_pts, band_pts, title="Surface + narrow band"):
    """Plotly 3-D: surface as small dark points, band as light blue cloud."""
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=surface_pts[:, 0], y=surface_pts[:, 1], z=surface_pts[:, 2],
        mode="markers", marker=dict(size=1.4, color="#222"),
        name="surface"))
    fig.add_trace(go.Scatter3d(
        x=band_pts[:, 0], y=band_pts[:, 1], z=band_pts[:, 2],
        mode="markers", marker=dict(size=2, color="#7ec8e3", opacity=0.35),
        name="band"))
    fig.update_layout(title=title, scene=dict(aspectmode="data"),
                      margin=dict(l=0, r=0, b=0, t=30))
    return fig


def plot_band_slice(band_pts, axis=2, level=None, tol=0.03, title="Band slice"):
    """Matplotlib 2-D slice through the band at `axis = level`. Helps see the
    thickness/shape of the band volume."""
    if level is None:
        level = float(np.mean(band_pts[:, axis]))
    sel = np.abs(band_pts[:, axis] - level) < tol
    other = [i for i in range(3) if i != axis]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(band_pts[sel, other[0]], band_pts[sel, other[1]],
               s=8, c="#7ec8e3")
    ax.set_aspect("equal")
    ax.set_title(f"{title}  (axis={axis}, level={level:.2f})")
    return fig

def plot_band_slice_colored(band_pts, values_before, values_after,
                            axis=2, level=None, tol=0.03,
                            normal_axis=None,
                            cmap="magma", suptitle="Band slice — normal variation"):
    """Side-by-side 2-D slice of the band coloured by a scalar field, to see
    how the value changes in the *normal* direction.

    The slice is taken perpendicular to `axis` (default z).  The horizontal
    axis of each panel is the tangential direction; the vertical axis is
    `normal_axis` (defaults to the remaining coordinate with the largest spread),
    so the gradient in the vertical direction shows how much u varies as you
    move away from the surface.
    """
    if level is None:
        level = float(np.median(band_pts[:, axis]))

    sel = np.abs(band_pts[:, axis] - level) < tol
    if sel.sum() == 0:
        raise ValueError(f"No band points within tol={tol} of level={level:.3f} on axis={axis}.")

    other = [i for i in range(3) if i != axis]

    # pick the axis with the larger spread as the "normal" direction
    if normal_axis is None:
        spans = [np.ptp(band_pts[sel, i]) for i in other]
        normal_axis = other[np.argmax(spans)]
    tang_axis = [i for i in other if i != normal_axis][0]

    tang_label   = ["x", "y", "z"][tang_axis]
    normal_label = ["x", "y", "z"][normal_axis]

    pts2d_tang   = band_pts[sel, tang_axis]
    pts2d_normal = band_pts[sel, normal_axis]
    v_before     = values_before[sel]
    v_after      = values_after[sel]

    # shared limits so both panels are directly comparable
    vmin = min(v_before.min(), v_after.min())
    vmax = max(v_before.max(), v_after.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained")
    for ax, vals, label in zip(axes, [v_before, v_after],
                                ["before neural_extension", "after neural_extension"]):
        sc = ax.scatter(pts2d_tang, pts2d_normal,
                        c=vals, cmap=cmap, vmin=vmin, vmax=vmax,
                        s=18, linewidths=0)
        ax.set_xlabel(f"{tang_label} ")
        ax.set_ylabel(f"{normal_label}")
        ax.set_title(label)
        ax.set_aspect("equal")
    fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02, label="u")
    fig.suptitle(f"{suptitle}  (slice at {['x','y','z'][axis]}={level:.2f})")
    # fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# §2.2 — highlight one local region of the space partition
# ---------------------------------------------------------------------------
def plot_local_region(band_pts, region_idx, central_pt=None, surface_pts=None,
                      title="One local region of the band"):
    """3-D scatter showing the full band in grey and one chosen local region
    highlighted in red, with the region's central surface point as a dot."""
    fig = plt.figure(figsize=(6, 5))
    # ax = fig.add_subplot(111, projection="3d")
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.scatter(band_pts[:, 0], band_pts[:, 1], band_pts[:, 2],
               s=1, c="#cccccc", alpha=0.25, label="band", depthshade=True, zorder=1)
    region = band_pts[region_idx]
    ax.scatter(region[:, 0], region[:, 1], region[:, 2],
               s=10, c="#c0392b", label=f"local region (n={len(region_idx)})", depthshade=True, zorder=2)
    if surface_pts is not None:
        ax.scatter(surface_pts[:, 0], surface_pts[:, 1], surface_pts[:, 2],
                   s=0.1, c="#8B8A8A", alpha=0.4, label="surface", depthshade=True, zorder=3)
    if central_pt is not None:
        ax.scatter(*central_pt, s=100, marker="*", c="#f1c40f",
                   edgecolors="k", label="central pt", depthshade=False, zorder=1000)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_box_aspect((1, 1, 1))
    return fig


# ---------------------------------------------------------------------------
# §2.3 — neural attention heatmap
# ---------------------------------------------------------------------------
def plot_attention(alpha, title="Neural attention matrix (one region)"):
    """imshow of the (N,N) attention matrix produced by SurfNO for one
    local region."""
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(alpha, cmap="magma", aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("input band index")
    ax.set_ylabel("output band index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


# ---------------------------------------------------------------------------
# Generic scalar-field on a point cloud
# ---------------------------------------------------------------------------
def plot_field_on_points(points, values, title="", ax=None, s=2,
                         cmap="magma", vmin=None, vmax=None):
    """3-D scatter coloured by a scalar field. Defaults to the data range
    (not symmetric), which suits the sequential `magma` colormap used by
    the rest of the demo."""
    if vmin is None: vmin = float(np.min(values))
    if vmax is None: vmax = float(np.max(values))
    if ax is None:
        fig = plt.figure(figsize=(5, 4.5))
        ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                    c=values, cmap=cmap, s=s, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    return ax, sc


def snapshot_grid(points, U_over_t, frame_indices, suptitle="Heat evolution"):
    """2 x 3 matplotlib panel of 6 snapshots of a time-dependent field on
    the band (or surface). Uses a *shared* colour scale across panels so
    students can read off the diffusion."""
    fig = plt.figure(figsize=(12, 7))
    vmin = float(np.min(U_over_t))
    vmax = float(np.max(U_over_t))
    for k, t in enumerate(frame_indices):
        ax = fig.add_subplot(2, 3, k + 1, projection="3d")
        plot_field_on_points(points, U_over_t[t], title=f"t = {t}", ax=ax,
                             vmin=vmin, vmax=vmax)
    fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sparsity pattern of the Laplacian
# ---------------------------------------------------------------------------
def plot_laplacian_spy(L, title="Sparsity of L (band Laplacian)"):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.spy(L, markersize=0.3)
    ax.set_title(title)
    return fig
