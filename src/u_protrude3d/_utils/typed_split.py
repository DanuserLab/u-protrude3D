"""Typed per-component protrusion splitting (simplified multi-bleb / ridge).

Biologically-motivated split decision made *within* each connected protrusion
component (patch), instead of an abstract merge/persistence rule:

1. **Classify the patch** by its curvature/ridge content:
   * high curvature  -> bleb (low ridge) or filopodium (also high ridge)
   * low curvature, high ridge -> lamellipodium (sheet-like)
2. **Curvature regime (bleb / filopodium):** re-threshold *curvature* to propose
   tip cores.  Accept the split only if >= 2 cores are **globular / tip-like**
   (round in PCA, i.e. low aspect ratio); otherwise keep the patch as one
   instance.
3. **Ridge regime (lamellipodium):** re-threshold *ridgeness*.  If it yields a
   single component, keep the patch whole; otherwise accept the (line-like)
   pieces.
4. Grow accepted cores over the patch and relabel so each instance is one
   connected component.

Reuses helpers from :mod:`morse_watershed` and :mod:`adaptive_split`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import igl

from .morse_watershed import (
    _smooth_scalar_graph,
    connected_component_relabel,
    grow_labels_over_domain,
    classify_region,
)
from .adaptive_split import _resolve_height_threshold, _zscore


@dataclass
class TypedSplitResult:
    """Return value of :func:`segment_protrusions_typed`."""
    labels: np.ndarray
    n_patches: int
    patch_types: dict = field(default_factory=dict)      # patch id -> 'curv' | 'ridge'
    background_level: object = None
    region_types: dict = field(default_factory=dict)     # label -> bleb/ridge/filopodium


def _aspect(pts):
    """PCA elongation of a point set (~1 globular, >2 elongated)."""
    if len(pts) < 3:
        return 1.0
    c = pts - pts.mean(axis=0)
    sv = np.linalg.svd(c, compute_uv=False)
    if len(sv) < 2 or sv[1] <= 1e-12:
        return 1.0
    return float(sv[0] / sv[1])


def _patch_threshold(values, method):
    if method == 'mean' or np.unique(values).size < 3:
        return float(np.mean(values))
    import skimage.filters as skf
    try:
        return float(skf.threshold_otsu(values))
    except Exception:
        return float(np.mean(values))


def segment_protrusions_typed(
    mesh,
    height,
    mean_curvature,
    ridgeness,
    height_background='otsu',
    adaptive_smooth_iters: int = 3000,
    adaptive_alpha: float = 0.5,
    otsu_classes: int = 3,
    otsu_level: int = -1,
    split_method: str = 'otsu',
    split_smooth_iters: int = 100,
    split_alpha: float = 0.5,
    curv_regime_z: float = 0.0,
    globular_max_aspect: float = 2.0,
    min_core_size: int = 10,
    min_patch_size: int = 20,
    min_region_size: int = 20,
    grow: bool = True,
    classify: bool = True,
) -> TypedSplitResult:
    """Segment protrusion instances by typed per-component re-thresholding.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    height, mean_curvature, ridgeness : (N,) float arrays.
    height_background : basal cut for the protrusive surface ('otsu' /
        'adaptive_otsu' / 'mean' / float / (N,) array).
    split_method : {'otsu', 'mean'} -- re-threshold rule inside each patch.
    split_smooth_iters, split_alpha : local-baseline scale for the adaptive
        contrast of the re-thresholded channel.
    curv_regime_z : float -- a patch is treated as bleb/filopodium (curvature
        regime) when its mean z-scored curvature >= this, else lamellipodium.
    globular_max_aspect : float -- a curvature core is accepted as a bleb tip if
        its PCA aspect ratio is <= this (round / tip-like).
    min_core_size, min_patch_size, min_region_size : size filters (vertices).
    grow : bool -- grow accepted cores to fill each patch (full coverage).
    classify : bool -- populate ``region_types``.

    Returns
    -------
    TypedSplitResult
    """
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    height = np.asarray(height, dtype=float)
    mc = np.asarray(mean_curvature, dtype=float)
    ridge = np.asarray(ridgeness, dtype=float)
    adjacency = igl.adjacency_list(F)

    # ---- 1. protrusive surface + patches ----
    Th = _resolve_height_threshold(height, F, height_background,
                                   adaptive_smooth_iters, adaptive_alpha,
                                   otsu_classes=otsu_classes, otsu_level=otsu_level)
    protrusive = height >= Th
    patch_labels = connected_component_relabel((protrusive * 1).astype(int), adjacency)
    for p in np.setdiff1d(np.unique(patch_labels), 0):
        if int(np.sum(patch_labels == p)) < min_patch_size:
            patch_labels[patch_labels == p] = 0
    protrusive = patch_labels > 0

    # ---- 2. per-channel adaptive contrast + patch type ----
    zc = _zscore(mc)
    con_curv = mc - _smooth_scalar_graph(mc, F, n_iters=split_smooth_iters, alpha=split_alpha)
    con_ridge = ridge - _smooth_scalar_graph(ridge, F, n_iters=split_smooth_iters, alpha=split_alpha)

    patch_ids = np.setdiff1d(np.unique(patch_labels), 0)
    branch = {}                       # patch -> 'curv' | 'ridge'
    signal = np.zeros(len(V))
    thr_vertex = np.zeros(len(V))
    for p in patch_ids:
        vp = np.where(patch_labels == p)[0]
        is_curv = float(np.mean(zc[vp])) >= curv_regime_z
        branch[int(p)] = 'curv' if is_curv else 'ridge'
        chan = con_curv if is_curv else con_ridge
        signal[vp] = chan[vp]
        thr_vertex[vp] = _patch_threshold(chan[vp], split_method)

    # ---- 3. cores = re-thresholded high regions ----
    core_mask = (signal >= thr_vertex) & protrusive
    core_labels = connected_component_relabel((core_mask * 1).astype(int), adjacency)
    core_ids = np.setdiff1d(np.unique(core_labels), 0)

    core_verts = {int(c): np.where(core_labels == c)[0] for c in core_ids}
    core_verts = {c: v for c, v in core_verts.items() if len(v) >= min_core_size}
    core_patch = {c: int(patch_labels[v[0]]) for c, v in core_verts.items()}
    core_aspect = {c: _aspect(V[v]) for c, v in core_verts.items()}

    # ---- 4. per-patch accept / reject -> seeds ----
    seed_labels = np.zeros(len(V), dtype=np.int64)
    nxt = 0
    for p in patch_ids:
        p = int(p)
        vp = np.where(patch_labels == p)[0]
        cores_p = [c for c in core_verts if core_patch[c] == p]

        if branch[p] == 'curv':
            globular = [c for c in cores_p if core_aspect[c] <= globular_max_aspect]
            accept = len(globular) >= 2
            keep = globular
        else:  # ridge / lamellipodium
            accept = len(cores_p) > 1
            keep = cores_p

        if accept:
            for c in keep:
                nxt += 1
                seed_labels[core_verts[c]] = nxt
        else:
            nxt += 1
            seed_labels[vp[np.argmax(height[vp])]] = nxt   # single instance for the patch

    # ---- 5. grow, enforce single component, clean up ----
    labels = grow_labels_over_domain(seed_labels, height, adjacency, protrusive) if grow else seed_labels
    labels = connected_component_relabel(labels, adjacency)
    for lab in np.setdiff1d(np.unique(labels), 0):
        if int(np.sum(labels == lab)) < min_region_size:
            labels[labels == lab] = 0

    uniq = np.setdiff1d(np.unique(labels), 0)
    remap = {o: i + 1 for i, o in enumerate(uniq)}
    out = np.zeros_like(labels)
    for o, n in remap.items():
        out[labels == o] = n
    labels = out

    region_types = {}
    if classify:
        for lab in np.setdiff1d(np.unique(labels), 0):
            region_types[int(lab)] = classify_region(V, ridge, labels == lab, height=height)

    return TypedSplitResult(
        labels=labels,
        n_patches=len(patch_ids),
        patch_types={p: branch[p] for p in branch},
        background_level=Th,
        region_types=region_types,
    )
