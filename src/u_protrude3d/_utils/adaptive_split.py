"""Two-stage adaptive protrusion instance segmentation.

A simple, robust alternative to the Morse persistence splitter for real (noisy)
meshes, where height-persistence tends to *under*-segment.  The pipeline is:

1. **Height threshold** (global or adaptive Otsu) -> binary protrusive surface.
2. **Connected components** of that binary -> coarse protrusion patches.
3. **Per patch**, an adaptive binary threshold of a *joint* curvature + ridgeness
   signal (a 2-channel vector) -> instance cores (seeds).  Protrusion tips /
   crests are curvature/ridge maxima; necks are low -> they become boundaries.
4. **Grow + relabel** cores over their patch, enforcing every output label is a
   single connected component.

Reuses helpers from :mod:`morse_watershed` (smoothing, connectivity, region grow,
adaptive-Otsu baseline, region typing), so this module stays small.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import igl

from .morse_watershed import (
    _smooth_scalar_graph,
    adaptive_otsu_threshold,
    otsu_level_threshold,
    connected_component_relabel,
    grow_labels_over_domain,
    classify_region,
)


@dataclass
class AdaptiveSplitResult:
    """Return value of :func:`segment_protrusions_adaptive`."""
    labels: np.ndarray                 # (N,) 0 = background, 1..K instances (each 1 CC)
    n_patches: int                     # coarse patches from the height threshold
    background_level: object           # scalar or (N,) array actually used
    region_types: dict = field(default_factory=dict)


def _zscore(x):
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def _merge_weak_necks(labels, height, mean_curvature, adjacency, base_level,
                      prominence_frac=0.3, curv_neck_thresh=-0.02):
    """Merge adjacent instances separated by a *weak* neck (the split decision).

    The over-segmentation from always cutting each component is undone here: two
    adjacent instances are kept apart only if the neck between them is a genuine
    boundary, judged by two complementary signals along their shared border --

    * **prominence** (height): how far both summits rise above the crossing pass,
      normalised by protrusion height above ``base_level``.  Deep valley = real.
    * **concavity** (curvature): the most concave point on the shared border.
      A concave crease = real neck.

    Kept separate if the neck is strong on *either* signal; merged otherwise.
    One union-find pass over the region-adjacency graph.
    """
    from collections import defaultdict

    labels = np.asarray(labels)
    labs = np.setdiff1d(np.unique(labels), 0)
    if len(labs) < 2:
        return labels
    summit = {int(l): float(height[labels == l].max()) for l in labs}

    neck_h = defaultdict(lambda: -np.inf)   # (a,b) -> pass height (max border height)
    neck_c = defaultdict(lambda: np.inf)    # (a,b) -> most concave border curvature
    for v in range(len(labels)):
        lv = labels[v]
        if lv == 0:
            continue
        for w in adjacency[v]:
            lw = labels[w]
            if lw != 0 and lw != lv:
                key = (min(int(lv), int(lw)), max(int(lv), int(lw)))
                cross = min(height[v], height[w])
                if cross > neck_h[key]:
                    neck_h[key] = cross
                cc = min(mean_curvature[v], mean_curvature[w])
                if cc < neck_c[key]:
                    neck_c[key] = cc

    parent = {int(l): int(l) for l in labs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), neck in neck_h.items():
        lo = min(summit[a], summit[b])
        prominence = lo - neck
        denom = lo - base_level + 1e-9
        rel_prom = prominence / denom if denom > 0 else 0.0
        concave = neck_c[(a, b)] < curv_neck_thresh
        strong = (rel_prom >= prominence_frac) or concave
        if not strong:
            parent[find(a)] = find(b)

    out = np.zeros_like(labels)
    for l in labs:
        out[labels == l] = find(int(l))
    return out


def _resolve_height_threshold(height, faces, background, smooth_iters, alpha,
                              otsu_classes=3, otsu_level=-1):
    if isinstance(background, str):
        if background == 'otsu':
            return otsu_level_threshold(height, otsu_classes, otsu_level)
        if background == 'adaptive_otsu':
            return adaptive_otsu_threshold(height, faces, smooth_iters, alpha,
                                           otsu_classes=otsu_classes, otsu_level=otsu_level)
        if background == 'mean':
            return float(np.mean(height))
        raise ValueError(f"height_background: unknown option {background!r}")
    return background


def segment_protrusions_adaptive(
    mesh,
    height,
    mean_curvature,
    ridgeness,
    height_background='adaptive_otsu',
    adaptive_smooth_iters: int = 3000,
    adaptive_alpha: float = 0.5,
    otsu_classes: int = 3,
    otsu_level: int = -1,
    joint_method: str = 'kmeans',
    joint_smooth_iters: int = 100,
    joint_alpha: float = 0.5,
    n_clusters: int = 2,
    channel_combine: str = 'max',
    min_patch_size: int = 20,
    min_region_size: int = 20,
    merge_weak_necks: bool = True,
    neck_prominence_frac: float = 0.3,
    neck_curv_thresh: float = -0.02,
    grow: bool = True,
    classify: bool = True,
) -> AdaptiveSplitResult:
    """Segment protrusion instances by height thresholding + per-patch joint split.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    height, mean_curvature, ridgeness : (N,) float arrays
        Per-vertex protrusion height, mean-curvature composite, and ridgeness.
    height_background : {'adaptive_otsu', 'otsu', 'mean'}, float, or (N,) array
        Basal cut for step 1 (protrusive surface).  ``'adaptive_otsu'`` is a
        per-vertex cut that tracks an uneven cell body.
    adaptive_smooth_iters, adaptive_alpha : diffusion scale for the adaptive cut.
    joint_method : {'kmeans', 'otsu', 'mean'}
        How the joint curvature+ridge signal is split within each patch.
        ``'kmeans'`` (default) clusters the 2-channel adaptive-contrast vector
        ``(curvature, ridgeness)`` into ``n_clusters`` and takes the highest cluster
        as instance cores.  ``'otsu'`` / ``'mean'`` threshold a scalar fusion of the
        two channels (see ``channel_combine``) instead.
    joint_smooth_iters, joint_alpha : float
        Scale of the local baseline subtracted from each channel (the "adaptive"
        part) -- larger = coarser baseline.
    n_clusters : int -- number of k-means clusters for ``joint_method='kmeans'``.
    channel_combine : {'max', 'l2'}
        Scalar fusion of the two channels for the ``'otsu'`` / ``'mean'`` methods.
    min_patch_size : int -- drop protrusive patches smaller than this.
    min_region_size : int -- drop final instances smaller than this.
    merge_weak_necks : bool -- THE split decision.  After the per-patch cut, merge
        adjacent instances whose separating neck is weak, so a single bumpy
        protrusion is not split while genuinely distinct protrusions are kept apart.
    neck_prominence_frac : float -- keep separate if both summits rise at least this
        fraction (of their height above base) above the crossing pass.
    neck_curv_thresh : float -- keep separate if the shared border is more concave
        than this (a genuine concave neck).
    grow : bool -- grow cores downhill to fill each patch (full coverage).
    classify : bool -- populate ``region_types`` (bleb / ridge / filopodium).

    Returns
    -------
    AdaptiveSplitResult
    """
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    height = np.asarray(height, dtype=float)
    mc = np.asarray(mean_curvature, dtype=float)
    ridge = np.asarray(ridgeness, dtype=float)
    adjacency = igl.adjacency_list(F)

    # ---- 1. protrusive surface via height threshold ----
    Th = _resolve_height_threshold(height, F, height_background,
                                   adaptive_smooth_iters, adaptive_alpha,
                                   otsu_classes=otsu_classes, otsu_level=otsu_level)
    protrusive = height >= Th

    # ---- 2. connected components -> patches ----
    patch_labels = connected_component_relabel((protrusive * 1).astype(int), adjacency)
    for p in np.setdiff1d(np.unique(patch_labels), 0):
        if int(np.sum(patch_labels == p)) < min_patch_size:
            patch_labels[patch_labels == p] = 0
    protrusive = patch_labels > 0

    # ---- 3. joint curvature+ridgeness split within each patch ----
    # per-channel adaptive contrast (signal minus a smoothed local baseline)
    cc = mc - _smooth_scalar_graph(mc, F, n_iters=joint_smooth_iters, alpha=joint_alpha)
    cr = ridge - _smooth_scalar_graph(ridge, F, n_iters=joint_smooth_iters, alpha=joint_alpha)
    zc, zr = _zscore(cc), _zscore(cr)

    core_mask = np.zeros(len(V), dtype=bool)

    if joint_method == 'kmeans':
        # cluster the 2-channel (curvature, ridgeness) contrast vector; the cluster
        # with the highest combined centroid is the protrusion cores (tips / crests)
        from sklearn.cluster import MiniBatchKMeans
        idx = np.where(protrusive)[0]
        feats = np.column_stack([zc[idx], zr[idx]])
        k = max(2, int(n_clusters))
        km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3).fit(feats)
        core_cluster = int(np.argmax(km.cluster_centers_.sum(axis=1)))
        core_mask[idx[km.labels_ == core_cluster]] = True
    else:
        import skimage.filters as skf
        score = (np.sqrt(np.clip(zc, 0, None) ** 2 + np.clip(zr, 0, None) ** 2)
                 if channel_combine == 'l2' else np.maximum(zc, zr))
        for p in np.setdiff1d(np.unique(patch_labels), 0):
            vp = np.where(patch_labels == p)[0]
            c = score[vp]
            if joint_method == 'mean' or np.unique(c).size < 3:
                thr = float(np.mean(c))
            else:
                try:
                    thr = float(skf.threshold_otsu(c))
                except Exception:
                    thr = float(np.mean(c))
            core_mask[vp] = c >= thr
    core_mask &= protrusive

    core_labels = connected_component_relabel((core_mask * 1).astype(int), adjacency)

    # fallback: a patch with no detected core becomes one instance seeded at its summit
    for p in np.setdiff1d(np.unique(patch_labels), 0):
        vp = np.where(patch_labels == p)[0]
        if core_labels[vp].max() == 0:
            core_labels[vp[np.argmax(height[vp])]] = core_labels.max() + 1

    # ---- 4. grow cores over patches, enforce single connected component ----
    labels = grow_labels_over_domain(core_labels, height, adjacency, protrusive) if grow else core_labels
    labels = connected_component_relabel(labels, adjacency)

    # ---- 5. split decision: merge instances across weak necks ----
    if merge_weak_necks:
        base = float(np.mean(Th)) if np.ndim(Th) else float(Th)
        labels = _merge_weak_necks(labels, height, mc, adjacency, base,
                                   prominence_frac=neck_prominence_frac,
                                   curv_neck_thresh=neck_curv_thresh)
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

    return AdaptiveSplitResult(
        labels=labels,
        n_patches=int(len(np.setdiff1d(np.unique(patch_labels), 0))),
        background_level=Th,
        region_types=region_types,
    )
