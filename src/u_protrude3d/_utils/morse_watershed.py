"""Morse / persistence-based instance splitting of surface protrusions.

This is an *alternative* to the binary-threshold + connected-components path used
in :mod:`segment.py`.  Instead of first classifying a patch's type (bleb / ridge /
filopodium) and then cutting it with a single thresholded signal, this module
treats the protrusion height field ``h`` as a Morse function on the mesh and
partitions the surface into the basins of attraction of its maxima, then collapses
insignificant basins up the merge tree.

Design principles
-----------------
1. **Type is never a gate.**  The same operation splits fused blebs and refuses to
   shatter a single ridge; motif type is only assigned *after* the regions exist
   (see :func:`classify_region`).
2. **Full coverage.**  Every protrusive vertex belongs to exactly one instance.
   Persistence simplification *merges* a killed maximum's basin into the surviving
   parent it meets at its death saddle -- it never discards the vertices.  This is
   the key difference from naive "flood only from the surviving seeds", which would
   leave the killed peaks and their upper slopes as holes.
3. **Each field has one job**

   * ``height``          -> *which* / *how many* instances, and the merge order.
   * ``mean_curvature``  -> optional boundary nudging (geodesic backend only).
   * ``ridgeness``       -> post-hoc region typing only.
4. **One data-driven knob.**  The persistence threshold ``tau`` separates real
   protrusions from smoothing / sampling noise, chosen automatically from the
   largest gap in the persistence diagram (anchored at 0) unless overridden.

The only external dependency is :mod:`igl` (vertex adjacency); everything else is
pure NumPy, so the routine is trivially testable in isolation.  The geodesic
backend additionally uses :mod:`potpourri3d` (imported lazily).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import igl


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MorseSplitResult:
    """Return value of :func:`split_protrusions_morse`."""
    labels: np.ndarray                 # (N,) int, 0 = background, 1..K = instances
    seed_vertices: np.ndarray          # (K,) summit vertex of each surviving instance
    persistence_threshold: float       # tau actually used
    persistences: np.ndarray           # (n_maxima,) persistence of every raw maximum
    region_types: dict = field(default_factory=dict)   # label -> 'bleb'|'ridge'|'filopodium'


# ---------------------------------------------------------------------------
# Pass 1 -- full watershed + merge tree (persistence of every maximum)
# ---------------------------------------------------------------------------

@dataclass
class WatershedTree:
    """Full Morse watershed of a height field plus its persistence merge tree."""
    basin_of: np.ndarray       # (N,) int -- initial basin id of every vertex (full coverage)
    summits: np.ndarray        # (B,) int -- summit vertex of each basin
    persistence: np.ndarray    # (B,) float -- persistence of each basin (inf for global max)
    merge_parent: np.ndarray   # (B,) int -- basin this basin merges INTO at death (-1 for global)
    death_saddle: np.ndarray   # (B,) int -- vertex of the saddle where the basin died (-1 for global)


def compute_persistence_watershed(height, adjacency) -> WatershedTree:
    """Full watershed of *height* and the persistence merge tree, in one sweep.

    Simulated-immersion in descending height.  Each local maximum opens a basin;
    every other vertex joins the basin of its tallest already-flooded neighbour, so
    **all** vertices are labelled (full coverage).  When two basins first meet at a
    saddle, the lower-summit basin is recorded as merging into the taller one, with
    persistence ``summit_height - saddle_height``.  The global maximum never merges
    (persistence ``+inf``, parent ``-1``).

    Parameters
    ----------
    height : (N,) float array
    adjacency : list of int arrays -- vertex one-ring neighbours (igl.adjacency_list)

    Returns
    -------
    WatershedTree
    """
    height = np.asarray(height, dtype=float)
    n = len(height)
    order = np.argsort(height)[::-1]           # high -> low
    visited = np.zeros(n, dtype=bool)

    par = np.full(n, -1, dtype=np.int64)       # union-find parent (par[root] == root)
    peak = np.zeros(n)                          # summit height at each root

    def find(x):
        root = x
        while par[root] != root:
            root = par[root]
        while par[x] != root:
            par[x], x = root, par[x]
        return root

    basin_of = np.full(n, -1, dtype=np.int64)
    summits = []                                # basin id -> summit vertex
    summit_bid = {}                             # summit vertex -> basin id
    root_summit = {}                            # current dsu root vertex -> summit vertex
    persistence = {}                            # summit vertex -> persistence
    merge_into = {}                             # summit vertex -> parent summit vertex (or -1)
    death = {}                                  # summit vertex -> death saddle vertex (or -1)

    for v in order:
        roots = {find(u) for u in adjacency[v] if visited[u]}

        if not roots:
            # brand-new local maximum opens a basin
            par[v] = v
            peak[v] = height[v]
            bid = len(summits)
            summits.append(v)
            summit_bid[v] = bid
            root_summit[v] = v
            persistence[v] = np.inf
            merge_into[v] = -1
            death[v] = -1
            basin_of[v] = bid
        else:
            roots = list(roots)
            tallest = max(roots, key=lambda r: peak[r])
            win_summit = root_summit[tallest]
            basin_of[v] = summit_bid[win_summit]

            # attach v to the tallest basin
            par[v] = tallest

            # every other touching basin dies here -> merges into the tallest
            for r in roots:
                if r == tallest:
                    continue
                s = root_summit[r]
                persistence[s] = peak[r] - height[v]
                merge_into[s] = win_summit
                death[s] = v
                par[r] = tallest                # union r into tallest
            # tallest remains the root; its summit mapping is unchanged

        visited[v] = True

    summits = np.array(summits, dtype=np.int64)
    persistence_arr = np.array([persistence[s] for s in summits], dtype=float)
    merge_parent = np.array(
        [summit_bid[merge_into[s]] if merge_into[s] != -1 else -1 for s in summits],
        dtype=np.int64,
    )
    death_saddle = np.array([death[s] for s in summits], dtype=np.int64)
    return WatershedTree(basin_of, summits, persistence_arr, merge_parent, death_saddle)


def compute_height_persistence(height, adjacency):
    """Thin wrapper: return only ``(summit_vertices, persistences)``.

    Kept for callers / tests that just want the persistence diagram.
    """
    tree = compute_persistence_watershed(height, adjacency)
    return tree.summits, tree.persistence


def auto_persistence_threshold(persistences):
    """Pick tau at the largest gap in the sorted finite persistence spectrum.

    Real protrusions and noise maxima form two clouds separated by a gap in the
    persistence diagram; tau is placed at the midpoint of the widest gap.  A zero
    baseline is prepended so the noise floor is anchored at 0 -- otherwise, when
    every maximum is genuine (no noise cluster), the widest gap would fall
    *between* two real protrusions and wrongly drop one.
    """
    finite = np.sort(persistences[np.isfinite(persistences)])
    if len(finite) == 0:
        return 0.0
    aug = np.concatenate([[0.0], finite])
    gaps = np.diff(aug)
    k = int(np.argmax(gaps))
    return 0.5 * (aug[k] + aug[k + 1])


# ---------------------------------------------------------------------------
# Pass 2a -- collapse the merge tree to a persistence threshold (full coverage)
# ---------------------------------------------------------------------------

def merge_basins_to_threshold(tree: WatershedTree, tau, min_region_size=0,
                              mean_curvature=None, curv_neck_thresh=None):
    """Collapse basins up the merge tree until only surviving roots remain.

    A basin with persistence < tau is unioned into the parent it meets at its death
    saddle; the union is transitive, so a killed leaf ends up in whichever surviving
    basin ultimately absorbs its branch.  Because the starting watershed covered
    every vertex, the result also covers every vertex -- killed peaks are *merged*,
    never dropped.

    Curvature-aware split
    ---------------------
    When *mean_curvature* and *curv_neck_thresh* are given, a low-persistence basin
    is kept separate anyway if its death saddle is a concave crease
    (``mean_curvature[saddle] < curv_neck_thresh``).  Two protrusions joined by a
    shallow *height* neck but a genuine concave neck therefore stay distinct -- the
    height field alone cannot separate them, but the curvature at the neck can.

    Optionally, surviving basins smaller than *min_region_size* are then absorbed
    into their own death-saddle parent (removes speck instances without holes).

    Returns
    -------
    labels : (N,) int -- 1..K contiguous instance labels (full coverage).
    surviving_summits : (K,) int -- summit vertex of each final instance.
    """
    n_basins = len(tree.summits)
    rep = np.arange(n_basins)

    def find(b):
        while rep[b] != b:
            rep[b] = rep[rep[b]]
            b = rep[b]
        return b

    use_curv = mean_curvature is not None and curv_neck_thresh is not None
    if use_curv:
        mean_curvature = np.asarray(mean_curvature)

    # merge low-persistence basins into their parents, children first
    for b in np.argsort(tree.persistence):
        if tree.persistence[b] < tau and tree.merge_parent[b] != -1:
            if use_curv and tree.death_saddle[b] >= 0 \
                    and mean_curvature[tree.death_saddle[b]] < curv_neck_thresh:
                continue   # concave neck -> keep these protrusions separate
            rep[find(b)] = find(tree.merge_parent[b])

    # optional: dissolve undersized survivors up their own branch
    if min_region_size > 0:
        for _ in range(8):
            reps = np.array([find(b) for b in range(n_basins)])
            final = reps[tree.basin_of]
            sizes = np.bincount(final, minlength=n_basins)
            changed = False
            for b in range(n_basins):
                if find(b) == b and 0 < sizes[b] < min_region_size and tree.merge_parent[b] != -1:
                    rep[b] = find(tree.merge_parent[b])
                    changed = True
            if not changed:
                break

    reps = np.array([find(b) for b in range(n_basins)])
    final_basin = reps[tree.basin_of]

    surviving = np.setdiff1d(np.unique(reps), [])
    remap = {r: i + 1 for i, r in enumerate(surviving)}
    labels = np.zeros(len(tree.basin_of), dtype=np.int32)
    for r, lab in remap.items():
        labels[final_basin == r] = lab

    surviving_summits = np.array([tree.summits[r] for r in surviving], dtype=np.int64)
    return labels, surviving_summits


# ---------------------------------------------------------------------------
# Pass 2b -- geodesic (heat-method) surface-Voronoi assignment from seeds
# ---------------------------------------------------------------------------

def geodesic_marker_watershed(mesh, seed_vertices):
    """Assign every vertex to the geodesically nearest seed (surface Voronoi).

    Uses the heat method (potpourri3d ``MeshHeatMethodDistanceSolver``) to measure
    distance *along the surface*, so tightly folded patches never leak labels
    across a concavity the way a Euclidean nearest-neighbour transfer would.  The
    solver is factorised once and each seed is a cheap back-substitution.  Every
    vertex receives a label (1..K); there are no unreached vertices.
    """
    import potpourri3d as pp3d

    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int32)
    solver = pp3d.MeshHeatMethodDistanceSolver(V, F)

    dist_stack = np.empty((len(V), len(seed_vertices)), dtype=np.float64)
    for i, s in enumerate(seed_vertices):
        dist_stack[:, i] = solver.compute_distance(int(s))

    return (np.argmin(dist_stack, axis=1) + 1).astype(np.int32)


# ---------------------------------------------------------------------------
# Connectivity: guarantee each label is a single connected component
# ---------------------------------------------------------------------------

def _smooth_scalar_graph(values, faces, n_iters=50, alpha=0.5):
    """Laplacian smoothing of a vertex scalar over the mesh graph (self-contained).

    Row-normalised adjacency diffusion: ``v <- (1-a)v + a * mean_of_neighbours``.
    Used to build the locally-varying baseline for the adaptive basal cut.
    """
    import scipy.sparse as sp
    values = np.asarray(values, dtype=float)
    A = igl.adjacency_matrix(np.asarray(faces))
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    P = sp.diags(1.0 / deg) @ A
    v = values.copy()
    for _ in range(int(n_iters)):
        v = (1.0 - alpha) * v + alpha * (P @ v)
    return v


def otsu_level_threshold(values, classes=3, level=-1):
    """Multi-Otsu threshold: ``threshold_multiotsu(values, classes)[level]``.

    The original pipeline detects the protrusive surface with ``classes=3, level=-1``
    (the *upper* of the two 3-class thresholds).
    """
    import skimage.filters as _skf
    return float(_skf.threshold_multiotsu(np.asarray(values), classes)[level])


def adaptive_otsu_threshold(height, faces, smooth_iters=3000, alpha=0.5,
                            otsu_classes=3, otsu_level=-1):
    """Per-vertex basal cut: local baseline + the global Otsu protrusion margin.

    The margin ``otsu - mean(height)`` (how far a protrusion sits above the typical
    surface) is added to a locally-smoothed baseline of the height, so the cut tracks
    a sloped or uneven cell body instead of using one global level.  ``otsu_classes``
    / ``otsu_level`` select the multi-Otsu threshold (3-class upper by default, as in
    the original pipeline).

    ``smooth_iters`` is a diffusion *scale*: it must be large enough that the baseline
    smooths over whole protrusions (otherwise each protrusion lifts its own baseline
    and is erased).  It scales with mesh resolution -- increase it for finer meshes.
    On a uniform-bodied cell the adaptive cut reduces to the global Otsu level.
    """
    height = np.asarray(height, dtype=float)
    margin = otsu_level_threshold(height, otsu_classes, otsu_level) - float(np.mean(height))
    baseline = _smooth_scalar_graph(height, faces, n_iters=smooth_iters, alpha=alpha)
    return baseline + margin


def erode_curvature(mean_curvature, adjacency, n_rings):
    """Morphological min-erosion of a curvature field over the mesh graph.

    Replaces each vertex value with the minimum over its *n_rings* neighbourhood, so
    a saddle inherits the *most concave* value nearby.  Makes the neck test robust to
    the exact death-saddle vertex, which is a single noisy sample of the crease.
    """
    g = np.asarray(mean_curvature, dtype=float).copy()
    for _ in range(int(n_rings)):
        g2 = g.copy()
        for v in range(len(g)):
            nb = adjacency[v]
            if nb:
                m = g[nb].min()
                if m < g2[v]:
                    g2[v] = m
        g = g2
    return g


def separate_touching_labels(labels, adjacency, n_rings):
    """Carve a background gap where two different instances meet.

    Erodes only the *inter-instance* boundaries: a labelled vertex with a neighbour
    of a different non-zero label becomes background (0).  The outer perimeter of a
    protrusion (labelled next to background) is left intact, so protrusions keep
    their extent but no longer conjoin at the necks -- each ring opens the gap by
    one vertex on each side.
    """
    labels = np.asarray(labels).copy()
    for _ in range(int(n_rings)):
        strip = []
        for v in range(len(labels)):
            lv = labels[v]
            if lv == 0:
                continue
            for w in adjacency[v]:
                if labels[w] != 0 and labels[w] != lv:
                    strip.append(v)
                    break
        labels[strip] = 0
    return labels


def grow_labels_over_domain(labels, height, adjacency, domain_mask):
    """Flood seed labels downhill to fill *domain_mask*, keeping seed identity.

    Marker-controlled region growing: processing vertices in descending height, an
    unlabelled domain vertex inherits the label of its highest already-labelled
    neighbour.  Fronts from two seeds meet at the neck between them, so the neck
    re-fills but is split at its midline.  A cleanup pass mops up any local pits.

    Used to re-grow the ``fill_depth`` cap cores back over the whole protrusion
    domain: full coverage *and* the separation the cap cores carry.
    """
    labels = np.asarray(labels).copy()
    order = [int(v) for v in np.argsort(height)[::-1] if domain_mask[v]]

    for v in order:
        if labels[v] != 0:
            continue
        best, best_h = 0, -np.inf
        for w in adjacency[v]:
            if labels[w] != 0 and height[w] > best_h:
                best_h, best = height[w], labels[w]
        labels[v] = best

    changed = True
    while changed:
        changed = False
        for v in order:
            if labels[v] != 0:
                continue
            for w in adjacency[v]:
                if labels[w] != 0:
                    labels[v] = labels[w]
                    changed = True
                    break
    return labels


def connected_component_relabel(labels, adjacency):
    """Relabel so every output label is exactly one connected component.

    Background (0) stays 0.  Two disconnected pieces that happen to share an input
    label -- e.g. one full-coverage basin fragmented by a background mask -- are
    given distinct output labels.
    """
    labels = np.asarray(labels)
    n = len(labels)
    out = np.zeros(n, dtype=np.int32)
    visited = np.zeros(n, dtype=bool)
    cur = 0
    for s in range(n):
        if labels[s] == 0 or visited[s]:
            continue
        cur += 1
        lab = labels[s]
        stack = [s]
        visited[s] = True
        while stack:
            u = stack.pop()
            out[u] = cur
            for w in adjacency[u]:
                if not visited[w] and labels[w] == lab:
                    visited[w] = True
                    stack.append(w)
    return out


# ---------------------------------------------------------------------------
# Post-hoc region typing (descriptor, not a gate)
# ---------------------------------------------------------------------------

def classify_region(vertices, ridgeness, region_mask, height=None,
                    ridge_frac_thresh=0.5, aspect_thresh=3.0):
    """Label a finished region 'bleb' | 'ridge' | 'filopodium' from its shape.

    Elongation is measured on the region's *elevated core* (height-weighted PCA),
    so the flat skirt around a protrusion barely contributes.  ``ridge_frac`` (the
    fraction of the elevated core that is ridge-like) then distinguishes a thin
    ridged finger (filopodium) from a broader crest (ridge); isotropic cores are
    blebs.
    """
    pts = vertices[region_mask]
    if len(pts) < 4:
        return 'bleb'

    if height is not None:
        w = np.clip(np.asarray(height)[region_mask], 0.0, None)
        if w.sum() <= 0:
            w = np.ones(len(pts))
    else:
        w = np.ones(len(pts))

    mean = (pts * w[:, None]).sum(0) / w.sum()
    c = pts - mean
    cov = (c.T * w) @ c / w.sum()
    eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
    aspect = np.sqrt(max(eig[0], 0) / (max(eig[1], 1e-12)))

    if height is not None:
        hr = np.asarray(height)[region_mask]
        core = hr >= np.median(hr)
    else:
        core = np.ones(len(pts), dtype=bool)
    core_ridge = ridgeness[region_mask][core]
    ridge_frac = float(np.mean(core_ridge > np.nanmean(ridgeness) + 1e-12)) if core_ridge.size else 0.0

    if aspect >= aspect_thresh and ridge_frac >= ridge_frac_thresh:
        return 'filopodium'
    if aspect >= aspect_thresh:
        return 'ridge'
    return 'bleb'


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def split_protrusions_morse(
    mesh,
    height,
    mean_curvature=None,
    ridgeness=None,
    persistence_threshold: Optional[float] = None,
    min_region_size: int = 5,
    background_level=None,
    adaptive_smooth_iters: int = 3000,
    adaptive_alpha: float = 0.5,
    otsu_classes: int = 3,
    otsu_level: int = -1,
    fill_depth: Optional[float] = None,
    regrow: bool = False,
    curv_neck_thresh: Optional[float] = None,
    neck_curv_erode: int = 0,
    separate_rings: int = 0,
    enforce_connected: bool = True,
    flood_method: str = 'watershed',
    classify: bool = True,
) -> MorseSplitResult:
    """Instance-split a protrusion surface by persistence-simplified Morse watershed.

    The full watershed covers every vertex; basins whose persistence falls below
    ``tau`` are merged up the tree into their surviving parent, so within the
    protrusion domain no killed peak is ever left unlabelled.

    The protrusion domain itself is defined here (not by any external mask): every
    vertex with ``height < background_level`` becomes background (0).  This removes
    residual labels at the necks / basal collar, which naturally separate adjacent
    protrusions once the neck dips below the basal level.

    Parameters
    ----------
    mesh : trimesh.Trimesh -- the (sub)mesh to segment.
    height : (N,) float array -- protrusion height field (the Morse function).
    mean_curvature : (N,) float array or None -- reserved for boundary nudging.
    ridgeness : (N,) float array or None -- used only for post-hoc region typing.
    persistence_threshold : float or None -- tau.  ``None`` => auto (largest gap).
    min_region_size : int -- basins / components smaller than this are removed.
    background_level : float, (N,) array, {'otsu', 'mean', 'adaptive_otsu'}, or None --
        vertices below this height become background (0).  ``'otsu'`` picks one basal
        level per cell (two-class Otsu of the height); ``'adaptive_otsu'`` makes it a
        per-vertex, spatially-varying cut = locally-smoothed baseline + the global Otsu
        protrusion margin, so it tracks a sloped / uneven cell body; ``'mean'`` uses the
        mean height; a float or per-vertex array sets it directly; ``None`` keeps all.
    adaptive_smooth_iters : int -- graph-smoothing iterations for the adaptive baseline.
    adaptive_alpha : float -- smoothing step for the adaptive baseline (0..1).
    fill_depth : float or None -- relative flood ceiling: within each instance, drop
        vertices lower than ``summit_height - fill_depth`` to background.  This
        suppresses the fill so it does not spread across the deep neck bridging two
        summits; connectivity then splits those into separate instances.  A pure
        height mechanism -- no curvature needed.  ``None`` fills to ``background_level``.
    regrow : bool -- after a ``fill_depth`` split, re-grow the separated cap cores
        back down over the full protrusion domain (marker watershed), so the necks
        re-fill while keeping their split identity -- full coverage *and* separation.
    curv_neck_thresh : float or None -- if given (together with ``mean_curvature``),
        two basins are kept separate when the saddle between them is a concave crease
        (``mean_curvature[saddle] < curv_neck_thresh``), even if their height
        persistence is below ``tau``.  Separates protrusions fused by a shallow
        height neck but a real concave neck.  ``None`` disables curvature use.
    neck_curv_erode : int -- rings of min-erosion applied to ``mean_curvature`` before
        the neck test, so the saddle sees the most concave point nearby (robust to the
        exact saddle vertex).  ``0`` uses the raw saddle value.
    separate_rings : int -- carve a background gap of this many rings along the
        boundaries where two instances touch, so neighbouring protrusions do not
        conjoin at their necks.  Only inter-instance boundaries are eroded; the outer
        perimeter is preserved.  ``0`` leaves basins touching.
    enforce_connected : bool -- if True, relabel so each label is exactly one
        connected component (splits any piece fragmented by the background cut).
    flood_method : {'watershed', 'geodesic'} -- ``'watershed'`` uses the merge-tree
        collapse (recommended, full coverage, curvature-free).  ``'geodesic'`` keeps
        only the surviving summits as seeds and assigns a surface-Voronoi partition
        via the heat method (requires ``potpourri3d``).  ``'descend'`` is an alias
        of ``'watershed'``.
    classify : bool -- if True, populate ``region_types``.

    Returns
    -------
    MorseSplitResult
        ``seed_vertices[i]`` is the highest vertex of instance ``i + 1``.
    """
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    height = np.asarray(height, dtype=float)
    adjacency = igl.adjacency_list(F)

    # auto basal level from the height field, per cell
    if isinstance(background_level, str):
        if background_level == 'otsu':
            background_level = otsu_level_threshold(height, otsu_classes, otsu_level)
        elif background_level == 'mean':
            background_level = float(np.mean(height))
        elif background_level == 'adaptive_otsu':
            background_level = adaptive_otsu_threshold(
                height, F, smooth_iters=adaptive_smooth_iters, alpha=adaptive_alpha,
                otsu_classes=otsu_classes, otsu_level=otsu_level)
        else:
            raise ValueError(f"background_level: unknown option {background_level!r}")

    tree = compute_persistence_watershed(height, adjacency)
    tau = (auto_persistence_threshold(tree.persistence)
           if persistence_threshold is None else float(persistence_threshold))

    if flood_method == 'geodesic':
        keep = tree.persistence >= tau
        if not np.any(keep):
            keep[np.argmax(tree.persistence)] = True
        seeds = tree.summits[keep]
        raw = geodesic_marker_watershed(mesh, seeds)
        labels_out = raw.astype(np.int32)
    else:
        neck_curv = mean_curvature
        if neck_curv is not None and neck_curv_erode > 0:
            neck_curv = erode_curvature(neck_curv, adjacency, neck_curv_erode)
        labels_out, _ = merge_basins_to_threshold(
            tree, tau, min_region_size=min_region_size,
            mean_curvature=neck_curv, curv_neck_thresh=curv_neck_thresh,
        )

    # --- basal cut: define the protrusion domain from the height field ----
    if background_level is not None:
        labels_out[height < background_level] = 0

    domain_mask = labels_out > 0        # full protrusion domain (before fill suppression)

    # --- relative flood ceiling: suppress fill below each summit -----------
    # drops the deep neck between two merged summits so connectivity can split them
    if fill_depth is not None:
        for lab in np.setdiff1d(np.unique(labels_out), 0):
            mask = labels_out == lab
            top = height[mask].max()
            labels_out[mask & (height < top - fill_depth)] = 0

    # --- carve a gap where neighbouring instances touch -------------------
    if separate_rings > 0:
        labels_out = separate_touching_labels(labels_out, adjacency, separate_rings)

    # --- guarantee each label == one connected component ------------------
    if enforce_connected:
        labels_out = connected_component_relabel(labels_out, adjacency)

    # --- drop specks introduced by the cut / fragmentation ----------------
    if min_region_size > 0:
        for lab in np.setdiff1d(np.unique(labels_out), 0):
            if int(np.sum(labels_out == lab)) < min_region_size:
                labels_out[labels_out == lab] = 0

    # --- best of both: re-grow separated cap cores over the full domain ---
    if regrow and fill_depth is not None:
        labels_out = grow_labels_over_domain(labels_out, height, adjacency, domain_mask)

    # --- relabel contiguous, one seed (highest vertex) per instance -------
    uniq = np.setdiff1d(np.unique(labels_out), 0)
    remap = {o: i + 1 for i, o in enumerate(uniq)}
    relabelled = np.zeros_like(labels_out)
    seeds_out = []
    for o, nlab in remap.items():
        mask = labels_out == o
        relabelled[mask] = nlab
        seeds_out.append(int(np.where(mask)[0][np.argmax(height[mask])]))
    labels_out = relabelled

    region_types = {}
    if classify and ridgeness is not None:
        for lab in np.setdiff1d(np.unique(labels_out), 0):
            region_types[int(lab)] = classify_region(
                V, ridgeness, labels_out == lab, height=height
            )

    return MorseSplitResult(
        labels=labels_out,
        seed_vertices=np.asarray(seeds_out, dtype=np.int64),
        persistence_threshold=tau,
        persistences=tree.persistence,
        region_types=region_types,
    )
