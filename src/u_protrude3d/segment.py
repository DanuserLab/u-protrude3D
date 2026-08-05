from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io as spio
import scipy.stats as spstats
import skimage.filters as skfilters
import skimage.io as skio
import skimage.segmentation as sksegmentation
import skimage.morphology as skmorph
import skimage.measure as skmeasure
import matplotlib.pyplot as plt
import igl
import point_cloud_utils as pcu
from matplotlib import cm
from skimage.feature import peak_local_max

import unwrap3D.Mesh.meshtools as meshtools
import unwrap3D.Segmentation.segmentation as segmentation

from .config import SegmentConfig
from ._utils.thresholding import binary_threshold
from ._utils.curvature import compute_curvatures
from ._utils.mesh_ops import mesh_smooth_scalar
from ._utils.label_ops import remove_small_label_components
from ._utils.metrics import average_precision, _relabel_sequential_vertex_labels
from ._utils.io_utils import load_mesh, load_mat_labels, ensure_dir, build_save_paths
from ._utils.colors import get_vertex_colors, scalar_to_vertex_colors, export_colored_obj
from ._utils.geometry import pca_rotate_mesh, compute_planarity


@dataclass
class SegmentResult:
    """Return value of :func:`segment_protrusions`."""
    vertex_labels: np.ndarray
    vertex_labels_cc: np.ndarray       # initial CC labels before bleb/ridge splitting
    mesh_V: np.ndarray
    mesh_F: np.ndarray
    cMCF_steps: np.ndarray
    basal_binary: np.ndarray
    dists: np.ndarray
    H: np.ndarray
    H_gauss: np.ndarray
    H_mean: np.ndarray
    ridgeness: np.ndarray        # per-vertex ridgeness scalar
    shape_index: np.ndarray      # per-vertex Koenderink Shape Index (+1=dome, +0.5=cylinder)
    height: np.ndarray           # per-vertex height field (cumulative cMCF displacement)
    cMCF_stop_ind: int
    output_paths: dict
    ap: Optional[np.ndarray] = None


@dataclass
class InvariantResult:
    """Return value of :func:`segment_protrusions_invariant`."""
    vertex_labels: np.ndarray       # final per-instance labels (after SI splitting + basal mask)
    vertex_labels_cc: np.ndarray    # initial CC labels (before per-patch SI splitting)
    basal_binary: np.ndarray        # protrusion binary mask used for final masking
    mesh_V: np.ndarray
    mesh_F: np.ndarray
    total_surface_area: float
    height: np.ndarray              # smoothed cMCF displacement height
    shape_index: np.ndarray         # Koenderink SI ∈ [-1, 1]
    curvedness_norm: np.ndarray     # sqrt((k1²+k2²)/2) * sqrt(A)  — dimensionless
    ridgeness_norm: np.ndarray      # ridgeness * sqrt(A)            — dimensionless
    H_mean_norm: np.ndarray         # H_mean * sqrt(A)               — dimensionless, signed
    K_norm: np.ndarray              # k1*k2 * A                      — dimensionless, signed
    curv_anisotropy: np.ndarray     # (|k_max|-|k_min|)/(|k_max|+|k_min|+ε) ∈ [0, 1]
    principal_ratio: np.ndarray     # |k_min|/|k_max|                ∈ [0, 1]
    valleys_norm: np.ndarray        # valleys * sqrt(A)              — dimensionless
    output_paths: dict


def _local_height_maxima(vertices, faces, height, min_hops=3, eps=0.0):
    """Return vertex indices that are local height maxima on the mesh graph.

    A vertex qualifies if its height >= every 1-ring neighbour minus ``eps``.
    ``eps=0`` (default) is strict; ``eps>0`` allows plateau vertices to qualify
    so that smooth dome tips (e.g. shape-index plateaus) yield at least one peak.
    Nearby peaks are then suppressed by BFS: among peaks within ``min_hops``
    graph hops of each other only the tallest survives.
    """
    from collections import deque

    # Build adjacency list
    n = len(vertices)
    adj = [set() for _ in range(n)]
    for tri in faces:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        adj[i].update((j, k)); adj[j].update((i, k)); adj[k].update((i, j))

    # 1-ring local maxima
    raw_peaks = [v for v in range(n) if all(height[v] >= height[nb] - eps for nb in adj[v])]
    if not raw_peaks:
        return np.array([], dtype=int)

    # Suppress nearby peaks: iterate tallest first, BFS to flag suppressed
    raw_peaks_sorted = sorted(raw_peaks, key=lambda v: -height[v])
    suppressed = set()
    kept = []
    for v in raw_peaks_sorted:
        if v in suppressed:
            continue
        kept.append(v)
        # BFS up to min_hops hops
        queue = deque([(v, 0)])
        visited = {v}
        while queue:
            u, d = queue.popleft()
            if d >= min_hops:
                continue
            for nb in adj[u]:
                if nb not in visited:
                    visited.add(nb)
                    suppressed.add(nb)
                    queue.append((nb, d + 1))
    return np.array(kept, dtype=int)


def _invariant_saddle_merge(protrude_submesh, submesh_labels, vertex_height,
                             height_scale, saddle_depth_threshold, max_iters=20,
                             vertex_si=None, saddle_criterion='height',
                             saddle_si_threshold=0.5, saddle_ar_threshold=2.0):
    """Merge adjacent label pairs whose shared boundary is not a genuine valley.

    Three merge criteria (selected by ``saddle_criterion``):

    ``'height'`` (default):
        depth = (min(h̄_A, h̄_B) − h̄_saddle) / (min(h̄_A, h̄_B) + ε)
        merge if depth < saddle_depth_threshold  (0–1; ~0.2 = shallow valley)

    ``'shape_index'``:
        merge if mean(SI[boundary_verts]) > saddle_si_threshold
        High SI at the boundary → dome-like → likely same protrusion.
        Low SI → cylindrical neck / genuine structural boundary → keep split.

    ``'adaptive'``:
        Classify each label by aspect ratio (PCA of vertex positions).
        AR < saddle_ar_threshold → compact (bleb-like) → SI criterion.
        AR ≥ saddle_ar_threshold → elongated (ridge/lamellipodia) → height criterion.
        Cross-type pairs (compact + elongated) are never merged.

    Uses union-find so all qualifying pairs are collapsed in one pass per iter.
    Returns the updated (N,) label array.
    """
    faces = protrude_submesh.faces
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    labels = np.asarray(submesh_labels, dtype=np.int32).copy()
    verts = protrude_submesh.vertices

    use_si = saddle_criterion == 'shape_index' and vertex_si is not None
    use_adaptive = saddle_criterion == 'adaptive'

    for _ in range(max_iters):
        uniq = np.setdiff1d(np.unique(labels), 0)
        if len(uniq) < 2:
            break

        h_mean = {int(lb): float(np.mean(vertex_height[labels == lb])) for lb in uniq}

        # 'adaptive': classify each label by its mean SI (per-vertex local curvature character).
        # SI near +1 → dome-like (compact/bleb); SI near +0.5 → cylinder (ridge/lamellipodia).
        # This is robust to irregular fragment shapes, unlike spatial aspect ratio.
        if use_adaptive and vertex_si is not None:
            si_mean_per_label = {int(lb): float(np.mean(vertex_si[labels == lb])) for lb in uniq}

        la, lb_ = labels[edges[:, 0]], labels[edges[:, 1]]
        cross = (la > 0) & (lb_ > 0) & (la != lb_)
        if not cross.any():
            break

        cross_e = edges[cross]
        la_c, lb_c = la[cross], lb_[cross]
        pairs = np.unique(np.sort(np.stack([la_c, lb_c], axis=1), axis=1), axis=0)

        parent = {int(lb): int(lb) for lb in uniq}

        def _find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        merged = False
        for a, b in pairs:
            ra, rb = _find(int(a)), _find(int(b))
            if ra == rb:
                continue
            mask = ((la_c == a) & (lb_c == b)) | ((la_c == b) & (lb_c == a))
            boundary_verts = np.unique(cross_e[mask])

            if use_si:
                si_boundary = float(np.mean(vertex_si[boundary_verts]))
                do_merge = si_boundary > saddle_si_threshold
            elif use_adaptive:
                si_ra = si_mean_per_label.get(ra, 0.0) if vertex_si is not None else 0.0
                si_rb = si_mean_per_label.get(rb, 0.0) if vertex_si is not None else 0.0
                compact_ra = si_ra > saddle_ar_threshold   # saddle_ar_threshold reused as SI cutoff
                compact_rb = si_rb > saddle_ar_threshold
                if compact_ra != compact_rb:
                    do_merge = False  # cross-type (bleb + ridge): never merge
                elif compact_ra:
                    # both dome-like → SI boundary criterion
                    si_boundary = float(np.mean(vertex_si[boundary_verts]))
                    do_merge = si_boundary > saddle_si_threshold
                else:
                    # both elongated → height depth criterion
                    h_saddle = float(np.mean(vertex_height[boundary_verts]))
                    h_lower = min(h_mean[ra], h_mean[rb])
                    depth = (h_lower - h_saddle) / (h_lower + 1e-12)
                    do_merge = depth < saddle_depth_threshold
            else:
                h_saddle = float(np.mean(vertex_height[boundary_verts]))
                h_lower = min(h_mean[ra], h_mean[rb])
                depth = (h_lower - h_saddle) / (h_lower + 1e-12)
                do_merge = depth < saddle_depth_threshold

            if do_merge:
                parent[rb] = ra
                merged = True

        if not merged:
            break

        # Relabel: map each label to its root
        root_map = {lb: _find(int(lb)) for lb in uniq}
        # Compact to sequential ints
        roots = sorted(set(root_map.values()))
        compact = {r: i + 1 for i, r in enumerate(roots)}
        new_labels = np.zeros_like(labels)
        for lb in uniq:
            new_labels[labels == lb] = compact[root_map[lb]]
        labels = new_labels

    return labels


def _invariant_split_large_patch(mesh, face_labels_select, shape_index,
                                  si_method, si_otsu_n_levels, si_otsu_level,
                                  min_cc_size):
    """Split a large patch into SI-binarised sub-patches (one per high-SI CC).

    Uses the same imputed-submesh approach as ``_split_large_patch`` /
    ``_process_std_patch`` to handle multi-boundary topology: the complement
    of the largest inverted CC is used as the working face set, so topological
    holes inside the patch are filled in.

    Returns a list of global face-index arrays.  If no high-SI faces exist
    (or no CCs meet ``min_cc_size``), the original ``face_labels_select`` is
    returned as a single-element list.
    """
    face_binary = np.zeros(len(mesh.faces), dtype=bool)
    face_binary[face_labels_select] = True
    invert_mesh = meshtools.create_mesh(mesh.vertices, faces=mesh.faces[~face_binary])
    comps = meshtools.connected_components_mesh(
        invert_mesh, original_face_indices=np.where(~face_binary)[0]
    )
    largest = comps[np.argmax([len(cc) for cc in comps])]
    impute_face_select = np.setdiff1d(np.arange(len(mesh.faces)), largest)

    # Face-level SI from per-vertex values; threshold computed locally on this patch
    si_face = np.mean(shape_index[mesh.faces[impute_face_select]], axis=1)
    high_si_mask, _ = binary_threshold(si_face, method=si_method,
                                        n_otsu_levels=si_otsu_n_levels, level=si_otsu_level)
    high_si_local = np.where(high_si_mask)[0]
    high_si_global = impute_face_select[high_si_local]

    if len(high_si_global) == 0:
        return [face_labels_select]

    high_si_mesh = meshtools.create_mesh(mesh.vertices, faces=mesh.faces[high_si_global])
    ccs = meshtools.connected_components_mesh(high_si_mesh, original_face_indices=high_si_global)
    ccs = [cc for cc in ccs if len(cc) >= min_cc_size]

    return ccs if ccs else [face_labels_select]


def _invariant_process_patch(
    mesh, face_labels_select, all_boundary_loop,
    shape_index, face_areas,
    global_vertex_labels, max_label,
    si_method, si_otsu_n_levels, si_otsu_level,
    n_diffusion_iters,
    min_cc_size,
    height=None,
    height_scale=1.0,
    watershed_seeding=None,   # None → SI threshold; 'height'/'shape_index' → geodesic watershed
    ws_min_peak_dist=3,
    ws_smooth_iters=10,
    ws_peak_eps=0.0,
    ws_saddle_merge=True,
    ws_saddle_criterion='height',
    ws_saddle_depth_threshold=0.2,
    ws_saddle_si_threshold=0.5,
    ws_saddle_ar_threshold=2.0,
):
    """Label a single patch using shape-index binarisation + label spreading.

    Submesh construction mirrors ``_process_std_patch``: single-boundary
    patches use the patch faces directly; multi-boundary patches use the
    imputed face set (all faces minus the largest external CC) so that
    topological holes inside the patch are included.

    Within the submesh:
    1. Face-level SI is computed from per-vertex SI averaged over faces.
    2. Faces are binarised as protrusion peaks using the configured SI threshold method.
    3. Connected components of those faces become per-instance seeds.
    4. ``labelspreading_mesh`` fills the rest of the submesh from those seeds.
    5. Labels are offset by ``max_label`` and transferred to ``global_vertex_labels``
       via nearest-neighbour lookup (same as ``_map_labels_to_global``).

    Returns the updated ``max_label``.
    """
    if len(all_boundary_loop) == 1:
        unique_verts_cc = np.unique(mesh.faces[face_labels_select].ravel())
        unique_verts_coords = mesh.vertices[unique_verts_cc].copy()
        protrude_submesh, attributes = meshtools.submesh(
            mesh,
            faces_sequence=[face_labels_select],
            mesh_face_attributes=igl.average_onto_faces(mesh.faces, shape_index[:, None]).T,
            repair=False, only_watertight=False, min_faces=None,
        )
        protrude_submesh = protrude_submesh[0]
        attr = attributes[0]
    else:
        face_binary = np.zeros(len(mesh.faces), dtype=bool)
        face_binary[face_labels_select] = True
        invert_mesh = meshtools.create_mesh(mesh.vertices, faces=mesh.faces[~face_binary])
        comps = meshtools.connected_components_mesh(
            invert_mesh, original_face_indices=np.where(~face_binary)[0]
        )
        largest = comps[np.argmax([len(cc) for cc in comps])]
        impute_face_select = np.setdiff1d(np.arange(len(mesh.faces)), largest)
        unique_verts_cc = np.unique(mesh.faces[impute_face_select].ravel())
        unique_verts_coords = mesh.vertices[unique_verts_cc].copy()
        protrude_submesh, attributes = meshtools.submesh(
            mesh,
            faces_sequence=[impute_face_select],
            mesh_face_attributes=igl.average_onto_faces(mesh.faces, shape_index[:, None]).T,
            repair=False, only_watertight=False, min_faces=None,
        )
        protrude_submesh = protrude_submesh[0]
        attr = attributes[0]

    # Always compute vertex_si (needed by SI-threshold path and shape_index watershed)
    attr_1d = np.asarray(attr).ravel()
    vertex_si = igl.average_onto_vertices(
        protrude_submesh.vertices, protrude_submesh.faces,
        np.vstack([attr_1d] * 3).T,
    )[:, 0]

    # Always map height to submesh (needed by saddle merge and height/SI-watershed)
    vertex_h = None
    if height is not None:
        _, sub_to_global = pcu.k_nearest_neighbors(
            protrude_submesh.vertices, mesh.vertices, k=1
        )
        vertex_h = height[sub_to_global.ravel()]

    def _si_ccs():
        """Shared helper: find SI-threshold CCs on the submesh."""
        si_face = np.mean(vertex_si[protrude_submesh.faces], axis=1)
        high_si_mask, _ = binary_threshold(si_face, method=si_method,
                                            n_otsu_levels=si_otsu_n_levels, level=si_otsu_level)
        high_si_faces = np.where(high_si_mask)[0]
        if len(high_si_faces) == 0:
            return []
        peak_mesh = meshtools.create_mesh(
            protrude_submesh.vertices, faces=protrude_submesh.faces[high_si_faces]
        )
        ccs = meshtools.connected_components_mesh(peak_mesh, original_face_indices=high_si_faces)
        return [cc for cc in ccs if len(cc) >= min_cc_size]

    def _geodesic_from_face_components(face_components):
        """Run geodesic watershed from face_components → vertex label array (1-indexed)."""
        if not face_components or len(face_components) == 1:
            return np.ones(len(protrude_submesh.vertices), dtype=np.int32)
        _, face_label_arr, _ = meshtools.mesh_watershed_segmentation_faces(
            protrude_submesh, face_components, method='heat', return_solver=True,
        )
        face_label_arr = np.asarray(face_label_arr, dtype=np.int32).ravel() + 1
        labels = np.zeros(len(protrude_submesh.vertices), dtype=np.int32)
        for fi, tri in enumerate(protrude_submesh.faces):
            lbl = int(face_label_arr[fi])
            for vi in tri:
                if labels[vi] == 0:
                    labels[vi] = lbl
        return labels

    if watershed_seeding == 'height' and vertex_h is not None:
        # --- Height local-maxima → geodesic watershed ---
        seed_scalar = vertex_h.copy()
        if ws_smooth_iters > 0:
            seed_scalar = mesh_smooth_scalar(
                protrude_submesh, seed_scalar, delta=0.5, n_iters=ws_smooth_iters,
            )
        peak_verts = _local_height_maxima(
            protrude_submesh.vertices, protrude_submesh.faces,
            seed_scalar, min_hops=ws_min_peak_dist, eps=ws_peak_eps,
        )
        face_components = []
        for pv in peak_verts:
            seed_faces = np.where(np.any(protrude_submesh.faces == pv, axis=1))[0]
            if len(seed_faces) > 0:
                face_components.append(seed_faces[:1])
        submesh_labels = _geodesic_from_face_components(face_components)

    elif watershed_seeding == 'shape_index':
        # --- SI-threshold CCs → geodesic watershed ---
        # Uses the same robust CC seeds as the default path but fills via geodesic
        # distance (guaranteed complete, no unlabelled vertices) instead of labelspreading.
        si_ccs = _si_ccs()
        face_components = si_ccs if si_ccs else []
        submesh_labels = _geodesic_from_face_components(face_components)

    else:
        # --- SI-binarisation path (default, watershed_seeding is None) ---
        si_ccs = _si_ccs()

        if not si_ccs:
            submesh_labels = np.ones(len(protrude_submesh.vertices), dtype=np.int32)
        else:
            submesh_labels = np.zeros(len(protrude_submesh.vertices), dtype=np.int32)
            for cc_idx, cc_faces in enumerate(si_ccs):
                vv = np.unique(protrude_submesh.faces[cc_faces].ravel())
                submesh_labels[vv] = cc_idx + 1

            W = meshtools.vertex_geometric_affinity_matrix(
                protrude_submesh, gamma=None, eps=1e-12, alpha=0.25, normalize=True
            )
            submesh_labels = meshtools.labelspreading_mesh(
                v=protrude_submesh.vertices, f=protrude_submesh.faces,
                x=np.where(submesh_labels > 0)[0],
                y=submesh_labels[submesh_labels > 0],
                W=W, niters=n_diffusion_iters, alpha_prop=0.9,
                return_proba=False, renorm=False,
            )
            submesh_labels = np.asarray(submesh_labels, dtype=np.int32)

    # --- Universal saddle merge (all paths) ---
    if ws_saddle_merge and vertex_h is not None and len(np.unique(submesh_labels)) > 2:
        submesh_labels = _invariant_saddle_merge(
            protrude_submesh, submesh_labels, vertex_h,
            height_scale, ws_saddle_depth_threshold,
            vertex_si=vertex_si,
            saddle_criterion=ws_saddle_criterion,
            saddle_si_threshold=ws_saddle_si_threshold,
            saddle_ar_threshold=ws_saddle_ar_threshold,
        )

    return _map_labels_to_global(
        unique_verts_coords, protrude_submesh, np.asarray(submesh_labels, dtype=np.int32),
        global_vertex_labels, unique_verts_cc, max_label,
    )


def segment_protrusions_invariant(
    mesh_path: str | os.PathLike,
    save_dir: str | os.PathLike,
    cfg: Optional[SegmentConfig] = None,
) -> InvariantResult:
    """Run the pipeline through initial CC segmentation using scale-invariant measures.

    Identical to :func:`segment_protrusions` steps 1–7 (load, curvature, cMCF,
    height, smooth, initial binary, initial CCs), but after smoothing computes a
    suite of dimensionless curvature descriptors and exports a coloured .obj for
    each so they can be inspected visually before committing to per-patch splitting.

    Scale-invariant normalisation
    ------------------------------
    All curvature quantities have units 1/length; multiplying by ``sqrt(A)``
    (where A = total surface area in mesh units²) yields dimensionless values
    that are independent of cell size, allowing the same threshold to apply
    across cells imaged at different resolutions or physical sizes.

    Computed measures
    -----------------
    shape_index     : Koenderink SI = (2/π)·arctan2(k1+k2, k1-k2) ∈ [−1,+1]
    curvedness_norm : sqrt((k1²+k2²)/2) · sqrt(A)
    ridgeness_norm  : ridgeness · sqrt(A)
    H_mean_norm     : H_mean · sqrt(A)  (signed)
    K_norm          : k1·k2 · A         (signed, Gaussian curvature × area)
    curv_anisotropy : |k1−k2| / (|k1|+|k2|+ε) ∈ [0,1]
    principal_ratio : |k_min| / (|k_max|+ε) ∈ [0,1]  (umbilicity proxy)
    valleys_norm    : valleys · sqrt(A)
    """
    if cfg is None:
        cfg = SegmentConfig()

    np.random.seed(cfg.random_seed)
    save_dir = ensure_dir(save_dir)
    d = Path(save_dir)

    protrude_colors = np.vstack(
        __import__('seaborn').color_palette('Spectral', cfg.n_protrude_colors)
    )
    rng = np.random.default_rng(cfg.random_seed)
    rng.shuffle(protrude_colors)

    # ------------------------------------------------------------------
    # 1. Load and pre-process mesh
    # ------------------------------------------------------------------
    mesh = load_mesh(mesh_path)
    mesh.export(str(d / 'mesh.obj'))
    mesh = meshtools.read_mesh(
        str(d / 'mesh.obj'), process=True, validate=True, keep_largest_only=True
    )
    if cfg.use_isotropic_remesh:
        try:
            edge_length = np.nanmean(igl.edge_lengths(mesh.vertices, mesh.faces))
            mesh = meshtools.incremental_isotropic_remesh(
                mesh, target_edge_length=cfg.isotropic_remesh_edge_length_factor * edge_length
            )
        except Exception:
            mesh = meshtools.decimate_resample_mesh(mesh, remesh_samples=0.95, predecimate=True)
    else:
        mesh = meshtools.decimate_resample_mesh(mesh, remesh_samples=0.95, predecimate=True)

    # ------------------------------------------------------------------
    # 2. Curvature
    # ------------------------------------------------------------------
    curv = compute_curvatures(mesh.vertices, mesh.faces, radius=cfg.curvature_radius)
    k1, k2 = curv['k1'], curv['k2']
    H_mean   = curv['H_mean']
    H_gauss  = curv['H_gauss_norm']
    ridgeness = curv['ridgeness']
    valleys   = curv['valleys']
    shape_index = curv['shape_index']
    norm = curv['norm']

    # ------------------------------------------------------------------
    # 3. SDF binary + external cMCF
    # ------------------------------------------------------------------
    mesh_binary = meshtools.voxelize_image_mesh_pts(
        mesh,
        pad=25,
        dilate_ksize=cfg.sdf_binary_dilate_ksize,
        erode_ksize=cfg.sdf_binary_erode_ksize,
        pitch=1.2,
    )

    H_normal, sdf_vol_normal, sdf_vol = segmentation.mean_curvature_binary(
        mesh_binary > 0, smooth=1.0, mask=False, smooth_gradient=1, eps=1e-12
    )

    w_curv = meshtools._normalize99(H_normal)
    if cfg.cmcf.erosion_only:
        vi = np.clip(mesh.vertices.astype(int), 0, np.array(mesh_binary.shape) - 1)
        wv = w_curv[vi[:, 0], vi[:, 1], vi[:, 2]]
        wv = wv[np.isfinite(wv)]
        tau = float(skfilters.threshold_otsu(wv)) if wv.size else 0.0
        w_curv = np.clip(w_curv - tau, 0.0, None)
    modified_sdf = w_curv[..., None] * sdf_vol_normal.transpose(1, 2, 3, 0)

    Usteps = meshtools.parametric_mesh_constant_img_flow(
        mesh,
        external_img_gradient=modified_sdf,
        niters=cfg.cmcf.n_iters,
        deltaL=cfg.cmcf.deltaL,
        step_size=cfg.cmcf.step_size,
        method='implicit',
        robust_L=True,
        mollify_factor=cfg.cmcf.mollify_factor,
        conformalize=False,
        gamma=1,
        alpha=0.2,
        beta=0.5,
        eps=1e-20,
        solver=cfg.cmcf.solver,
        noprogress=False,
        normalize_grad=False,
    )

    if cfg.cmcf.extra_smooth:
        Usteps_smooth = []
        for ttt in range(Usteps.shape[-1]):
            if ttt == 0:
                Usteps_smooth.append(Usteps[..., ttt])
            else:
                Ustep_mesh = meshtools.create_mesh(Usteps[..., ttt], faces=mesh.faces)
                U, _, _ = meshtools.conformalized_mean_curvature_flow(
                    Ustep_mesh,
                    max_iter=1,
                    rescale_output=True,
                    delta=cfg.cmcf.deltaL_smooth,
                    conformalize=True,
                    robust_L=True,
                    mollify_factor=cfg.cmcf.mollify_factor,
                    solver=cfg.cmcf.solver,
                )
                Usteps_smooth.append(U[..., -1])
        Usteps = np.array(Usteps_smooth).transpose(1, 2, 0)

    if cfg.cmcf.erosion_project_binary:
        Usteps = _project_usteps_into_binary(Usteps, mesh_binary, sdf_vol, sdf_vol_normal)

    gauss_steps = np.array([
        np.nanmean(np.abs(igl.gaussian_curvature(Usteps[..., t], mesh.faces)))
        for t in range(Usteps.shape[-1])
    ])
    ind = int(np.argmin(gauss_steps)) + cfg.offset_ref_ind

    fig_mcf, ax_mcf = plt.subplots(figsize=(8, 4))
    ax_mcf.plot(gauss_steps)
    ax_mcf.vlines(ind, gauss_steps.min(), gauss_steps.max(), color='k', linestyles='dashed')
    ax_mcf.set_ylabel('Mean |Gaussian curvature|')
    ax_mcf.set_xlabel('cMCF iteration')
    ax_mcf.set_title('Reference iteration determination')
    fig_mcf.tight_layout()
    fig_mcf.savefig(str(d / 'external_MCF_iteration_determination.svg'), dpi=300, bbox_inches='tight')
    plt.close(fig_mcf)

    # ------------------------------------------------------------------
    # 4. Protrusion height
    # ------------------------------------------------------------------
    smooth_iters = cfg.n_smooth_scalar_fn_iters
    dists = np.nansum(np.linalg.norm(np.diff(Usteps, axis=-1), axis=1)[:, :ind], axis=-1)

    ref_mesh = mesh.copy()
    ref_mesh.vertices = Usteps[..., ind].copy()
    interior_bool = mesh_binary[
        ref_mesh.vertices[:, 0].astype(int),
        ref_mesh.vertices[:, 1].astype(int),
        ref_mesh.vertices[:, 2].astype(int),
    ]
    dists[interior_bool == 0] *= -1

    face_areas = igl.doublearea(mesh.vertices, mesh.faces) / 2.0
    A     = float(np.sum(face_areas))
    sqA   = float(np.sqrt(A))
    eps_c = 1e-12
    _raw_curvedness     = np.sqrt((k1**2 + k2**2) / 2.0)
    _raw_kmax = np.where(np.abs(k1) >= np.abs(k2), np.abs(k1), np.abs(k2))
    _raw_kmin = np.where(np.abs(k1) >= np.abs(k2), np.abs(k2), np.abs(k1))
    spio.savemat(
        str(d / 'raw_surface_stats.mat'),
        {'normalizing_H': norm, 'H_gauss_norm': H_gauss, 'H_mean': H_mean,
         'ridgeness': ridgeness, 'valleys': valleys, 'dists': dists, 'k1': k1, 'k2': k2,
         'shape_index': shape_index,
         'curvedness_norm':   _raw_curvedness * sqA,
         'ridgeness_norm':    ridgeness * sqA,
         'H_mean_norm':       H_mean * sqA,
         'K_norm':            k1 * k2 * A,
         'curv_anisotropy':   (_raw_kmax - _raw_kmin) / (_raw_kmax + _raw_kmin + eps_c),
         'principal_ratio':   _raw_kmin / (_raw_kmax + eps_c),
         'valleys_norm':      valleys * sqA},
    )

    # ------------------------------------------------------------------
    # 5. Smooth all scalar fields
    # ------------------------------------------------------------------
    dists       = mesh_smooth_scalar(mesh, dists,       delta=0.5, n_iters=smooth_iters)
    ridgeness   = mesh_smooth_scalar(mesh, ridgeness,   delta=0.5, n_iters=smooth_iters)
    valleys     = mesh_smooth_scalar(mesh, valleys,     delta=0.5, n_iters=smooth_iters)
    H_mean      = mesh_smooth_scalar(mesh, H_mean,      delta=0.5, n_iters=smooth_iters)
    H_gauss     = mesh_smooth_scalar(mesh, H_gauss,     delta=0.5, n_iters=smooth_iters)
    shape_index = mesh_smooth_scalar(mesh, shape_index, delta=0.5, n_iters=smooth_iters)
    k1          = mesh_smooth_scalar(mesh, k1,          delta=0.5, n_iters=smooth_iters)
    k2          = mesh_smooth_scalar(mesh, k2,          delta=0.5, n_iters=smooth_iters)

    # ------------------------------------------------------------------
    # 6. Scale-invariant measures
    # ------------------------------------------------------------------
    curvedness     = np.sqrt((k1**2 + k2**2) / 2.0)
    curvedness_norm = curvedness * sqA
    ridgeness_norm  = ridgeness  * sqA
    H_mean_norm     = H_mean     * sqA
    K               = k1 * k2
    K_norm          = K * A
    kmax = np.where(np.abs(k1) >= np.abs(k2), np.abs(k1), np.abs(k2))
    kmin = np.where(np.abs(k1) >= np.abs(k2), np.abs(k2), np.abs(k1))
    # R = (|k_max| - |k_min|) / (|k_max| + |k_min| + ε): 0 = isotropic/dome, 1 = anisotropic/ridge
    curv_anisotropy = (np.abs(kmax) - np.abs(kmin)) / (np.abs(kmax) + np.abs(kmin) + eps_c)
    principal_ratio = kmin / (kmax + eps_c)
    valleys_norm    = valleys * sqA

    spio.savemat(
        str(d / 'smooth_surface_stats.mat'),
        {'smooth_H_gauss_norm': H_gauss, 'smooth_H_mean': H_mean,
         'smooth_ridgeness': ridgeness, 'smooth_valleys': valleys,
         'smooth_dists': dists, 'smooth_iters': smooth_iters,
         'smooth_shape_index':    shape_index,
         'smooth_curvedness_norm': curvedness_norm,
         'smooth_ridgeness_norm':  ridgeness_norm,
         'smooth_H_mean_norm':     H_mean_norm,
         'smooth_K_norm':          K_norm,
         'smooth_curv_anisotropy': curv_anisotropy,
         'smooth_principal_ratio': principal_ratio,
         'smooth_valleys_norm':    valleys_norm},
    )

    # Percentile helpers for symmetric / one-sided clamping
    def _pct(arr, q):
        return float(np.percentile(arr, q))

    # ------------------------------------------------------------------
    # 7. Export per-measure coloured meshes
    # ------------------------------------------------------------------
    def _export(scalar, cmap, fname, vmin=None, vmax=None):
        export_colored_obj(
            mesh, scalar_to_vertex_colors(scalar, cmap, vmin=vmin, vmax=vmax),
            d / fname,
        )

    _export(dists, cm.coolwarm, 'inv_height.obj')
    _export(shape_index, cm.coolwarm, 'inv_shape_index.obj', vmin=-1.0, vmax=1.0)

    cn99 = _pct(curvedness_norm, 99)
    _export(curvedness_norm, cm.coolwarm, 'inv_curvedness_norm.obj', vmin=0.0, vmax=cn99)

    rn99 = _pct(ridgeness_norm, 99)
    _export(ridgeness_norm, cm.coolwarm, 'inv_ridgeness_norm.obj', vmin=0.0, vmax=rn99)

    hn99 = _pct(np.abs(H_mean_norm), 99)
    _export(H_mean_norm, cm.coolwarm, 'inv_H_mean_norm.obj', vmin=-hn99, vmax=hn99)

    kn99 = _pct(np.abs(K_norm), 99)
    _export(K_norm, cm.coolwarm, 'inv_K_norm.obj', vmin=-kn99, vmax=kn99)

    _export(curv_anisotropy, cm.coolwarm, 'inv_curv_anisotropy.obj', vmin=0.0, vmax=1.0)
    _export(principal_ratio, cm.coolwarm, 'inv_principal_ratio.obj', vmin=0.0, vmax=1.0)

    vn99 = _pct(valleys_norm, 99)
    _export(valleys_norm, cm.coolwarm, 'inv_valleys_norm.obj', vmin=0.0, vmax=vn99)

    # ------------------------------------------------------------------
    # 8. Initial height binarization
    # ------------------------------------------------------------------
    icfg = cfg.initial_height
    if icfg.use_auto:
        if icfg.use_mean:
            threshold = np.nanmean(dists)
        elif icfg.use_otsu:
            threshold = skfilters.threshold_multiotsu(dists, icfg.otsu_n_levels)[icfg.otsu_level]
        else:
            threshold = np.nanmean(dists)
    else:
        threshold = icfg.manual_threshold

    binary_dists = dists >= threshold

    W_geom = meshtools.vertex_geometric_affinity_matrix(
        mesh, gamma=None, eps=1e-12, alpha=0.5, normalize=True
    )
    z_labels, _ = meshtools.labelspreading_mesh_binary(
        mesh.vertices, mesh.faces,
        (binary_dists == 1) * 1,
        W=W_geom, niters=icfg.prop_iters, return_proba=True, thresh=icfg.prop_rebinarize,
    )
    binary_dists_new = meshtools.remove_small_mesh_label_holes_binary(
        mesh.vertices, mesh.faces,
        labels=np.array(z_labels) * 1,
        vertex_labels_bool=True, physical_size=True, minsize=5,
    )

    export_colored_obj(mesh, _binary_colors(binary_dists_new), d / 'inv_height_binary.obj')

    # ------------------------------------------------------------------
    # 9. Initial connected components
    # ------------------------------------------------------------------
    binary_faces = spstats.mode(binary_dists_new[mesh.faces] * 1, axis=1)[0]
    binary_faces = np.squeeze(binary_faces)
    binary_mesh = meshtools.create_mesh(mesh.vertices, faces=mesh.faces[binary_faces > 0])
    protrusion_cc_labels = meshtools.connected_components_mesh(
        binary_mesh, original_face_indices=np.where(binary_faces > 0)[0]
    )
    protrusion_cc_labels = [cc for cc in protrusion_cc_labels if len(cc) >= cfg.min_size_comps_initial]

    cc_vertex_labels = np.zeros(len(mesh.vertices), dtype=np.uint32)
    for cc_idx, cc_faces in enumerate(protrusion_cc_labels):
        cc_vertex_labels[np.unique(mesh.faces[cc_faces].ravel())] = cc_idx + 1

    cc_colors = get_vertex_colors(cc_vertex_labels, palette=protrude_colors)
    cc_label_faces = spstats.mode(cc_vertex_labels[mesh.faces] * 1, axis=-1)[0]
    if len(cc_label_faces.shape) == 2:
        cc_label_faces = cc_label_faces[:, 0]
    cc_boundary_verts = []
    for cc in np.setdiff1d(np.unique(cc_vertex_labels), 0):
        cc_faces = mesh.faces[cc_label_faces == cc]
        if len(cc_faces) == 0:
            continue
        b_loop = igl.boundary_loop(cc_faces)
        if len(b_loop) > 0:
            cc_boundary_verts.append(b_loop)
    if cc_boundary_verts:
        cc_colors[np.hstack(cc_boundary_verts)] = [0, 0, 0]
    export_colored_obj(mesh, cc_colors, d / 'inv_initial_cc_labels.obj')

    # ------------------------------------------------------------------
    # 10. Per-patch labelling using shape index
    # ------------------------------------------------------------------
    # Two-pass: large patches are first split into SI-based sub-regions
    # (analogous to _split_large_patch but using SI instead of H).
    # All patches (original std + sub-patches) are then processed by the
    # same invariant helper: binarise SI, find CCs, label-spread, map back.
    area_protrusions = np.array([np.sum(face_areas[cc]) for cc in protrusion_cc_labels])
    thresh_area = float(np.maximum(
        cfg.large_patch.min_max_area,
        cfg.large_patch.max_area_thresh_factor * np.nanmedian(area_protrusions),
    ))
    flagged_large = set(np.where(area_protrusions >= thresh_area)[0])

    global_vertex_labels = np.zeros(len(mesh.vertices), dtype=np.uint32)
    max_label = 0

    # 99th-percentile protrusive height — normalization scale for saddle-depth threshold
    protrusive_h = dists[binary_dists_new > 0]
    height_scale = float(np.percentile(protrusive_h, 99)) if len(protrusive_h) > 0 else 1.0

    _plot_patch_areas(area_protrusions, thresh_area, d / 'initial_patch_areas.svg')

    icfg_inv = cfg.invariant

    # Pass 1: split large patches
    refined_ccs = []
    for jjj, face_labels_select in enumerate(protrusion_cc_labels):
        if jjj in flagged_large:
            sub_patches = _invariant_split_large_patch(
                mesh, face_labels_select, shape_index,
                icfg_inv.si_segment_method,
                icfg_inv.si_segment_otsu_n_levels,
                icfg_inv.si_segment_otsu_level,
                cfg.min_size_comps_protrude_patch,
            )
            refined_ccs.extend(sub_patches)
        else:
            refined_ccs.append(face_labels_select)

    # Pass 2: per-patch labelling
    for face_labels_select in refined_ccs:
        all_boundary_loop = igl.all_boundary_loop(mesh.faces[face_labels_select])
        max_label = _invariant_process_patch(
            mesh, face_labels_select, all_boundary_loop,
            shape_index, face_areas,
            global_vertex_labels, max_label,
            icfg_inv.si_segment_method,
            icfg_inv.si_segment_otsu_n_levels,
            icfg_inv.si_segment_otsu_level,
            icfg_inv.n_diffusion_iters,
            cfg.min_size_comps_protrude_patch,
            height=dists,
            height_scale=height_scale,
            watershed_seeding=icfg_inv.watershed_seeding,
            ws_min_peak_dist=icfg_inv.ws_min_peak_dist,
            ws_smooth_iters=icfg_inv.ws_smooth_iters,
            ws_peak_eps=icfg_inv.ws_peak_eps,
            ws_saddle_merge=icfg_inv.ws_saddle_merge,
            ws_saddle_criterion=icfg_inv.ws_saddle_criterion,
            ws_saddle_depth_threshold=icfg_inv.ws_saddle_depth_threshold,
            ws_saddle_si_threshold=icfg_inv.ws_saddle_si_threshold,
            ws_saddle_ar_threshold=icfg_inv.ws_saddle_ar_threshold,
        )

    # ------------------------------------------------------------------
    # 11. Basal binary mask (identical to segment_protrusions)
    # ------------------------------------------------------------------
    H_combined = 0.5 * (H_gauss + H_mean)

    threshold_lower = skfilters.threshold_multiotsu(dists, 2)[0]
    basal_dists = dists >= threshold_lower
    W_geom2 = meshtools.vertex_geometric_affinity_matrix(
        mesh, gamma=None, eps=1e-12, alpha=0.25, normalize=True
    )
    basal_dists, _ = meshtools.labelspreading_mesh_binary(
        mesh.vertices, mesh.faces, (basal_dists == 1) * 1,
        W=W_geom2, niters=5, return_proba=True, thresh=0.25,
    )
    binary_dists_lower = meshtools.remove_small_mesh_label_holes_binary(
        mesh.vertices, mesh.faces, labels=np.array(basal_dists) * 1,
        vertex_labels_bool=True, physical_size=True, minsize=5,
    )

    threshold_curv = skfilters.threshold_multiotsu(H_combined, 2)[-1]
    basal_curv = H_combined >= threshold_curv
    W_geom3 = meshtools.vertex_geometric_affinity_matrix(
        mesh, gamma=None, eps=1e-12, alpha=0.5, normalize=True
    )
    basal_curv, _ = meshtools.labelspreading_mesh_binary(
        mesh.vertices, mesh.faces, (basal_curv == 1) * 1,
        W=W_geom3, niters=5, return_proba=True, thresh=0.25,
    )
    basal_binary = np.logical_or(binary_dists_lower, np.array(basal_curv) > 0)

    export_colored_obj(mesh, _binary_colors(basal_binary), d / 'inv_basal_binary.obj')

    # Final label spreading + basal mask
    W_final = meshtools.vertex_geometric_affinity_matrix(
        mesh, gamma=None, eps=1e-12, alpha=0.5, normalize=True
    )
    protrusion_labels_final = meshtools.labelspreading_mesh(
        v=mesh.vertices, f=mesh.faces,
        x=np.where(global_vertex_labels > 0)[0],
        y=global_vertex_labels[global_vertex_labels > 0],
        W=W_final, niters=5, alpha_prop=0.5, return_proba=False, renorm=False,
    )
    global_vertex_labels = protrusion_labels_final * basal_binary
    global_vertex_labels = remove_small_label_components(mesh, global_vertex_labels, min_size=5)

    # ------------------------------------------------------------------
    # 12. Export final coloured mesh
    # ------------------------------------------------------------------
    final_colors = get_vertex_colors(global_vertex_labels, palette=protrude_colors)
    final_label_faces = spstats.mode(global_vertex_labels[mesh.faces] * 1, axis=-1)[0]
    if len(final_label_faces.shape) == 2:
        final_label_faces = final_label_faces[:, 0]
    final_boundary_verts = []
    for cc in np.setdiff1d(np.unique(global_vertex_labels), 0):
        cc_faces = mesh.faces[final_label_faces == cc]
        if len(cc_faces) == 0:
            continue
        b_loop = igl.boundary_loop(cc_faces)
        if len(b_loop) > 0:
            final_boundary_verts.append(b_loop)
    if final_boundary_verts:
        final_colors[np.hstack(final_boundary_verts)] = [0, 0, 0]
    export_colored_obj(mesh, final_colors, d / 'inv_final_labels.obj')

    spio.savemat(
        str(d / 'instance_protrusion_segmentation_stats.mat'),
        {
            'protrusion_labels':            global_vertex_labels,
            'protrusion_labels_initial_cc': cc_vertex_labels,
            'external_cMCF_steps':          Usteps.astype(np.float32),
            'basal_binary':                 np.asarray(basal_binary),
            'protrusion_dists':             dists,
            'protrusion_H_mean':            H_mean,
            'protrusion_H_gauss':           H_gauss,
            'cMCF_stop_ind':                ind,
        },
    )

    out_paths = {
        'height_obj':          d / 'inv_height.obj',
        'shape_index_obj':     d / 'inv_shape_index.obj',
        'curvedness_norm_obj': d / 'inv_curvedness_norm.obj',
        'ridgeness_norm_obj':  d / 'inv_ridgeness_norm.obj',
        'H_mean_norm_obj':     d / 'inv_H_mean_norm.obj',
        'K_norm_obj':          d / 'inv_K_norm.obj',
        'curv_anisotropy_obj': d / 'inv_curv_anisotropy.obj',
        'principal_ratio_obj': d / 'inv_principal_ratio.obj',
        'valleys_norm_obj':    d / 'inv_valleys_norm.obj',
        'height_binary_obj':   d / 'inv_height_binary.obj',
        'initial_cc_obj':      d / 'inv_initial_cc_labels.obj',
        'basal_binary_obj':    d / 'inv_basal_binary.obj',
        'final_labels_obj':    d / 'inv_final_labels.obj',
        'raw_stats_mat':       d / 'raw_surface_stats.mat',
        'smooth_stats_mat':    d / 'smooth_surface_stats.mat',
        'instance_stats_mat':  d / 'instance_protrusion_segmentation_stats.mat',
        'patch_area_plot_svg': d / 'initial_patch_areas.svg',
        'cMCF_plot_svg':       d / 'external_MCF_iteration_determination.svg',
    }

    return InvariantResult(
        vertex_labels=global_vertex_labels,
        vertex_labels_cc=cc_vertex_labels,
        basal_binary=np.asarray(basal_binary),
        mesh_V=mesh.vertices,
        mesh_F=mesh.faces,
        total_surface_area=A,
        height=dists,
        shape_index=shape_index,
        curvedness_norm=curvedness_norm,
        ridgeness_norm=ridgeness_norm,
        H_mean_norm=H_mean_norm,
        K_norm=K_norm,
        curv_anisotropy=curv_anisotropy,
        principal_ratio=principal_ratio,
        valleys_norm=valleys_norm,
        output_paths={k: str(v) for k, v in out_paths.items()},
    )


def segment_protrusions(
    mesh_path: str | os.PathLike,
    save_dir: str | os.PathLike,
    cfg: Optional[SegmentConfig] = None,
    tif_path: Optional[str | os.PathLike] = None,
    gt_label_path: Optional[str | os.PathLike] = None,
) -> SegmentResult:
    """Run the full 3D protrusion segmentation pipeline on a single cell mesh.

    Parameters
    ----------
    mesh_path : str or Path
        .obj or .mat file containing the cell surface mesh.
    save_dir : str or Path
        Directory where output .obj, .mat, and figure files are written.
    cfg : SegmentConfig or None
        Algorithm parameters.  Uses defaults when None.
    tif_path : str, Path or None
        Optional TIFF volume used to build the SDF binary and map GT labels.
    gt_label_path : str, Path or None
        Optional .tif ground-truth label volume for inline AP scoring.
        When provided, the TIFF is loaded and AP is computed against GT.
        (Requires *tif_path* as well.)

    Returns
    -------
    SegmentResult
    """
    if cfg is None:
        cfg = SegmentConfig()

    np.random.seed(cfg.random_seed)
    save_dir = ensure_dir(save_dir)
    paths = build_save_paths(save_dir, '')

    # Colour palette
    protrude_colors = np.vstack(
        __import__('seaborn').color_palette('Spectral', cfg.n_protrude_colors)
    )
    rng = np.random.default_rng(cfg.random_seed)
    rng.shuffle(protrude_colors)

    # ------------------------------------------------------------------
    # 1. Load and pre-process mesh
    # ------------------------------------------------------------------
    mesh = load_mesh(mesh_path)
    mesh.export(str(paths['mesh_obj']))
    mesh = meshtools.read_mesh(
        str(paths['mesh_obj']), process=True, validate=True, keep_largest_only=True
    )
    if cfg.use_isotropic_remesh:
        try:
            edge_length = np.nanmean(igl.edge_lengths(mesh.vertices, mesh.faces))
            mesh = meshtools.incremental_isotropic_remesh(
                mesh, target_edge_length=cfg.isotropic_remesh_edge_length_factor * edge_length
            )
        except Exception:
            mesh = meshtools.decimate_resample_mesh(mesh, remesh_samples=0.95, predecimate=True)
    else:
        mesh = meshtools.decimate_resample_mesh(mesh, remesh_samples=0.95, predecimate=True)

    # ------------------------------------------------------------------
    # 2. Optional GT labelling from TIFF
    # ------------------------------------------------------------------
    protrude_labels_gt = None
    if tif_path is not None:
        tif = skio.imread(str(tif_path))
        tif_expand = sksegmentation.expand_labels(tif, distance=2)
        protrude_labels_gt = tif_expand[
            (mesh.vertices[:, 2] - 1).astype(int),
            (mesh.vertices[:, 1] - 1).astype(int),
            (mesh.vertices[:, 0] - 1).astype(int),
        ]
        gt_palette = np.vstack(
            __import__('seaborn').color_palette('Spectral', int(protrude_labels_gt.max()))
        )
        gt_colors = np.ones((len(mesh.vertices), 3)) * 0.75
        for cc in np.setdiff1d(np.unique(protrude_labels_gt), [0, 1]):
            gt_colors[protrude_labels_gt == cc] = gt_palette[cc - 1]

        spio.savemat(
            str(paths['gt_labels_mat']),
            {
                'v': mesh.vertices,
                'f': mesh.faces,
                'protrude_labels': protrude_labels_gt,
                'protrude_labels_vertex_colors': np.uint8(255 * gt_colors),
                'protrude_labels_gt_palette': gt_palette,
            },
        )
        gt_mesh = meshtools.create_mesh(
            mesh.vertices, mesh.faces, vertex_colors=np.uint8(255 * gt_colors)
        )
        if gt_mesh.volume < 0:
            gt_mesh.faces = gt_mesh.faces[:, ::-1]
        gt_mesh.export(str(paths['mesh_gt_color_obj']))

    # ------------------------------------------------------------------
    # 3. Curvature
    # ------------------------------------------------------------------
    curv = compute_curvatures(mesh.vertices, mesh.faces, radius=cfg.curvature_radius)
    k1, k2 = curv['k1'], curv['k2']
    norm = curv['norm']
    H_gauss = curv['H_gauss_norm']
    H_mean = curv['H_mean']
    ridgeness = curv['ridgeness']
    valleys = curv['valleys']
    shape_index = curv['shape_index']

    # ------------------------------------------------------------------
    # 4. SDF binary + external cMCF
    # ------------------------------------------------------------------
    mesh_binary = meshtools.voxelize_image_mesh_pts(
        mesh,
        pad=25,
        dilate_ksize=cfg.sdf_binary_dilate_ksize,
        erode_ksize=cfg.sdf_binary_erode_ksize,
        pitch=1.2,
    )

    H_normal, sdf_vol_normal, sdf_vol = segmentation.mean_curvature_binary(
        mesh_binary > 0, smooth=1.0, mask=False, smooth_gradient=1, eps=1e-12
    )

    w_curv = meshtools._normalize99(H_normal)
    if cfg.cmcf.erosion_only:
        # Discover the basal curvature level from the surface-sampled weights (Otsu),
        # then drive the flow only where curvature exceeds it: protrusions (convex,
        # high curvature) erode inward, near-basal regions get ~zero force and are
        # preserved.  No H_basal input needed -- the basal level is discovered.
        vi = np.clip(mesh.vertices.astype(int), 0, np.array(mesh_binary.shape) - 1)
        wv = w_curv[vi[:, 0], vi[:, 1], vi[:, 2]]
        wv = wv[np.isfinite(wv)]
        tau = float(skfilters.threshold_otsu(wv)) if wv.size else 0.0
        w_curv = np.clip(w_curv - tau, 0.0, None)
    modified_sdf = w_curv[..., None] * sdf_vol_normal.transpose(1, 2, 3, 0)

    Usteps = meshtools.parametric_mesh_constant_img_flow(
        mesh,
        external_img_gradient=modified_sdf,
        niters=cfg.cmcf.n_iters,
        deltaL=cfg.cmcf.deltaL,
        step_size=cfg.cmcf.step_size,
        method='implicit',
        robust_L=True,
        mollify_factor=cfg.cmcf.mollify_factor,
        conformalize=False,
        gamma=1,
        alpha=0.2,
        beta=0.5,
        eps=1e-20,
        solver=cfg.cmcf.solver,
        noprogress=False,
        normalize_grad=False,
    )

    if cfg.cmcf.extra_smooth:
        Usteps_smooth = []
        for ttt in range(Usteps.shape[-1]):
            if ttt == 0:
                Usteps_smooth.append(Usteps[..., ttt])
            else:
                Ustep_mesh = meshtools.create_mesh(Usteps[..., ttt], faces=mesh.faces)
                U, _, _ = meshtools.conformalized_mean_curvature_flow(
                    Ustep_mesh,
                    max_iter=1,
                    rescale_output=True,
                    delta=cfg.cmcf.deltaL_smooth,
                    conformalize=True,
                    robust_L=True,
                    mollify_factor=cfg.cmcf.mollify_factor,
                    solver=cfg.cmcf.solver,
                )
                Usteps_smooth.append(U[..., -1])
        Usteps = np.array(Usteps_smooth).transpose(1, 2, 0)

    if cfg.cmcf.erosion_project_binary:
        Usteps = _project_usteps_into_binary(Usteps, mesh_binary, sdf_vol, sdf_vol_normal)

    # Choose reference iteration
    gauss_steps = np.array([
        np.nanmean(np.abs(igl.gaussian_curvature(Usteps[..., t], mesh.faces)))
        for t in range(Usteps.shape[-1])
    ])
    ind = int(np.argmin(gauss_steps)) + cfg.offset_ref_ind

    # Save cMCF reference
    ref_mesh = mesh.copy()
    ref_mesh.vertices = Usteps[..., ind].copy()
    ref_mesh.export(str(paths['cMCF_ref_obj']))

    fig, ax = plt.subplots()
    ax.plot(gauss_steps)
    ax.vlines(ind, gauss_steps.min(), gauss_steps.max(), color='k', linestyles='dashed')
    ax.set_ylabel('Discrete Gauss Curvature')
    ax.set_xlabel('external MCF Iterations')
    fig.savefig(str(paths['cMCF_plot_svg']), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5. Protrusion height
    # ------------------------------------------------------------------
    smooth_iters = cfg.n_smooth_scalar_fn_iters
    dists = np.nansum(np.linalg.norm(np.diff(Usteps, axis=-1), axis=1)[:, :ind], axis=-1)

    interior_bool = mesh_binary[
        ref_mesh.vertices[:, 0].astype(int),
        ref_mesh.vertices[:, 1].astype(int),
        ref_mesh.vertices[:, 2].astype(int),
    ]
    dists[interior_bool == 0] *= -1

    # Save raw stats
    spio.savemat(
        str(paths['raw_stats_mat']),
        {'normalizing_H': norm, 'H_gauss_norm': H_gauss, 'H_mean': H_mean,
         'ridgeness': ridgeness, 'valleys': valleys, 'dists': dists, 'k1': k1, 'k2': k2},
    )

    # Smooth all scalar fields
    dists = mesh_smooth_scalar(mesh, dists, delta=0.5, n_iters=smooth_iters)
    ridgeness = mesh_smooth_scalar(mesh, ridgeness, delta=0.5, n_iters=smooth_iters)
    valleys = mesh_smooth_scalar(mesh, valleys, delta=0.5, n_iters=smooth_iters)
    H_gauss = mesh_smooth_scalar(mesh, H_gauss, delta=0.5, n_iters=smooth_iters)
    H_mean = mesh_smooth_scalar(mesh, H_mean, delta=0.5, n_iters=smooth_iters)
    shape_index = mesh_smooth_scalar(mesh, shape_index, delta=0.5, n_iters=smooth_iters)

    spio.savemat(
        str(paths['smooth_stats_mat']),
        {'smooth_H_gauss_norm': H_gauss, 'smooth_H_mean': H_mean,
         'smooth_ridgeness': ridgeness, 'smooth_valleys': valleys,
         'smooth_dists': dists, 'smooth_iters': smooth_iters},
    )

    # Export height and curvature colour meshes
    export_colored_obj(
        mesh, scalar_to_vertex_colors(dists, cm.coolwarm), paths['height_color_obj']
    )
    export_colored_obj(
        mesh,
        scalar_to_vertex_colors(H_gauss / cfg.voxel_size, cm.Spectral_r, vmin=-1.0, vmax=1.0),
        paths['curvature_color_obj'],
    )
    export_colored_obj(
        mesh,
        scalar_to_vertex_colors(shape_index, cm.coolwarm, vmin=-1.0, vmax=1.0),
        paths['shape_index_color_obj'],
    )

    # ------------------------------------------------------------------
    # 6. Initial height binarization
    # ------------------------------------------------------------------
    icfg = cfg.initial_height
    if icfg.use_auto:
        if icfg.use_mean:
            threshold = np.nanmean(dists)
        elif icfg.use_otsu:
            print('hellllloooo')
            threshold = skfilters.threshold_multiotsu(dists, icfg.otsu_n_levels)[icfg.otsu_level]
        else:
            threshold = np.nanmean(dists)
    else:
        threshold = icfg.manual_threshold

    binary_dists = dists >= threshold

    # Propagate binary label
    W_geom = meshtools.vertex_geometric_affinity_matrix(
        mesh, gamma=None, eps=1e-12, alpha=0.5, normalize=True
    )
    z_labels, _ = meshtools.labelspreading_mesh_binary(
        mesh.vertices, mesh.faces,
        (binary_dists == 1) * 1,
        W=W_geom, niters=icfg.prop_iters, return_proba=True, thresh=icfg.prop_rebinarize,
    )
    binary_dists_new = meshtools.remove_small_mesh_label_holes_binary(
        mesh.vertices, mesh.faces,
        labels=np.array(z_labels) * 1,
        vertex_labels_bool=True, physical_size=True, minsize=5,
    )

    export_colored_obj(
        mesh,
        _binary_colors(binary_dists_new),
        paths['height_binary_obj'],
    )

    # ------------------------------------------------------------------
    # 7. Initial connected components
    # ------------------------------------------------------------------
    binary_faces = spstats.mode(binary_dists_new[mesh.faces] * 1, axis=1)[0]
    binary_faces = np.squeeze(binary_faces)
    binary_mesh = meshtools.create_mesh(mesh.vertices, faces=mesh.faces[binary_faces > 0])
    protrusion_cc_labels = meshtools.connected_components_mesh(
        binary_mesh, original_face_indices=np.where(binary_faces > 0)[0]
    )
    protrusion_cc_labels = [cc for cc in protrusion_cc_labels if len(cc) >= cfg.min_size_comps_initial]

    # Build initial per-vertex CC labels (one label per connected component,
    # before any bleb/ridge splitting).  Saved as an alternative segmentation.
    initial_cc_vertex_labels = np.zeros(len(mesh.vertices), dtype=np.uint32)
    for cc_idx, cc_faces in enumerate(protrusion_cc_labels):
        verts_in_cc = np.unique(mesh.faces[cc_faces].ravel())
        initial_cc_vertex_labels[verts_in_cc] = cc_idx + 1

    cc_colors = get_vertex_colors(initial_cc_vertex_labels, palette=protrude_colors)
    export_colored_obj(mesh, cc_colors, paths['initial_cc_labels_obj'])

    # ------------------------------------------------------------------
    # 8. Per-patch refinement
    # ------------------------------------------------------------------
    H = 0.5 * (H_gauss + H_mean)
    face_areas = igl.doublearea(mesh.vertices, mesh.faces) / 2.0
    area_protrusions = np.array([np.sum(face_areas[cc]) for cc in protrusion_cc_labels])
    thresh_area = np.maximum(
        cfg.large_patch.min_max_area,
        cfg.large_patch.max_area_thresh_factor * np.nanmedian(area_protrusions),
    )
    flagged_regions = set(np.where(area_protrusions >= thresh_area)[0])

    # Diagnostic: patch area distribution with large-patch threshold line.
    _plot_patch_areas(area_protrusions, thresh_area, paths['patch_area_plot_svg'])

    global_vertex_labels = np.zeros(len(mesh.vertices), dtype=np.uint32)
    max_label = 0
    savemeshfolder = ensure_dir(paths['protrude_submesh_dir'])

    # Pass 1: split large patches into standard-size sub-patches.
    refined_ccs = []
    for jjj in range(len(protrusion_cc_labels)):
        face_labels_select = protrusion_cc_labels[jjj]
        if jjj in flagged_regions:
            sub_patches = _split_large_patch(
                jjj, mesh, face_labels_select, H, ridgeness, shape_index, face_areas, cfg,
            )
            refined_ccs.extend(sub_patches)
        else:
            refined_ccs.append(face_labels_select)

    # Pass 2: process every refined CC through the standard patch pipeline.
    for jjj, face_labels_select in enumerate(refined_ccs):
        all_boundary_loop = igl.all_boundary_loop(mesh.faces[face_labels_select])
        max_label = _process_std_patch(
            jjj, mesh, face_labels_select, all_boundary_loop,
            H, ridgeness, valleys, shape_index, dists, face_areas, Usteps, ind,
            global_vertex_labels, max_label,
            savemeshfolder, cfg,
        )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 10. Final diffusion + basal mask 
    # ------------------------------------------------------------------
    threshold_lower = skfilters.threshold_multiotsu(dists, 2)[0] 
    basal_dists = dists >= threshold_lower
    W_geom2 = meshtools.vertex_geometric_affinity_matrix(
        mesh, gamma=None, eps=1e-12, alpha=0.25, normalize=True
    )
    basal_dists, _ = meshtools.labelspreading_mesh_binary(
        mesh.vertices, mesh.faces, (basal_dists == 1) * 1,
        W=W_geom2, niters=5, return_proba=True, thresh=0.25,
    )
    binary_dists_lower = meshtools.remove_small_mesh_label_holes_binary(
        mesh.vertices, mesh.faces, labels=np.array(basal_dists) * 1,
        vertex_labels_bool=True, physical_size=True, minsize=5,
    )

    threshold_curv = skfilters.threshold_multiotsu(0.5 * (H_mean + H), 2)[-1]
    basal_curv = 0.5 * (H_mean + H) >= threshold_curv
    W_geom3 = meshtools.vertex_geometric_affinity_matrix(
        mesh, gamma=None, eps=1e-12, alpha=0.5, normalize=True
    )
    basal_curv, _ = meshtools.labelspreading_mesh_binary(
        mesh.vertices, mesh.faces, (basal_curv == 1) * 1,
        W=W_geom3, niters=5, return_proba=True, thresh=0.25,
    )
    basal_binary = np.logical_or(binary_dists_lower, np.array(basal_curv) > 0)

    export_colored_obj(mesh, _binary_colors(basal_binary), paths['basal_binary_obj'])

    # Final label spreading
    W_final = meshtools.vertex_geometric_affinity_matrix(
        mesh, gamma=None, eps=1e-12, alpha=0.5, normalize=True
    )
    protrusion_labels_final = meshtools.labelspreading_mesh(
        v=mesh.vertices, f=mesh.faces,
        x=np.where(global_vertex_labels > 0)[0],
        y=global_vertex_labels[global_vertex_labels > 0],
        W=W_final, niters=5, alpha_prop=0.5, return_proba=False, renorm=False,
    )
    global_vertex_labels = protrusion_labels_final * basal_binary
    global_vertex_labels = remove_small_label_components(mesh, global_vertex_labels, min_size=5)

    # Assign unique labels to any basal_binary region not yet covered.
    # After spreading + basal masking some connected protrusive patches may have
    # lost their label (small-component removal, spreading didn't reach them, etc.).
    # We find faces that are protrusive (basal_binary) but unlabelled, run CC
    # labelling on those faces, and append them as new instances.

    # max_label = int(np.max(global_vertex_labels))
    # remainder = np.logical_and(basal_binary==1,  # it should be a protrusion 
    #                                global_vertex_labels==0) # has not been labelled
    # if remainder.sum() == 0:
    #     pass
    # else:
    #     # perform connected comps on this residual. 
    #     binary_faces = spstats.mode(remainder[mesh.faces]*1, axis=1)[0]; binary_faces = np.squeeze(binary_faces)
    #     # create new mesh with modified connectivity, then run connected component
    #     binary_mesh = meshtools.create_mesh(mesh.vertices, 
    #                                         faces=mesh.faces[binary_faces>0])
        
    #     # now get mesh_connected_comps to label # we get 715
    #     protrusion_cc_labels = meshtools.connected_components_mesh(binary_mesh, 
    #                                                                 original_face_indices=np.arange(len(mesh.faces))[binary_faces>0])
        
    #     # map this to vertex. 
    #     remainder_mesh_labels = np.zeros(len(remainder), dtype=np.uint16)
        
    #     for cc_ii, cc in enumerate(protrusion_cc_labels):
    #         H_faces_cc = mesh.faces[cc] 
    #         H_unique_verts_cc = np.unique(H_faces_cc.ravel())
    #         remainder_mesh_labels[H_unique_verts_cc] = cc_ii+1+max_label # give the labels. 
        
    #     global_vertex_labels[remainder_mesh_labels>0] = remainder_mesh_labels[remainder_mesh_labels>0]


    # max_label = int(np.max(global_vertex_labels))
    # face_basal = np.squeeze(spstats.mode(basal_binary[mesh.faces] * 1, axis=1)[0]) > 0
    # face_labelled = np.squeeze(spstats.mode(global_vertex_labels[mesh.faces] * 1, axis=1)[0]) > 0
    # uncovered_face_mask = face_basal & ~face_labelled

    # if np.any(uncovered_face_mask):
    #     uncovered_mesh = meshtools.create_mesh(
    #         mesh.vertices, faces=mesh.faces[uncovered_face_mask]
    #     )
    #     uncovered_ccs = meshtools.connected_components_mesh(
    #         uncovered_mesh, original_face_indices=np.where(uncovered_face_mask)[0]
    #     )
    #     # Build a full edge→label lookup once so we can find the dominant
    #     # neighbour of each uncovered CC without rebuilding it per CC.
    #     all_edges = np.vstack([
    #         mesh.faces[:, [0, 1]],
    #         mesh.faces[:, [1, 2]],
    #         mesh.faces[:, [2, 0]],
    #     ])
    #     labelled_mask = global_vertex_labels > 0
    #     if labelled_mask.any():
    #         labelled_mean_height = float(np.mean(dists[labelled_mask]))
    #     else:
    #         labelled_mean_height = float(np.mean(dists)) + 1e-8

    #     for cc_faces in uncovered_ccs:
    #         verts_in_cc = np.unique(mesh.faces[cc_faces].ravel())
    #         cc_mean_height = float(np.mean(dists[verts_in_cc]))

    #         # Neck filter: skip entirely if below height threshold
    #         if cc_mean_height < cfg.uncovered_neck_height_frac * labelled_mean_height:
    #             continue

    #         # Find boundary edges: one end in this CC, other end labelled
    #         ea = all_edges[:, 0]; eb = all_edges[:, 1]
    #         in_cc_a = np.isin(ea, verts_in_cc)
    #         in_cc_b = np.isin(eb, verts_in_cc)
    #         cross_a = in_cc_a & (global_vertex_labels[eb] > 0)
    #         cross_b = in_cc_b & (global_vertex_labels[ea] > 0)
    #         boundary_verts = np.unique(np.concatenate([
    #             eb[cross_a], ea[cross_b],
    #         ])) if (cross_a.any() or cross_b.any()) else np.array([], dtype=int)

    #         if len(boundary_verts) > 0:
    #             saddle_h = float(np.mean(dists[boundary_verts]))
    #             # If boundary height is nearly as high as the CC itself, there
    #             # is no valley — the CC is a spill-over from the neighbour.
    #             # Absorb it (plurality vote on adjacent labels).
    #             # If the boundary is clearly lower, the CC sits on its own peak
    #             # and deserves a distinct label.
    #             if saddle_h > cfg.merge_saddle_factor * cc_mean_height:
    #                 neighbour_labels = np.concatenate([
    #                     global_vertex_labels[eb[cross_a]],
    #                     global_vertex_labels[ea[cross_b]],
    #                 ])
    #                 counts = np.bincount(neighbour_labels)
    #                 global_vertex_labels[verts_in_cc] = int(np.argmax(counts))
    #             else:
    #                 max_label += 1
    #                 global_vertex_labels[verts_in_cc] = max_label
    #         else:
    #             # No labelled neighbour at all — assign as new label
    #             max_label += 1
    #             global_vertex_labels[verts_in_cc] = max_label

    # # ------------------------------------------------------------------
    # # 10b. Saddle-based post-merge (after diffusion + uncovered-fill)
    # # ------------------------------------------------------------------
    # if cfg.merge_adjacent_blebs:
    #     global_vertex_labels = _greedy_merge_instances(
    #         mesh, global_vertex_labels, dists, cfg,
    #         max_merge_iters=cfg.merge_max_iters,
    #     )

    # Export final coloured mesh
    colors = get_vertex_colors(global_vertex_labels, palette=protrude_colors)
    global_vertex_labels_faces = spstats.mode(global_vertex_labels[mesh.faces] * 1, axis=-1)[0]
    if len(global_vertex_labels_faces.shape) == 2:
        global_vertex_labels_faces = global_vertex_labels_faces[:, 0]

    vertex_set_0 = []
    for cc in np.setdiff1d(np.unique(global_vertex_labels), 0):
        b_loop = igl.boundary_loop(mesh.faces[global_vertex_labels_faces == cc])
        if len(b_loop) > 0:
            vertex_set_0.append(b_loop)
    if vertex_set_0:
        colors[np.hstack(vertex_set_0)] = [0, 0, 0]

    export_colored_obj(mesh, colors, paths['final_labels_obj'])

    # ------------------------------------------------------------------
    # 10. Save statistics
    # ------------------------------------------------------------------
    spio.savemat(
        str(paths['instance_stats_mat']),
        {
            'protrusion_labels': global_vertex_labels,
            'protrusion_labels_initial_cc': initial_cc_vertex_labels,
            'external_cMCF_steps': Usteps.astype(np.float32),
            'basal_binary': basal_binary,
            'protrusion_dists': dists,
            'protrusions_H': H,
            'protrusion_H_gauss': H_gauss,
            'protrusion_H_mean': H_mean,
            'cMCF_stop_ind': ind,
        },
    )

    # ------------------------------------------------------------------
    # 11. Optional inline AP vs GT
    # ------------------------------------------------------------------
    ap_result = None
    if protrude_labels_gt is not None:
        gt_ = _relabel_sequential_vertex_labels(protrude_labels_gt)
        pred_ = _relabel_sequential_vertex_labels(global_vertex_labels)
        iou_thresholds = np.linspace(0.0, 1, 11)
        ap_result, _, _, _ = average_precision([gt_], [pred_], threshold=list(iou_thresholds))

    return SegmentResult(
        vertex_labels=global_vertex_labels,
        vertex_labels_cc=initial_cc_vertex_labels,
        mesh_V=mesh.vertices,
        mesh_F=mesh.faces,
        cMCF_steps=Usteps,
        basal_binary=basal_binary,
        dists=dists,
        H=H,
        H_gauss=H_gauss,
        H_mean=H_mean,
        ridgeness=ridgeness,
        shape_index=shape_index,
        height=dists,
        cMCF_stop_ind=ind,
        output_paths={k: v for k, v in paths.items()},
        ap=ap_result,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _project_usteps_into_binary(Usteps, mesh_binary, sdf_vol, sdf_vol_normal):
    """Clamp any vertex that left *mesh_binary* back onto its surface each step.

    Guarantees every intermediate cMCF shape lies within the binary mask: for each
    step, vertices sampling ``mesh_binary == 0`` are pushed back to the zero level
    set along the (outward) SDF normal by their signed distance.  Modifies and
    returns *Usteps* in place.
    """
    shape = np.array(mesh_binary.shape) - 1
    for t in range(Usteps.shape[-1]):
        c = np.clip(Usteps[..., t].astype(int), 0, shape)
        outside = mesh_binary[c[:, 0], c[:, 1], c[:, 2]] == 0
        if np.any(outside):
            d = sdf_vol[c[outside, 0], c[outside, 1], c[outside, 2]]
            n = sdf_vol_normal[:, c[outside, 0], c[outside, 1], c[outside, 2]].T
            Usteps[outside, :, t] = Usteps[outside, :, t] - d[:, None] * n
    return Usteps


def _stretch_pts_set(pts):
    """Elongation (aspect ratio) of a point set from its 2D (x,y) bounding extents.

    Returns ~1 for round/globular patches, >1 for elongated ones.
    The +0.1 epsilon prevents division by zero on degenerate flat patches.
    """
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return 1.0
    xx = np.max(pts[:, 0]) - np.min(pts[:, 0])
    yy = np.max(pts[:, 1]) - np.min(pts[:, 1])
    return (np.max([xx, yy]) + 0.1) / (np.min([xx, yy]) + 0.1)


def _stretch_pts_set3D(pts):
    """Elongation (aspect ratio) of a point set from its 3D bounding extents."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return 1.0
    xx = np.max(pts[:, 0]) - np.min(pts[:, 0])
    yy = np.max(pts[:, 1]) - np.min(pts[:, 1])
    zz = np.max(pts[:, 2]) - np.min(pts[:, 2])
    return (np.max([xx, yy, zz]) + 0.1) / (np.min([xx, yy, zz]) + 0.1)


# def _pca_mesh_patch_and_check_simple(submesh, face_indices, dilate_ksize=2,
#                                      planarity_frac=0.1):
#     """Whether a face-patch of *submesh* is a single, simple (planar) dome.

#     A patch is "simple" when it has exactly one boundary loop and almost all of its
#     interior vertices project inside that boundary in the PCA plane -- i.e. it is not
#     a multi-lobed or folded region needing further splitting.  Reconstruction of the
#     unported ``meshtools.pca_mesh_patch_and_check_simple`` used by the large-patch
#     bleb branch; reuses :func:`pca_rotate_mesh` and :func:`compute_planarity`.
#     """
#     faces = submesh.faces[face_indices]
#     verts_idx = np.unique(faces.ravel())
#     if len(verts_idx) < 4:
#         return True
#     remap = np.full(len(submesh.vertices), -1, dtype=np.int64)
#     remap[verts_idx] = np.arange(len(verts_idx))
#     patch = meshtools.create_mesh(submesh.vertices[verts_idx], faces=remap[faces])

#     if len(igl.all_boundary_loop(patch.faces)) != 1:
#         return False
#     boundary = igl.boundary_loop(patch.faces)
#     if len(boundary) < 3:
#         return True

#     pca_model, mean_pts = pca_rotate_mesh(patch, contour_level=0.5)
#     pca_pts = pca_model.transform(patch.vertices - mean_pts[None, :])
#     _, _, pca_img_pts = meshtools.pts_to_image2D(pca_pts[:, :2], padsize=5)
#     frac_outside, _ = compute_planarity(pca_img_pts, boundary, dilate_ksize=dilate_ksize)
#     return frac_outside <= planarity_frac

def _PCA_rotate_mesh(binary, mesh=None, mesh_contour_level=.5):
    r""" Compute principal components of a given binary through extracting the surface mesh or a used specified surface mesh 

    Parameters
    ----------
    binary : array
        input binary image 
    mesh : trimesh.Trimesh
        a user-specified surface mesh
    mesh_contour_level : scalar
        if only a binary is provided Marching cubes is used to extract a surface mesh at the isolevel given by ``mesh_contour_level``

    Returns
    -------
    pca_model : scikit-learn PCA model instance
        a fitted princial components model for the mesh. see sklearn.decomposition.PCA for attributes
    mean_pts : (3,) array
        the centroid of the surface mesh with which points were demeaned prior to PCA

    """
    import numpy as np 
    from sklearn.decomposition import PCA
    import igl 
    
    if mesh is not None:
        # we don't have a given surface, instead we need to segment. 
        v = mesh.vertices.copy()
        f = mesh.faces.copy()
    else:
        # if use_surface:
        try: 
            from skimage.measure import marching_cubes_lewiner
            v, f, _, _ = marching_cubes_lewiner(binary, level=mesh_contour_level)
        except:
            from skimage.measure import marching_cubes
            v, f, _, _ = marching_cubes(binary, level=mesh_contour_level, method='lewiner')
            
    # else:
    #     pts = np.argwhere(binary>0)
    #     pts = np.vstack(pts)
    barycenter = igl.barycenter(v,f)
    weights = igl.doublearea(v, f)
    # print(weights.shape)
    # print(barycenter.shape)
    mean_pts = np.nansum( (weights / float(np.sum(weights)))[:,None] * barycenter, axis=0)
    
    pts_ = v - mean_pts[None,:]
    pca_model = PCA(n_components=pts_.shape[-1], random_state=0, whiten=False)
    pca_model.fit(pts_)

    return pca_model, mean_pts


def _pca_mesh_patch_and_check_simple(mesh, face_labels_select, dilate_ksize=2,
                                     planarity_frac=0.1):
    """Whether a face-patch of *submesh* is a single, simple (planar) dome.

    A patch is "simple" when it has exactly one boundary loop and almost all of its
    interior vertices project inside that boundary in the PCA plane -- i.e. it is not
    a multi-lobed or folded region needing further splitting.  Reconstruction of the
    unported ``meshtools.pca_mesh_patch_and_check_simple`` used by the large-patch
    bleb branch; reuses :func:`pca_rotate_mesh` and :func:`compute_planarity`.
    """
    import igl 
    import skimage.draw as skdraw
    import skimage.morphology as skmorph
    import unwrap3D.Mesh.meshtools as meshtools
    
    # this is the new mesh to do 
    protrude_submesh, attributes = meshtools.submesh(mesh,
                                          faces_sequence=[face_labels_select],
                                          mesh_face_attributes=np.ones(len(mesh.faces))[:,None],
                                          # faces_sequence=[impute_face_select],
                                          # mesh_face_attributes=igl.average_onto_faces(mesh.faces, H[:,None]), 
                                                                # face_binary,
                                          # mesh_face_attributes=np.vstack([igl.average_onto_faces(mesh.faces, H[:,None]),
                                          #                                 face_binary*1.,
                                          #                                 face_areas,
                                          #                                 igl.average_onto_faces(mesh.faces, ridgeness[:,None]),
                                          #                                 igl.average_onto_faces(mesh.faces, valleys[:,None])]).T,
                                          repair=False,
                                          only_watertight=False,
                                          min_faces=None)
    protrude_submesh = protrude_submesh[0]

    protrude_submesh_boundary = igl.boundary_loop(protrude_submesh.faces)  ### can be used to check simple patches (those that will be bleb merged)

             
    # local PCA patch projection -> 
    pca_model, mean_pts = _PCA_rotate_mesh(binary=None, 
                                                    mesh=protrude_submesh, 
                                                    mesh_contour_level=.5)
    # cosine = np.abs(np.nansum(protrude_mean_patch_normal[None,:] * pca_model.components_, axis=-1)) # this gives alignment
    # best_match = np.argmax(cosine)
    pca_pts = pca_model.transform(protrude_submesh.vertices - mean_pts[None,:])

    # map points to image 
    pca_pts_image, pca_pts_range, pca_img_pts = meshtools.pts_to_image2D(pca_pts[:,:2], padsize = 5)
            
    """
    Determining the simpleness of region. 
    """
    canvas = np.zeros(pca_pts_image.shape, dtype=np.int32)
    rr, cc = skdraw.polygon(pca_img_pts[protrude_submesh_boundary,0].astype(np.int32),
                            pca_img_pts[protrude_submesh_boundary,1].astype(np.int32), shape=pca_pts_image.shape)
    canvas[rr,cc] = 1
    canvas = skmorph.binary_dilation(canvas,skmorph.disk(dilate_ksize))
    
    # test all points contained
    inner = np.setdiff1d(np.arange(len(pca_img_pts)), protrude_submesh_boundary)
    test_bool = canvas[(pca_img_pts[inner,0]).astype(np.int32),
                        (pca_img_pts[inner,1]).astype(np.int32)]
        
    """
    is simple or not simple 
    """
    check = np.mean(test_bool==0) <= planarity_frac

    return check


def _plot_patch_areas(area_protrusions, thresh_area, save_path):
    """Bar chart of initial CC patch areas with the large-patch threshold marked.

    Patches are sorted largest-to-smallest so the threshold cut is easy to read.
    Bars above the threshold are drawn in a distinct colour so the split is
    immediately visible.
    """
    areas = np.asarray(area_protrusions)
    order = np.argsort(areas)[::-1]
    sorted_areas = areas[order]
    is_large = sorted_areas >= thresh_area

    fig, ax = plt.subplots(figsize=(max(6, len(areas) * 0.35 + 1), 4))
    x = np.arange(len(sorted_areas))
    colors_bar = np.where(is_large, '#d62728', '#1f77b4')   # red = large, blue = std
    ax.bar(x, sorted_areas, color=colors_bar, width=0.8, zorder=2)
    ax.axhline(thresh_area, color='black', linewidth=1.4, linestyle='--', zorder=3,
               label=f'large-patch threshold = {thresh_area:.1f}')

    ax.set_xlabel('Patch index (sorted by area)', fontsize=11)
    ax.set_ylabel('Surface area (mesh units²)', fontsize=11)
    ax.set_title(
        f'Initial CC patch areas  |  {int(is_large.sum())} large (red),'
        f' {int((~is_large).sum())} std (blue)',
        fontsize=11,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_xlim(-0.5, len(sorted_areas) - 0.5)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linewidth=0.4, alpha=0.5, zorder=1)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _check_hcc_valleys(mesh, labels, vertex_H, uniq_regions, valley_factor):
    """Return True if all adjacent H-CC pairs have a genuine valley between them.

    For each pair of H-CC regions that share a face edge, the mean |H| on the
    shared boundary is compared to the lower of the two CCs' mean |H|.  If the
    boundary H exceeds ``valley_factor * instance_H_min`` for any pair, the two
    CCs are topographically connected (one dome with local dips) and the split
    should be vetoed → returns False.

    Returns True only when every adjacent pair is separated by a genuine
    low-curvature valley, i.e., all boundaries have H well below both CCs.
    """
    if valley_factor <= 0.0:
        return True  # check disabled

    # Build set of vertex pairs that straddle two different CC labels.
    # We look at each edge of each face and check if its two vertices belong to
    # different (non-zero) CC labels.
    labels = np.asarray(labels)
    edges = np.vstack([
        mesh.faces[:, [0, 1]],
        mesh.faces[:, [1, 2]],
        mesh.faces[:, [2, 0]],
    ])
    la = labels[edges[:, 0]]
    lb = labels[edges[:, 1]]
    cross = (la > 0) & (lb > 0) & (la != lb)
    if not np.any(cross):
        return True  # no adjacent CCs — nothing to check

    cross_edges = edges[cross]
    la_cross = la[cross]
    lb_cross = lb[cross]

    # Unique adjacent pairs (canonical order: smaller label first)
    pair_keys = np.sort(np.stack([la_cross, lb_cross], axis=1), axis=1)
    pair_keys = np.unique(pair_keys, axis=0)

    abs_H = np.abs(vertex_H)
    H_per_region = {rr: float(np.mean(abs_H[labels == rr])) for rr in uniq_regions}

    for a, b in pair_keys:
        # Boundary vertices: any vertex on an edge straddling (a, b)
        mask_ab = ((la_cross == a) & (lb_cross == b)) | ((la_cross == b) & (lb_cross == a))
        boundary_verts = np.unique(cross_edges[mask_ab])
        saddle_H = float(np.mean(abs_H[boundary_verts]))
        instance_H_min = min(H_per_region.get(a, 0.0), H_per_region.get(b, 0.0))
        if saddle_H > valley_factor * instance_H_min:
            return False  # connected through high-H — veto the split

    return True


def _instance_signal(mask, H_mean, ridgeness, height, cfg, h99, H99, r99,
                      w_height=0.4):
    """Per-instance biological signal score (same formula as optimize.score_segmentation).

    Parameters
    ----------
    mask : boolean array, length = n_vertices
    h99, H99, r99 : float — 99th-percentile normalisation constants for height,
        |H_mean|, and ridgeness over all protrusive vertices.
    w_height : float — weight on height vs shape signal.
    """
    eps = 1e-8
    thr = cfg.std_patch.curv_ridge_ratio_threshold
    w = w_height

    r_mean = float(np.mean(ridgeness[mask]))
    H_abs_mean = float(np.mean(np.abs(H_mean[mask])))
    ratio = r_mean / (r_mean + H_abs_mean + eps)

    S_height = float(np.mean(height[mask])) / (h99 + eps)
    S_shape = (H_abs_mean / (H99 + eps)) if ratio < thr else (r_mean / (r99 + eps))
    return w * S_height + (1.0 - w) * S_shape


def _greedy_merge_instances(mesh, global_vertex_labels, height, cfg,
                             max_merge_iters=100):
    """Merge adjacent label pairs that are topographically on the same dome.

    Uses the HEIGHT field (cMCF displacement distance) rather than curvature:
    if the mean height on the shared boundary (saddle) between two instances
    exceeds ``cfg.merge_saddle_factor * min(height_a, height_b)``, the saddle
    is elevated — the two fragments sit on the same protrusion dome — and they
    are merged.  When the saddle is in a genuine inter-protrusion valley the
    mean boundary height is low and the criterion is not met.

    Height is preferred over curvature because |H| varies strongly across a
    single bleb (tip >> flank >> rim), making curvature-based comparisons
    unreliable.  Height is smoother and encodes the global topology of each
    protrusion rather than local surface bending.
    """
    factor = cfg.merge_saddle_factor
    if factor <= 0.0:
        return global_vertex_labels

    labels_arr = np.array(global_vertex_labels, dtype=np.int32)

    edges = np.vstack([
        mesh.faces[:, [0, 1]],
        mesh.faces[:, [1, 2]],
        mesh.faces[:, [2, 0]],
    ])

    for _ in range(max_merge_iters):
        la = labels_arr[edges[:, 0]]
        lb = labels_arr[edges[:, 1]]
        cross = (la > 0) & (lb > 0) & (la != lb)
        if not np.any(cross):
            break

        cross_edges = edges[cross]
        la_cross = la[cross]
        lb_cross = lb[cross]
        pair_keys = np.sort(np.stack([la_cross, lb_cross], axis=1), axis=1)
        pair_keys = np.unique(pair_keys, axis=0)

        # Per-instance mean height
        uniq_all = np.unique(np.concatenate([pair_keys[:, 0], pair_keys[:, 1]]))
        h_per = {int(u): float(np.mean(height[labels_arr == u])) for u in uniq_all}

        # Union-find: merge all qualifying pairs in one pass
        parent = {int(u): int(u) for u in uniq_all}

        def _find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        merged_any = False
        for a, b in pair_keys:
            a, b = int(a), int(b)
            ra, rb = _find(a), _find(b)
            if ra == rb:
                continue
            mask_ab = ((la_cross == a) & (lb_cross == b)) | \
                      ((la_cross == b) & (lb_cross == a))
            boundary_verts = np.unique(cross_edges[mask_ab])
            saddle_h = float(np.mean(height[boundary_verts]))
            h_a = h_per.get(ra, h_per[a])
            h_b = h_per.get(rb, h_per[b])
            if saddle_h > factor * min(h_a, h_b):
                parent[rb] = ra
                merged_any = True

        if not merged_any:
            break

        for old_lbl in list(parent.keys()):
            root = _find(old_lbl)
            if root != old_lbl:
                labels_arr[labels_arr == old_lbl] = root

    return labels_arr


def _binary_colors(binary):
    """Grey/blue colour array from a boolean/int array."""
    from matplotlib import cm as _cm
    import unwrap3D.Visualisation.colors as _vc
    binary = np.asarray(binary)
    colors = _vc.get_colors(binary * 1, colormap=_cm.coolwarm)[:, :3]
    colors[binary == 0] = [0.75, 0.75, 0.75]
    return colors


def _map_labels_to_global(unique_verts_coords, protrude_submesh, protrude_submesh_labels,
                           global_vertex_labels, unique_verts_cc, max_label):
    """Transfer sub-mesh labels back to the global vertex array via kNN."""
    protrude_submesh_labels = np.asarray(protrude_submesh_labels)
    mask = protrude_submesh_labels > 0
    protrude_submesh_labels[mask] = protrude_submesh_labels[mask] + max_label
    max_label = int(np.max(protrude_submesh_labels))
    _, corrs = pcu.k_nearest_neighbors(unique_verts_coords, protrude_submesh.vertices, k=1)
    global_vertex_labels[unique_verts_cc] = protrude_submesh_labels[np.squeeze(corrs.ravel())]
    return max_label


def _split_large_patch(jjj, mesh, face_labels_select, H, ridgeness, shape_index, face_areas, cfg):
    """Split an over-sized CC into standard-size sub-patches.

    Returns a list of face-index arrays (indices into ``mesh.faces``), one per
    sub-patch.  Bleb-classified patches are subdivided by adaptive curvature
    segmentation; ridge-like or ambiguous patches are returned as a single
    element list so ``_process_std_patch`` handles the splitting.
    """
    lcfg = cfg.large_patch
    face_binary = np.zeros(len(mesh.faces), dtype=bool)
    face_binary[face_labels_select] = True

    invert_mesh = meshtools.create_mesh(mesh.vertices, faces=mesh.faces[~face_binary])
    comps = meshtools.connected_components_mesh(
        invert_mesh, original_face_indices=np.where(~face_binary)[0]
    )
    largest = comps[np.argmax([len(cc) for cc in comps])]
    impute_face_select = np.setdiff1d(np.arange(len(mesh.faces)), largest)

    protrude_submesh, attributes = meshtools.submesh(
        mesh,
        faces_sequence=[impute_face_select],
        mesh_face_attributes=np.vstack([
            igl.average_onto_faces(mesh.faces, H[:, None]),
            face_areas,
            igl.average_onto_faces(mesh.faces, ridgeness[:, None]),
            igl.average_onto_faces(mesh.faces, shape_index[:, None]),
        ]).T,
        repair=False, only_watertight=False, min_faces=None,
    )
    protrude_submesh = protrude_submesh[0]

    vertex_H_disk = igl.average_onto_vertices(
        protrude_submesh.vertices, protrude_submesh.faces,
        np.vstack([attributes[0][:, 0]] * 3).T,
    )[:, 0]
    vertex_ridgeness_disk = igl.average_onto_vertices(
        protrude_submesh.vertices, protrude_submesh.faces,
        np.vstack([attributes[0][:, 2]] * 3).T,
    )[:, 0]
    vertex_si_disk = igl.average_onto_vertices(
        protrude_submesh.vertices, protrude_submesh.faces,
        np.vstack([attributes[0][:, 3]] * 3).T,
    )[:, 0]

    mean_ridgeness_patch = float(np.nanmean(vertex_ridgeness_disk))
    mean_abs_H_patch = float(np.nanmean(np.abs(vertex_H_disk)))
    curv_ridge_ratio = mean_ridgeness_patch / (mean_ridgeness_patch + mean_abs_H_patch + 1e-8)
    mean_si_patch = float(np.nanmean(vertex_si_disk))

    _thr = lcfg.curv_ridge_ratio_threshold
    _off = lcfg.curv_ridge_ratio_offset

    # Optionally replace curv_ridge_ratio with the scale-invariant Shape Index.
    # SI near +1 = dome/bleb, near +0.5 = cylinder/ridge.  We remap SI to [0,1]
    # by (1 - SI) / 2 so that low values → bleb (matches curv_ridge_ratio=0 for bleb).
    if lcfg.use_shape_index_classifier:
        curv_ridge_ratio = (1.0 - mean_si_patch) / 2.0

    if curv_ridge_ratio < _thr - _off:
        # Bleb-like: adaptive H-seg → CCs → return as separate sub-patches
        vertex_H_work = vertex_H_disk.copy()
        if lcfg.apply_power_H_correct:
            vertex_H_work = (vertex_H_work - vertex_H_work.min()) ** lcfg.power_H_correct

        if lcfg.H_segment_use_local_adaptive:
            smooth_H = mesh_smooth_scalar(
                protrude_submesh, vertex_H_work, delta=0.5,
                n_iters=lcfg.H_local_adaptive_smooth_iters,
            )
            vertex_H_work = vertex_H_work - smooth_H

        binary_H, _ = binary_threshold(
            vertex_H_work,
            method=lcfg.H_segment_method,
            n_otsu_levels=lcfg.H_segment_otsu_n_levels,
            level=lcfg.H_segment_otsu_level,
        )

        if lcfg.H_segment_erode_steps > 0:
            W_disk = meshtools.vertex_geometric_affinity_matrix(
                protrude_submesh, gamma=None, eps=1e-12, alpha=1.0, normalize=True
            )
            for _ in range(lcfg.H_segment_erode_steps):
                z_labels, _ = meshtools.labelspreading_mesh_binary(
                    protrude_submesh.vertices, protrude_submesh.faces,
                    (binary_H == 0) * 1, W=W_disk, niters=1,
                    return_proba=True, thresh=cfg.initial_height.prop_rebinarize,
                )
                binary_H = ~(np.asarray(z_labels) > 0)

        H_binary_faces = np.squeeze(
            spstats.mode(binary_H[protrude_submesh.faces] * 1, axis=1)[0]
        )
        H_binary_mesh = meshtools.create_mesh(
            protrude_submesh.vertices, faces=protrude_submesh.faces[H_binary_faces > 0]
        )
        H_cc_labels = meshtools.connected_components_mesh(
            H_binary_mesh, original_face_indices=np.where(H_binary_faces > 0)[0]
        )
        H_cc_labels = [cc for cc in H_cc_labels if len(cc) >= cfg.min_size_comps_protrude_patch]

        # Map each CC back to global face indices and return as sub-patches.
        # protrude_submesh was built from impute_face_select in face order, so
        # submesh face i == impute_face_select[i] in the global mesh.
        result = [impute_face_select[cc] for cc in H_cc_labels]
        if not result:
            result = [face_labels_select]  # fallback: keep whole patch
        return result

    else:
        # Ridge-like or ambiguous: pass the whole patch to _process_std_patch
        # which will apply its own bleb/ridge classification and splitting.
        return [face_labels_select]


def _merge_ridge_ccs(mesh, cc_face_lists, vertex_height, factor, max_iters=20):
    """Merge adjacent ridge CC face-lists where boundary height > factor * min(h_a, h_b)."""
    if len(cc_face_lists) <= 1:
        return cc_face_lists
    edges = np.vstack([
        mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]],
    ])
    for _ in range(max_iters):
        vert_cc = np.zeros(len(mesh.vertices), dtype=np.int32)
        for idx, cc in enumerate(cc_face_lists):
            vert_cc[np.unique(mesh.faces[cc].ravel())] = idx + 1
        h_per = {
            idx + 1: float(np.mean(vertex_height[vert_cc == idx + 1]))
            for idx in range(len(cc_face_lists))
        }
        la = vert_cc[edges[:, 0]]; lb = vert_cc[edges[:, 1]]
        cross = (la > 0) & (lb > 0) & (la != lb)
        if not cross.any():
            break
        cross_e = edges[cross]; la_c = la[cross]; lb_c = lb[cross]
        pairs = np.unique(np.sort(np.stack([la_c, lb_c], axis=1), axis=1), axis=0)
        parent = {idx + 1: idx + 1 for idx in range(len(cc_face_lists))}

        def _find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        merged = False
        for a, b in pairs:
            ra, rb = _find(int(a)), _find(int(b))
            if ra == rb:
                continue
            mask = ((la_c == a) & (lb_c == b)) | ((la_c == b) & (lb_c == a))
            saddle_h = float(np.mean(vertex_height[np.unique(cross_e[mask])]))
            if saddle_h > factor * min(h_per.get(ra, h_per[a]), h_per.get(rb, h_per[b])):
                parent[rb] = ra
                merged = True
        if not merged:
            break
        groups: dict = {}
        for idx in range(len(cc_face_lists)):
            root = _find(idx + 1)
            groups.setdefault(root, []).append(idx)
        cc_face_lists = [
            np.concatenate([cc_face_lists[i] for i in idxs])
            for idxs in groups.values()
        ]
    return cc_face_lists


def _process_std_patch(jjj, mesh, face_labels_select, all_boundary_loop,
                        H, ridgeness, valleys, shape_index, height, face_areas, Usteps, ind,
                        global_vertex_labels, max_label, savemeshfolder, cfg):
    """Handle a standard-sized patch."""
    scfg = cfg.std_patch

    if len(all_boundary_loop) == 1:
        faces_cc = mesh.faces[face_labels_select]
        unique_verts_cc = np.unique(faces_cc.ravel())
        unique_verts_coords = mesh.vertices[unique_verts_cc].copy()
        protrude_submesh, attributes = meshtools.submesh(
            mesh,
            faces_sequence=[face_labels_select],
            mesh_face_attributes=np.vstack([
                igl.average_onto_faces(mesh.faces, H[:, None]),
                face_areas,
                igl.average_onto_faces(mesh.faces, ridgeness[:, None]),
                igl.average_onto_faces(mesh.faces, valleys[:, None]),
                igl.average_onto_faces(mesh.faces, shape_index[:, None]),
                igl.average_onto_faces(mesh.faces, height[:, None]),
            ]).T,
            repair=False, only_watertight=False, min_faces=None,
        )
        protrude_submesh = protrude_submesh[0]
        attr = attributes[0]
        vertex_H_disk = igl.average_onto_vertices(
            protrude_submesh.vertices, protrude_submesh.faces, np.vstack([attr[:, 0]] * 3).T
        )[:, 0]
        vertex_areas_disk = igl.average_onto_vertices(
            protrude_submesh.vertices, protrude_submesh.faces, np.vstack([attr[:, 1]] * 3).T
        )[:, 0]
        vertex_ridgeness_disk = igl.average_onto_vertices(
            protrude_submesh.vertices, protrude_submesh.faces, np.vstack([attr[:, 2]] * 3).T
        )[:, 0]
        vertex_valleys_disk = igl.average_onto_vertices(
            protrude_submesh.vertices, protrude_submesh.faces, np.vstack([attr[:, 3]] * 3).T
        )[:, 0]
        vertex_si_disk = igl.average_onto_vertices(
            protrude_submesh.vertices, protrude_submesh.faces, np.vstack([attr[:, 4]] * 3).T
        )[:, 0]
        vertex_height_disk = igl.average_onto_vertices(
            protrude_submesh.vertices, protrude_submesh.faces, np.vstack([attr[:, 5]] * 3).T
        )[:, 0]
    else:
        face_binary = np.zeros(len(mesh.faces), dtype=bool)
        face_binary[face_labels_select] = True
        invert_mesh = meshtools.create_mesh(mesh.vertices, faces=mesh.faces[~face_binary])
        comps = meshtools.connected_components_mesh(
            invert_mesh, original_face_indices=np.where(~face_binary)[0]
        )
        largest = comps[np.argmax([len(cc) for cc in comps])]
        impute_face_select = np.setdiff1d(np.arange(len(mesh.faces)), largest)

        faces_cc = mesh.faces[impute_face_select]
        unique_verts_cc = np.unique(faces_cc.ravel())
        unique_verts_coords = mesh.vertices[unique_verts_cc].copy()

        protrude_submesh, attributes = meshtools.submesh(
            mesh,
            faces_sequence=[impute_face_select],
            mesh_face_attributes=np.vstack([
                igl.average_onto_faces(mesh.faces, H[:, None]),
                face_binary * 1.0,
                face_areas,
                igl.average_onto_faces(mesh.faces, ridgeness[:, None]),
                igl.average_onto_faces(mesh.faces, valleys[:, None]),
                igl.average_onto_faces(mesh.faces, shape_index[:, None]),
                igl.average_onto_faces(mesh.faces, height[:, None]),
            ]).T,
            repair=False, only_watertight=False, min_faces=None,
        )
        protrude_submesh = protrude_submesh[0]
        attr = attributes[0]
        vertex_H_disk = igl.average_onto_vertices(
            protrude_submesh.vertices, protrude_submesh.faces, np.vstack([attr[:, 0]] * 3).T
        )[:, 0]
        vertex_areas_disk = igl.average_onto_vertices(
            protrude_submesh.vertices, protrude_submesh.faces, np.vstack([attr[:, 1]] * 3).T
        )[:, 0]
        vertex_ridgeness_disk = igl.average_onto_vertices(
            protrude_submesh.vertices, protrude_submesh.faces, np.vstack([attr[:, 3]] * 3).T
        )[:, 0]
        vertex_valleys_disk = igl.average_onto_vertices(
            protrude_submesh.vertices, protrude_submesh.faces, np.vstack([attr[:, 4]] * 3).T
        )[:, 0]
        vertex_si_disk = igl.average_onto_vertices(
            protrude_submesh.vertices, protrude_submesh.faces, np.vstack([attr[:, 5]] * 3).T
        )[:, 0]
        vertex_height_disk = igl.average_onto_vertices(
            protrude_submesh.vertices, protrude_submesh.faces, np.vstack([attr[:, 6]] * 3).T
        )[:, 0]

    mean_ridgeness_patch = float(np.nanmean(vertex_ridgeness_disk))
    mean_abs_H_patch = float(np.nanmean(np.abs(vertex_H_disk)))
    curv_ridge_ratio = mean_ridgeness_patch / (mean_ridgeness_patch + mean_abs_H_patch + 1e-8)
    mean_si_patch = float(np.nanmean(vertex_si_disk))

    if scfg.use_shape_index_classifier:
        curv_ridge_ratio = (1.0 - mean_si_patch) / 2.0

    protrude_submesh.export(str(savemeshfolder / f'protrude_{jjj:06d}.obj'))

    # PCA projection
    pca_model, mean_pts = meshtools.PCA_rotate_mesh(
        binary=None, mesh=protrude_submesh, mesh_contour_level=0.5
    )
    pca_pts = pca_model.transform(protrude_submesh.vertices - mean_pts[None, :])
    pca_pts_image, _, pca_img_pts = meshtools.pts_to_image2D(pca_pts[:, :2], padsize=5)

    protrude_submesh_boundary = igl.boundary_loop(protrude_submesh.faces)

    # H-based segmentation
    vertex_H_disk_work = vertex_H_disk.copy()
    if scfg.apply_power_H_correct:
        vertex_H_disk_work = (vertex_H_disk_work - vertex_H_disk_work.min()) ** scfg.power_H_correct

    binary_height_prop_rebinarize = cfg.initial_height.prop_rebinarize

    if scfg.H_segment_use_local_adaptive:
        smooth_H = mesh_smooth_scalar(
            protrude_submesh, vertex_H_disk_work, delta=0.5,
            n_iters=scfg.H_local_adaptive_smooth_iters,
        )
        diff_H = vertex_H_disk_work - smooth_H
        binary_H_protrude, _ = binary_threshold(
            diff_H, method=scfg.H_segment_method,
            n_otsu_levels=scfg.H_segment_otsu_n_levels,
            level=scfg.H_segment_otsu_level,
        )
    else:
        binary_H_protrude, _ = binary_threshold(
            vertex_H_disk_work, method=scfg.H_segment_method,
            n_otsu_levels=scfg.H_segment_otsu_n_levels,
            level=scfg.H_segment_otsu_level,
        )

    if scfg.H_segment_erode_steps > 0:
        W_disk = meshtools.vertex_geometric_affinity_matrix(
            protrude_submesh, gamma=None, eps=1e-12, alpha=1.0, normalize=True
        )
        for _ in range(scfg.H_segment_erode_steps):
            z_labels, _ = meshtools.labelspreading_mesh_binary(
                protrude_submesh.vertices, protrude_submesh.faces,
                (binary_H_protrude == 0) * 1, W=W_disk, niters=1,
                return_proba=True, thresh=binary_height_prop_rebinarize,
            )
            binary_H_protrude = ~(np.asarray(z_labels) > 0)

    H_binary_faces = np.squeeze(
        spstats.mode(binary_H_protrude[protrude_submesh.faces] * 1, axis=1)[0]
    )
    H_binary_mesh = meshtools.create_mesh(
        protrude_submesh.vertices, faces=protrude_submesh.faces[H_binary_faces > 0]
    )
    H_cc_labels = meshtools.connected_components_mesh(
        H_binary_mesh, original_face_indices=np.where(H_binary_faces > 0)[0]
    )
    H_cc_labels = [cc for cc in H_cc_labels if len(cc) >= cfg.min_size_comps_protrude_patch]

    protrude_submesh_labels = np.zeros(len(protrude_submesh.vertices), dtype=np.uint16)
    for ii, cc in enumerate(H_cc_labels):
        verts = np.unique(protrude_submesh.faces[cc].ravel())
        protrude_submesh_labels[verts] = ii + 1

    protrude_submesh_labels_master = protrude_submesh_labels.copy()
    uniq_regions = np.setdiff1d(np.unique(protrude_submesh_labels), 0)
    if len(uniq_regions) == 0:
        protrude_submesh_labels = np.ones_like(protrude_submesh_labels)
        uniq_regions = np.array([1])
        protrude_submesh_labels_master = protrude_submesh_labels.copy()

    # Watershed centroid detection in 2D projection
    import scipy.ndimage as ndimage
    dist_tform = ndimage.distance_transform_edt(pca_pts_image)
    dist_tform = ndimage.gaussian_filter(dist_tform, sigma=1.0)
    edt_gradients = np.array(np.gradient(dist_tform))
    edt_gradients /= (np.linalg.norm(edt_gradients, axis=0) + 1e-20)
    wseg = segmentation.gradient_watershed2D_binary(
        pca_pts_image, gradient_img=edt_gradients.transpose(1, 2, 0),
        interp=True, delta=1.0, n_iter=200,
    )[0]
    regprops = skmeasure.regionprops(wseg)
    coords = np.vstack([re.centroid for re in regprops])
    coord_labels = np.zeros_like(wseg)
    coord_labels[coords[:, 0].astype(int), coords[:, 1].astype(int)] = 1
    coord_labels = skmorph.binary_dilation(coord_labels, skmorph.disk(3))
    coord_labels = coord_labels * (dist_tform > 0)
    coord_labels = skmeasure.label(coord_labels * 1)

    # Planarity check
    from skimage.draw import polygon as draw_polygon
    canvas = np.zeros(pca_pts_image.shape, dtype=bool)
    bpts = pca_img_pts[protrude_submesh_boundary]
    for off in [(0, 0), (0.5, 0.5)]:
        rr, cc = draw_polygon(
            (bpts[:, 0] + off[0]).astype(int),
            (bpts[:, 1] + off[1]).astype(int),
            shape=pca_pts_image.shape,
        )
        canvas[rr, cc] = True
    canvas = skmorph.binary_dilation(canvas, skmorph.disk(scfg.planarity_check_dilate_binary))
    inner = np.setdiff1d(np.arange(len(pca_img_pts)), protrude_submesh_boundary)
    if len(inner) > 0:
        test_bool = canvas[pca_img_pts[inner, 0].astype(int), pca_img_pts[inner, 1].astype(int)]
        check = np.sum(test_bool == 0) / float(len(test_bool)) <= scfg.planarity_check_frac
    else:
        check = True

    if len(uniq_regions) == 1:
        protrude_submesh_labels = np.ones(len(protrude_submesh.vertices), dtype=np.uint8)
        max_label = _map_labels_to_global(
            unique_verts_coords, protrude_submesh, protrude_submesh_labels,
            global_vertex_labels, unique_verts_cc, max_label,
        )
        return max_label

    _thr = scfg.curv_ridge_ratio_threshold
    _off = scfg.curv_ridge_ratio_offset

    if check and curv_ridge_ratio < _thr - _off:
        # Multi-bleb check
        areas_regions = [np.sum(vertex_areas_disk[protrude_submesh_labels == rr]) for rr in uniq_regions]
        areas_occ = np.sum(areas_regions) / float(np.sum(vertex_areas_disk))
        aspect_regions = [
            _stretch_pts_set(protrude_submesh.vertices[protrude_submesh_labels == rr])
            for rr in uniq_regions
        ]
        mean_aspect_regions = float(np.mean(aspect_regions))
        watershed_points = [
            np.setdiff1d(np.unique(coord_labels[
                (pca_img_pts[protrude_submesh_labels == rr, 0] - 1).astype(int),
                (pca_img_pts[protrude_submesh_labels == rr, 1] - 1).astype(int),
            ]), 0)
            for rr in uniq_regions
        ]
        watershed_points_nunique = len(np.unique(np.hstack(watershed_points)))
        n_regions_with_pts = int(np.sum([len(c) > 0 for c in watershed_points]))
        # Fraction of H-CCs that contain at least one watershed centroid.
        # Using only len(uniq_regions) in the denominator avoids inflation from
        # spurious background centroids produced by the 2D EDT watershed.
        recovered_frac = n_regions_with_pts / float(len(uniq_regions))

        pca_boundary_pts = pca_pts[protrude_submesh_boundary, :2].copy()
        pca_boundary_pts_smooth = meshtools._smooth_boundary_pts(pca_boundary_pts, winsize=5)

        def _reorient(pts):
            center = pts.mean(0)
            angs = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
            if np.sign(np.nanmedian(np.diff(angs))) < 0:
                return pts[::-1]
            return pts

        pca_boundary_pts_smooth = _reorient(pca_boundary_pts_smooth)
        pca_boundary_curvature = meshtools._curvature_splines(
            pca_boundary_pts_smooth[:, 0], pca_boundary_pts_smooth[:, 1], k=4, error=0.5
        )[-1]

        if scfg.multibleb_force_split:
            to_split = True
        else:
            to_split = (
                recovered_frac > scfg.multibleb_min_recovered_frac
                and areas_occ > scfg.multibleb_min_occ_area_fraction
                and mean_aspect_regions < scfg.multibleb_max_mean_aspect_ratio
            )
            if to_split and scfg.multibleb_check_neck_neg_curvature:
                to_split = np.min(pca_boundary_curvature) < scfg.multibleb_neck_neg_curvature_thresh
            if to_split and scfg.multibleb_valley_factor > 0.0:
                to_split = _check_hcc_valleys(
                    protrude_submesh, protrude_submesh_labels_master,
                    vertex_H_disk, uniq_regions,
                    valley_factor=scfg.multibleb_valley_factor,
                )

        if to_split:
            protrude_submesh_labels = protrude_submesh_labels_master.copy()
            W = meshtools.vertex_geometric_affinity_matrix(
                protrude_submesh, gamma=None, eps=1e-12, alpha=0.25, normalize=True
            )
            protrude_submesh_labels = meshtools.labelspreading_mesh(
                v=protrude_submesh.vertices, f=protrude_submesh.faces,
                x=np.where(protrude_submesh_labels > 0)[0],
                y=protrude_submesh_labels[protrude_submesh_labels > 0],
                W=W, niters=5, alpha_prop=0.9, return_proba=False, renorm=False,
            )
        else:
            protrude_submesh_labels = np.ones(len(protrude_submesh.vertices), dtype=np.uint8)
    elif not check or curv_ridge_ratio > _thr + _off:
        # Ridge segmentation (non-planar patch, or strongly ridge-like ratio)
        if scfg.ridge_no_split:
            protrude_submesh_labels = np.ones(len(protrude_submesh.vertices), dtype=np.uint8)
        else:
            vertex_ridge_work = vertex_ridgeness_disk.copy()
            if scfg.apply_power_ridge_correct:
                vertex_ridge_work = (vertex_ridge_work - vertex_ridge_work.min()) ** scfg.power_ridge_correct

            if scfg.ridge_use_local_adaptive:
                smooth_ridge = mesh_smooth_scalar(
                    protrude_submesh, vertex_ridge_work, delta=0.5,
                    n_iters=scfg.ridge_local_adaptive_smooth_iters,
                )
                diff_ridge = vertex_ridge_work - smooth_ridge
                binary_ridge, _ = binary_threshold(
                    diff_ridge, method=scfg.ridge_segment_method,
                    n_otsu_levels=scfg.ridge_otsu_n_levels, level=scfg.ridge_otsu_level,
                )
            else:
                binary_ridge, _ = binary_threshold(
                    vertex_ridge_work, method=scfg.ridge_segment_method,
                    n_otsu_levels=scfg.ridge_otsu_n_levels, level=scfg.ridge_otsu_level,
                )

            if scfg.ridge_segment_erode_steps > 0:
                W_disk = meshtools.vertex_geometric_affinity_matrix(
                    protrude_submesh, gamma=None, eps=1e-12, alpha=1.0, normalize=True
                )
                for _ in range(scfg.ridge_segment_erode_steps):
                    z_labels, _ = meshtools.labelspreading_mesh_binary(
                        protrude_submesh.vertices, protrude_submesh.faces,
                        (binary_ridge == 0) * 1, W=W_disk, niters=1,
                        return_proba=True, thresh=binary_height_prop_rebinarize,
                    )
                    binary_ridge = ~(np.asarray(z_labels) > 0)

            ridge_binary_faces = np.squeeze(
                spstats.mode(binary_ridge[protrude_submesh.faces] * 1, axis=1)[0]
            )
            ridge_binary_mesh = meshtools.create_mesh(
                protrude_submesh.vertices, faces=protrude_submesh.faces[ridge_binary_faces > 0]
            )
            ridge_cc_labels = meshtools.connected_components_mesh(
                ridge_binary_mesh, original_face_indices=np.where(ridge_binary_faces > 0)[0]
            )
            ridge_cc_labels = [cc for cc in ridge_cc_labels if len(cc) >= cfg.min_size_comps_protrude_patch]

            # Multi-ridge guard: drop CCs whose mean Shape Index exceeds the
            # bleb threshold — they are dome-like patches misclassified as ridges.
            # They stay unlabelled (label 0) and get absorbed by label-spreading.
            si_thr = scfg.ridge_reclass_si_threshold
            if si_thr <= 1.0:
                ridge_cc_labels = [
                    cc for cc in ridge_cc_labels
                    if float(np.mean(vertex_si_disk[np.unique(protrude_submesh.faces[cc].ravel())])) <= si_thr
                ]

            # Ridge-fragment merge: collapse adjacent ridge CCs separated by a
            # height valley that is shallower than the factor × lower CC height.
            # Works regardless of whether use_shape_index_classifier is on.
            if scfg.ridge_merge_saddle_factor > 0.0 and len(ridge_cc_labels) > 1:
                ridge_cc_labels = _merge_ridge_ccs(
                    protrude_submesh, ridge_cc_labels,
                    vertex_height_disk, scfg.ridge_merge_saddle_factor,
                )

            ridge_submesh_labels = np.zeros(len(protrude_submesh.vertices), dtype=np.uint16)
            for ii, cc in enumerate(ridge_cc_labels):
                verts = np.unique(protrude_submesh.faces[cc].ravel())
                ridge_submesh_labels[verts] = ii + 1

            if len(ridge_cc_labels) > 0:
                protrude_submesh_labels = ridge_submesh_labels.copy()
                W = meshtools.vertex_geometric_affinity_matrix(
                    protrude_submesh, gamma=None, eps=1e-12, alpha=0.25, normalize=True
                )
                protrude_submesh_labels = meshtools.labelspreading_mesh(
                    v=protrude_submesh.vertices, f=protrude_submesh.faces,
                    x=np.where(protrude_submesh_labels > 0)[0],
                    y=protrude_submesh_labels[protrude_submesh_labels > 0],
                    W=W, niters=5, alpha_prop=0.9, return_proba=False, renorm=False,
                )

    else:
        # Ambiguous zone: ratio in [thr-off, thr+off] with check=True → single label
        protrude_submesh_labels = np.ones(len(protrude_submesh.vertices), dtype=np.uint8)

    max_label = _map_labels_to_global(
        unique_verts_coords, protrude_submesh, protrude_submesh_labels,
        global_vertex_labels, unique_verts_cc, max_label,
    )
    return max_label
