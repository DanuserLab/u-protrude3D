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


@dataclass
class SegmentResult:
    """Return value of :func:`segment_protrusions`."""
    vertex_labels: np.ndarray
    mesh_V: np.ndarray
    mesh_F: np.ndarray
    cMCF_steps: np.ndarray
    basal_binary: np.ndarray
    dists: np.ndarray
    H: np.ndarray
    H_gauss: np.ndarray
    H_mean: np.ndarray
    cMCF_stop_ind: int
    output_paths: dict
    ap: Optional[np.ndarray] = None


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

    modified_sdf = meshtools._normalize99(H_normal)[..., None] * sdf_vol_normal.transpose(1, 2, 3, 0)

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

    # ------------------------------------------------------------------
    # 6. Initial height binarization
    # ------------------------------------------------------------------
    icfg = cfg.initial_height
    if icfg.use_auto:
        if icfg.use_mean:
            threshold = np.nanmean(dists)
        elif icfg.use_otsu:
            lvl = 0 if icfg.use_lower_threshold else -1
            threshold = skfilters.threshold_multiotsu(dists, 3)[lvl]
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

    global_vertex_labels = np.zeros(len(mesh.vertices), dtype=np.uint32)
    max_label = 0
    savemeshfolder = ensure_dir(paths['protrude_submesh_dir'])

    for jjj in range(len(protrusion_cc_labels)):
        face_labels_select = protrusion_cc_labels[jjj]
        all_boundary_loop = igl.all_boundary_loop(mesh.faces[face_labels_select])

        if jjj in flagged_regions:
            max_label = _process_large_patch(
                jjj, mesh, face_labels_select, H, ridgeness, face_areas,
                valleys, Usteps, ind, global_vertex_labels, max_label,
                cfg,
            )
        else:
            max_label = _process_std_patch(
                jjj, mesh, face_labels_select, all_boundary_loop,
                H, ridgeness, valleys, face_areas, Usteps, ind,
                global_vertex_labels, max_label,
                savemeshfolder, cfg,
            )

    # ------------------------------------------------------------------
    # 9. Final diffusion + basal mask
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
        mesh_V=mesh.vertices,
        mesh_F=mesh.faces,
        cMCF_steps=Usteps,
        basal_binary=basal_binary,
        dists=dists,
        H=H,
        H_gauss=H_gauss,
        H_mean=H_mean,
        cMCF_stop_ind=ind,
        output_paths={k: v for k, v in paths.items()},
        ap=ap_result,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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


def _process_large_patch(jjj, mesh, face_labels_select, H, ridgeness, face_areas,
                          valleys, Usteps, ind, global_vertex_labels, max_label, cfg):
    """Handle a patch flagged as over-sized."""
    lcfg = cfg.large_patch
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
        np.vstack([attributes[0][:, 3]] * 3).T,
    )[:, 0]
    vertex_areas_disk = igl.average_onto_vertices(
        protrude_submesh.vertices, protrude_submesh.faces,
        np.vstack([attributes[0][:, 1]] * 3).T,
    )[:, 0]

    mean_ridgeness_patch = float(np.nanmean(vertex_ridgeness_disk))

    if mean_ridgeness_patch < lcfg.ridgeness_threshold:
        # Bleb-like: segment by curvature H
        binary_H, _ = binary_threshold(
            vertex_H_disk,
            method=lcfg.H_segment_method,
            n_otsu_levels=lcfg.H_segment_otsu_n_levels,
            level=lcfg.H_segment_otsu_level,
        )
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

        H_cc_labels_simple = []
        face_H_protrude = attributes[0][:, 0].copy()

        for lab_ii, lab in enumerate(H_cc_labels):
            is_simple = meshtools.pca_mesh_patch_and_check_simple(
                protrude_submesh, lab, dilate_ksize=cfg.std_patch.planarity_check_dilate_binary
            )
            if is_simple:
                H_cc_labels_simple.append(lab)
            else:
                _, rethr = binary_threshold(
                    face_H_protrude[lab],
                    method=lcfg.second_seg_H_method,
                    n_otsu_levels=lcfg.second_seg_H_otsu_n_levels,
                    level=lcfg.second_seg_H_otsu_level,
                )
                sub_binary = np.zeros(len(protrude_submesh.faces), dtype=bool)
                sub_binary[lab] = face_H_protrude[lab] >= rethr
                cplx_mesh = meshtools.create_mesh(
                    protrude_submesh.vertices, faces=protrude_submesh.faces[sub_binary]
                )
                cplx_cc = meshtools.connected_components_mesh(
                    cplx_mesh, original_face_indices=np.where(sub_binary)[0]
                )
                for cc in cplx_cc:
                    if len(cc) >= cfg.min_size_comps_protrude_patch:
                        H_cc_labels_simple.append(cc)

        protrude_submesh_labels = np.zeros(len(protrude_submesh.vertices), dtype=np.uint16)
        for ii, cc in enumerate(H_cc_labels_simple):
            verts = np.unique(protrude_submesh.faces[cc].ravel())
            protrude_submesh_labels[verts] = ii + 1

    else:
        # Ridge-like: segment by ridgeness
        binary_ridge, _ = binary_threshold(
            vertex_ridgeness_disk,
            method=lcfg.ridge_segment_method,
            n_otsu_levels=lcfg.ridge_otsu_n_levels,
            level=lcfg.ridge_otsu_level,
        )
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

        protrude_submesh_labels = np.zeros(len(protrude_submesh.vertices), dtype=np.uint16)
        for ii, cc in enumerate(ridge_cc_labels):
            verts = np.unique(protrude_submesh.faces[cc].ravel())
            protrude_submesh_labels[verts] = ii + 1

        uniq_regions = np.setdiff1d(np.unique(protrude_submesh_labels), 0)
        areas_regions = [np.sum(vertex_areas_disk[protrude_submesh_labels == rr]) for rr in uniq_regions]
        areas_occ = np.sum(areas_regions) / float(np.sum(vertex_areas_disk))

        if areas_occ > lcfg.occ_threshold:
            protrude_submesh_labels = np.ones(len(protrude_submesh_labels), dtype=np.uint8)

        # Diffuse
        W = meshtools.vertex_geometric_affinity_matrix(
            protrude_submesh, gamma=None, eps=1e-12, alpha=0.25, normalize=True
        )
        protrude_submesh_labels = meshtools.labelspreading_mesh(
            v=protrude_submesh.vertices, f=protrude_submesh.faces,
            x=np.where(protrude_submesh_labels > 0)[0],
            y=protrude_submesh_labels[protrude_submesh_labels > 0],
            W=W, niters=5, alpha_prop=0.9, return_proba=False, renorm=False,
        )

    max_label = _map_labels_to_global(
        unique_verts_coords, protrude_submesh, protrude_submesh_labels,
        global_vertex_labels, unique_verts_cc, max_label,
    )
    return max_label


def _process_std_patch(jjj, mesh, face_labels_select, all_boundary_loop,
                        H, ridgeness, valleys, face_areas, Usteps, ind,
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

    mean_ridgeness_patch = float(np.nanmean(vertex_ridgeness_disk))
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

    if check and mean_ridgeness_patch < scfg.sure_noridge_threshold:
        # Multi-bleb check
        areas_regions = [np.sum(vertex_areas_disk[protrude_submesh_labels == rr]) for rr in uniq_regions]
        areas_occ = np.sum(areas_regions) / float(np.sum(vertex_areas_disk))
        aspect_regions = [
            meshtools.stretch_pts_set(protrude_submesh.vertices[protrude_submesh_labels == rr])
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
        recovered_frac = (watershed_points_nunique + n_regions_with_pts) / float(len(uniq_regions) + len(coords))

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

        to_split = (
            recovered_frac > scfg.multibleb_min_recovered_frac
            and areas_occ > scfg.multibleb_min_occ_area_fraction
            and mean_aspect_regions < scfg.multibleb_max_mean_aspect_ratio
        )
        if to_split and scfg.multibleb_check_neck_neg_curvature:
            to_split = np.min(pca_boundary_curvature) < scfg.multibleb_neck_neg_curvature_thresh

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
    else:
        # Ridge segmentation
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

    max_label = _map_labels_to_global(
        unique_verts_coords, protrude_submesh, protrude_submesh_labels,
        global_vertex_labels, unique_verts_cc, max_label,
    )
    return max_label
