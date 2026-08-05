from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io as spio
import scipy.ndimage as ndimage
import scipy.stats as spstats
import skimage.io as skio
import skimage.segmentation as sksegmentation
import skimage.transform as sktform
import matplotlib.pyplot as plt
import igl

import unwrap3D.Mesh.meshtools as meshtools
import unwrap3D.Segmentation.segmentation as unwrap3D_segmentation

from .config import VolumeConfig
from ._utils.io_utils import load_mesh, ensure_dir


@dataclass
class VolumeResult:
    """Return value of :func:`volumize_protrusions`."""
    volume_labels: np.ndarray
    cell_binary: np.ndarray
    basal_binary: np.ndarray
    surface_labels: np.ndarray
    output_paths: dict


def iterative_tutte_cotangent_spherical_map(
    v, f,
    deltaL=5e-3, min_iter=10, max_iter=25, mollify_factor=1e-5, scale=1.0
):
    """Iterative spherical quasi-conformal map for a genus-0 closed surface.

    Parameters
    ----------
    v : (N, 3) float array – vertex positions
    f : (N, 3) int array – face indices
    deltaL : float – step size
    min_iter, max_iter : int – iteration bounds
    mollify_factor : float – robust Laplacian mollification
    scale : float – output sphere radius

    Returns
    -------
    param : (N, 3) float array – spherical coordinates
    n_inverted : list of int – inverted face count per iteration
    """
    import scipy.sparse as spsparse
    import robust_laplacian

    mesh = meshtools.create_mesh(v, f)
    inverted = np.sign(mesh.volume) == -1

    pts = v.copy()
    center = igl.doublearea(v, f)[:, None] * igl.barycenter(v, f)
    center = np.sum(center / np.sum(igl.doublearea(v, f)), axis=0)
    pts = pts - center[None, :]
    pts = pts * scale / (np.linalg.norm(pts, axis=-1)[:, None] + 1e-20)

    U = pts.copy()
    F = mesh.faces.copy()
    L = igl.cotmatrix(v, f)
    n_inverted = []

    for ii in range(max_iter):
        U_prev = U.copy()
        _, M = robust_laplacian.mesh_laplacian(np.array(U), np.array(F), mollify_factor=mollify_factor)
        S = M - deltaL * L
        b = M.dot(U_prev)
        U = spsparse.linalg.spsolve(S, b)

        area = np.sum(igl.doublearea(U, F) * 0.5)
        c = np.sum((0.5 * igl.doublearea(U, F) / area)[..., None] * igl.barycenter(U, F), axis=0)
        U = U - c[None, :]
        U = U * scale / (np.linalg.norm(U, axis=-1)[:, None] + 1e-12)

        out_mesh = meshtools.create_mesh(vertices=U, faces=f)
        f_normals = out_mesh.face_normals.copy()
        if inverted:
            f_normals = f_normals * -1
        centers = igl.barycenter(out_mesh.vertices, out_mesh.faces)
        inner = np.nansum(centers * f_normals, axis=-1)
        n_inv = int(np.sum(inner < 0))
        n_inverted.append(n_inv)

        if n_inv == 0 and ii >= min_iter:
            break

    return U.copy(), n_inverted


def _mesh_shrinkwrap_meniscus(basal_submesh, cfg):
    """Mesh-based shrinkwrap with optional per-step meniscus constraint.

    When cfg.mesh_sw_enable_hole_masking is True: bootstraps the hole distance
    threshold from an initial 10-step wrap, then runs the full deformation with
    that threshold passed to parametric_mesh_mesh_flow.  Vertices farther than
    the threshold from the target receive zero force — membrane stiffness bridges
    them naturally, mimicking the zero-gradient-at-holes behaviour of GVF.

    When cfg.mesh_sw_enable_hole_masking is False (default): runs a single
    standard shrinkwrap call with no hole masking.
    """
    _common = dict(
        decay_rate=cfg.decay_rate,
        remesh_iters=10,
        min_size=cfg.mesh_sw_min_size,
        min_lr=cfg.min_lr,
        genus0_alpha_frac=cfg.mesh_sw_genus0_alpha_frac,
        genus0_alpha_auto=cfg.mesh_sw_genus0_alpha_auto,
        genus0_tol=1e-3,
        deltaL=5e-4,
        alpha=0.1,
        beta=0.5,
        solver='pardiso',
        anchor_factor=cfg.mesh_sw_anchor_factor,
        concavity_boost_factor=cfg.mesh_sw_concavity_boost,
        force_sigma=cfg.mesh_sw_force_sigma,
        convergence_sq_dist=cfg.mesh_sw_convergence_sq_dist,
        topology_repair_alpha_frac=cfg.mesh_sw_topology_repair_alpha_frac,
        vfc_sigma=cfg.mesh_sw_vfc_sigma,
        vfc_blend=cfg.mesh_sw_vfc_blend,
        smooth_iters=cfg.mesh_sw_smooth_iters,
        balloon_factor=cfg.mesh_sw_balloon_factor,
        debugviz=False,
    )

    if not cfg.mesh_sw_enable_hole_masking:
        mesh_wrap, mesh_genus0, stats = meshtools.shrinkwrap_genus0_meshbased(
            basal_submesh,
            total_shrinkwrap_iters=cfg.total_shrinkwrap_iters,
            enable_hole_masking=False,
            hole_sq_dist_threshold=None,
            **_common,
        )
        return mesh_wrap, mesh_genus0, stats

    # Hole masking enabled: bootstrap threshold from a short initial run.
    mesh_wrap_init, mesh_genus0, _ = meshtools.shrinkwrap_genus0_meshbased(
        basal_submesh,
        total_shrinkwrap_iters=10,
        enable_hole_masking=False,
        hole_sq_dist_threshold=None,
        **_common,
    )

    if cfg.mesh_sw_hole_sq_dist_threshold is not None and cfg.mesh_sw_hole_sq_dist_threshold > 0:
        hole_thresh = cfg.mesh_sw_hole_sq_dist_threshold
    else:
        mean_el = igl.avg_edge_length(
            np.array(mesh_wrap_init.vertices), np.array(mesh_wrap_init.faces, dtype=np.int32))
        hole_thresh = (cfg.mesh_sw_hole_edge_length_factor * mean_el) ** 2

    mesh_wrap, _, stats = meshtools.shrinkwrap_genus0_meshbased(
        basal_submesh,
        total_shrinkwrap_iters=cfg.total_shrinkwrap_iters,
        enable_hole_masking=True,
        hole_sq_dist_threshold=hole_thresh,
        **_common,
    )

    return mesh_wrap, mesh_genus0, stats


def volumize_protrusions(
    mesh_path: str | os.PathLike,
    protrusion_labels: np.ndarray,
    tif_path: str | os.PathLike,
    save_dir: str | os.PathLike,
    cMCF_steps: Optional[np.ndarray] = None,
    cfg: Optional[VolumeConfig] = None,
) -> VolumeResult:
    """Map surface protrusion labels into a 3D volumetric representation.

    The function:
    1. Identifies the basal (non-protrusion) surface.
    2. Shrinkwraps a genus-0 mesh to the basal surface using GVF.
    3. Advects the full mesh inward using a signed-distance gradient.
    4. Paints each voxel visited by a vertex with its protrusion label.

    Parameters
    ----------
    mesh_path : str or Path
        .obj mesh file (typically the coloured segmentation output of
        :func:`segment_protrusions`).
    protrusion_labels : (N_vertices,) int array
        Per-vertex instance labels from :func:`segment_protrusions`
        (0 = background, 1..K = instances).
    tif_path : str or Path
        Reference TIFF volume defining the cell binary space.
    save_dir : str or Path
        Directory where output TIFF and .obj files are written.
    cMCF_steps : (N, 3, T) array or None
        External cMCF trajectory from :func:`segment_protrusions`.
        Currently reserved for future use; pass None.
    cfg : VolumeConfig or None
        Algorithm parameters. Uses defaults when None.

    Returns
    -------
    VolumeResult
        Fields: volume_labels (Z, Y, X int16), cell_binary, basal_binary,
        surface_labels, output_paths.
    """
    if cfg is None:
        cfg = VolumeConfig()

    np.random.seed(cfg.random_seed)
    save_dir = ensure_dir(save_dir)
    protrusion_labels = np.asarray(protrusion_labels)

    # Colour palette
    protrude_colors = np.vstack(
        __import__('seaborn').color_palette('Spectral', cfg.n_protrude_colors)
    )
    rng = np.random.default_rng(cfg.random_seed)
    rng.shuffle(protrude_colors)

    # ------------------------------------------------------------------
    # 1. Load mesh
    # ------------------------------------------------------------------
    mesh = load_mesh(mesh_path)

    # ------------------------------------------------------------------
    # 2. Build basal sub-mesh (remove protrusion faces with 1-ring erosion)
    # ------------------------------------------------------------------
    protrusion_labels_faces = spstats.mode(
        protrusion_labels[mesh.faces] * 1, axis=-1
    )[0]
    if len(protrusion_labels_faces.shape) == 2:
        protrusion_labels_faces = protrusion_labels_faces[:, 0]

    # Erode labels by one ring at the boundary
    protrusion_labels_shrink = protrusion_labels.copy()
    vertex_set_0 = []
    for cc in np.setdiff1d(np.unique(protrusion_labels), 0):
        b_loop = igl.boundary_loop(mesh.faces[protrusion_labels_faces == cc])
        if len(b_loop) > 0:
            vertex_set_0.append(b_loop)
    if vertex_set_0:
        protrusion_labels_shrink[np.hstack(vertex_set_0)] = 0

    # Offset labels (0 → background, 1 → basal, 2..K+1 → protrusions)
    protrusion_labels_shrink = protrusion_labels_shrink + 1
    protrusion_labels_work = protrusion_labels + 1

    no_protrusion_faces = np.ones(len(mesh.faces), dtype=bool)
    no_protrusion_faces[
        np.nanmean(((protrusion_labels_shrink > 1) * 1)[mesh.faces], axis=1) > 0.5
    ] = False

    basal_submesh, _ = meshtools.submesh(
        mesh,
        faces_sequence=[no_protrusion_faces],
        mesh_face_attributes=np.ones(len(mesh.faces))[:, None],
        repair=False, only_watertight=False, min_faces=None,
    )
    basal_submesh = basal_submesh[0]
    basal_submesh.export(str(save_dir / 'no_protrusion_mesh.obj'))

    # ------------------------------------------------------------------
    # 3. Shrinkwrap basal surface
    # ------------------------------------------------------------------
    basefname = Path(mesh_path).stem
    if cfg.use_mesh_based_shrinkwrap:
        mesh_shrinkwrap, mesh_genus0, iter_stats_raw = _mesh_shrinkwrap_meniscus(
            basal_submesh, cfg)
        chamfer_dists          = list(iter_stats_raw[0])
        all_meshes_iter_genus0 = list(iter_stats_raw[1])
        all_meshes_iter        = list(iter_stats_raw[2])
        mesh_genus0.export(str(save_dir / 'initial_genus0_alphawrap.obj'))
        meshtools.decimate_resample_mesh(mesh_genus0, remesh_samples=0.5).export(
            str(save_dir / 'initial_genus0_alphawrap_remeshed.obj')
        )
    else:
        mesh_shrinkwrap, _, mesh_genus0, iter_stats = meshtools.shrinkwrap_genus0_basic(
            basal_submesh,
            use_GVF=True,
            GVF_mu=cfg.gvf_mu,
            GVF_iterations=cfg.gvf_iters,
            voxelize_padsize=cfg.voxelize_padsize,
            voxelize_dilate_ksize=cfg.voxelize_dilate_ksize,
            voxelize_erode_ksize=cfg.voxelize_erode_ksize,
            extra_pad=cfg.extra_pad,
            genus0_alpha_frac=0.2,
            genus0_alpha_auto=False,
            genus0_tol=1e-3,
            total_shrinkwrap_iters=cfg.total_shrinkwrap_iters,
            decay_rate=cfg.decay_rate,
            remesh_iters=10,
            conformalize=False,
            min_size=10e3,
            upsample=1,
            min_lr=cfg.min_lr,
            make_manifold=False,
            watertight_fraction=0.1,
            deltaL=5e-4,
            alpha=0.1,
            beta=0.5,
            solver='pardiso',
            curvature_weighting=False,
            vfc_sigma=cfg.gvf_vfc_sigma,
            vfc_blend=cfg.gvf_vfc_blend,
            debugviz=False,
        )
        chamfer_dists          = iter_stats[0]
        all_meshes_iter_genus0 = iter_stats[1]
        all_meshes_iter        = iter_stats[2]

    mesh_shrinkwrap = meshtools.largest_component_mesh(mesh_shrinkwrap)

    mesh_wrap = mesh_shrinkwrap.copy()
    for _ in range(cfg.n_punchout_refinements):
        gc.collect()
        mesh_wrap = meshtools.remove_nonshrinkwrap_triangles(
            mesh_wrap, mesh,
            q_thresh=0.1, max_dist_cutoff=1.25, sigma_dist_cutoff=2,
            sigma_area_cutoff=2, nearest_k=1, dilate_voxels=2, erode_voxels=1, minsize=10,
        )
        mesh_wrap = meshtools.largest_component_mesh(mesh_wrap)
        if cfg.use_mesh_based_shrinkwrap:
            mesh_shrinkwrap, _, punch_stats = meshtools.shrinkwrap_genus0_meshbased(
                mesh_in=mesh_wrap,
                total_shrinkwrap_iters=100,
                decay_rate=cfg.decay_rate,
                remesh_iters=10,
                min_size=10_000,
                min_lr=cfg.min_lr,
                genus0_alpha_frac=0.2,
                genus0_alpha_auto=True,
                genus0_tol=0.1,
                deltaL=5e-4,
                alpha=0.1,
                beta=0.5,
                solver='pardiso',
                anchor_factor=cfg.mesh_sw_anchor_factor,
                concavity_boost_factor=cfg.mesh_sw_concavity_boost,
                force_sigma=cfg.mesh_sw_force_sigma,
                convergence_sq_dist=cfg.mesh_sw_convergence_sq_dist,
                topology_repair_alpha_frac=cfg.mesh_sw_topology_repair_alpha_frac,
                vfc_sigma=cfg.mesh_sw_vfc_sigma,
                vfc_blend=cfg.mesh_sw_vfc_blend,
                smooth_iters=cfg.mesh_sw_smooth_iters,
                balloon_factor=cfg.mesh_sw_balloon_factor,
                debugviz=False,
            )
            mesh_shrinkwrap = meshtools.largest_component_mesh(mesh_shrinkwrap)
            all_meshes_iter.extend(punch_stats[2])
            all_meshes_iter_genus0.extend(punch_stats[1])
            chamfer_dists.extend(punch_stats[0])
        else:
            mesh_shrinkwrap, _, _, iter_stats2 = meshtools.attract_surface_mesh(
                mesh_in=mesh_wrap, mesh_ref=basal_submesh,
                use_GVF=True, GVF_mu=cfg.gvf_mu, GVF_iterations=cfg.gvf_iters,
                voxelize_padsize=cfg.voxelize_padsize,
                voxelize_dilate_ksize=cfg.voxelize_dilate_ksize,
                voxelize_erode_ksize=cfg.voxelize_erode_ksize,
                extra_pad=cfg.extra_pad,
                tightest_genus0_initial=True,
                genus0_alpha_frac=0.2, genus0_alpha_auto=False, genus0_tol=0.1,
                total_shrinkwrap_iters=100, decay_rate=cfg.decay_rate, remesh_iters=10,
                conformalize=False, min_size=10e3, upsample=1, min_lr=cfg.min_lr,
                make_manifold=False, watertight_fraction=0.1,
                deltaL=5e-4, alpha=0.1, beta=0.5, solver='pardiso',
                curvature_weighting=False,
                vfc_sigma=cfg.gvf_vfc_sigma,
                vfc_blend=cfg.gvf_vfc_blend,
                debugviz=False,
            )
            mesh_shrinkwrap = meshtools.largest_component_mesh(mesh_shrinkwrap)
            all_meshes_iter.append(iter_stats2[2])
            all_meshes_iter_genus0.append(iter_stats2[1])
            chamfer_dists.append(iter_stats2[0])

    # Flatten iteration lists: mesh-based path uses extend (already flat);
    # GVF path uses append (nested after punchout rounds).
    if cfg.use_mesh_based_shrinkwrap or cfg.n_punchout_refinements == 0:
        all_meshes_iter_flat        = all_meshes_iter
        all_meshes_iter_genus0_flat = all_meshes_iter_genus0
        all_chamfer_dists           = np.array(chamfer_dists)
    else:
        n = cfg.n_punchout_refinements
        all_meshes_iter_flat = all_meshes_iter[:-n] + [
            x for item in all_meshes_iter[-n:] for x in item
        ]
        all_meshes_iter_genus0_flat = all_meshes_iter_genus0[:-n] + [
            x for item in all_meshes_iter_genus0[-n:] for x in item
        ]
        all_chamfer_dists = np.hstack(chamfer_dists)

    gauss_steps = np.array([
        np.nanmean(np.abs(igl.gaussian_curvature(m.vertices, m.faces)))
        for m in all_meshes_iter_flat
    ])

    # Pick the final shrinkwrap mesh from the iteration sequence.
    if cfg.shrinkwrap_iter_select == 'min_loss':
        all_chamfer_norm = (all_chamfer_dists - all_chamfer_dists.min()) / (
            all_chamfer_dists.max() - all_chamfer_dists.min() + 1e-20
        )
        gauss_norm = (gauss_steps - gauss_steps.min()) / (
            gauss_steps.max() - gauss_steps.min() + 1e-20
        )
        loss = 0.5 * all_chamfer_norm + 0.5 * gauss_norm
        selected_idx = int(np.argmin(loss))
    else:  # 'last'
        selected_idx = len(all_meshes_iter_flat) - 1
    final_mesh = meshtools.largest_component_mesh(all_meshes_iter_flat[selected_idx])
    final_mesh.export(str(save_dir / 'min_loss_mesh_ds.obj'))

    mesh_shrinkwrap = meshtools.incremental_isotropic_remesh(final_mesh)
    mesh_shrinkwrap.export(str(save_dir / 'no_protrusion_mesh_shrinkwrap_ds.obj'))

    # ------------------------------------------------------------------
    # 4. Build SDF gradient and advect full mesh inward
    # ------------------------------------------------------------------
    mesh_binary = meshtools.voxelize_image_mesh_pts(
        mesh, pad=25, dilate_ksize=cfg.voxelize_dilate_ksize,
        erode_ksize=cfg.voxelize_erode_ksize, pitch=1.2,
    )
    mesh_ref_binary = meshtools.voxelize_image_mesh_pts(
        mesh_shrinkwrap, pad=25,
        dilate_ksize=cfg.voxelize_dilate_ksize,
        erode_ksize=cfg.voxelize_erode_ksize,
        pitch=1.2, vol_shape=mesh_binary.shape,
    )
    mesh_ref_binary[mesh_binary == 0] = 0
    mesh_ref_binary = unwrap3D_segmentation.largest_component_vol(mesh_ref_binary)

    ds = 1.5
    mesh_ref_ds = ndimage.zoom(mesh_ref_binary, zoom=[1.0 / ds] * 3, order=0)
    poisson_inner = unwrap3D_segmentation.poisson_dist_tform_3D(mesh_ref_ds > 0, pts=None)
    poisson_inner = sktform.resize(poisson_inner, mesh_ref_binary.shape, order=1)
    poisson_outer = ndimage.distance_transform_edt(~(mesh_ref_binary > 0))
    poisson_sdf = (
        -1 * poisson_outer * (~(mesh_ref_binary > 0))
        + 1 * poisson_inner * (mesh_ref_binary > 0)
    )

    grad = np.array(np.gradient(poisson_sdf))
    grad /= np.linalg.norm(grad, axis=0)[None, ...] + 1e-20
    grad = np.array([ndimage.gaussian_filter(grad[i], sigma=3) for i in range(3)])
    grad /= np.linalg.norm(grad, axis=0)[None, ...] + 1e-20
    grad = (-grad) * (mesh_ref_binary > 0)[None, ...] + grad * (~(mesh_ref_binary > 0))[None, ...]
    grad = grad.astype(np.float32)

    Usteps_sdf = meshtools.parametric_mesh_constant_img_flow(
        mesh,
        external_img_gradient=grad.transpose(1, 2, 3, 0),
        niters=200, deltaL=5e-4, step_size=1, method='implicit',
        robust_L=True, mollify_factor=1e-5, conformalize=False,
        gamma=1, alpha=0.2, beta=0.5, eps=1e-20,
        solver='pardiso', noprogress=False, normalize_grad=False,
    )

    # ------------------------------------------------------------------
    # 5. Paint volume labels
    # ------------------------------------------------------------------
    volume_protrusion_labels = np.zeros(mesh_binary.shape)
    for jj in range(Usteps_sdf.shape[-1]):
        verts_jj = Usteps_sdf[..., jj]
        volume_protrusion_labels[
            verts_jj[..., 0].astype(np.int32),
            verts_jj[..., 1].astype(np.int32),
            verts_jj[..., 2].astype(np.int32),
        ] = protrusion_labels_work

    volume_protrusion_labels = sksegmentation.expand_labels(volume_protrusion_labels)
    volume_protrusion_labels = unwrap3D_segmentation.largest_component_vol_labels(
        volume_protrusion_labels, connectivity=2
    )
    volume_protrusion_labels[mesh_binary == 0] = 0
    volume_protrusion_labels[mesh_ref_binary > 0] = 0
    volume_protrusion_labels[volume_protrusion_labels == 1] = 0
    volume_protrusion_labels = np.uint16(volume_protrusion_labels)
    volume_protrusion_labels = unwrap3D_segmentation.detect_and_relabel_multi_component_labels(
        volume_protrusion_labels, min_size=10
    )

    # ------------------------------------------------------------------
    # 6. Save outputs
    # ------------------------------------------------------------------
    skio.imsave(str(save_dir / 'volumize_cell_binary.tif'), np.uint8(255.0 * (mesh_binary > 0)))
    skio.imsave(str(save_dir / 'volumize_basal_cell_binary.tif'), np.uint8(255.0 * (mesh_ref_binary > 0)))
    skio.imsave(str(save_dir / 'volumize_protrusion_cell_labels.tif'), volume_protrusion_labels)

    # Colour volume
    volume_color = np.zeros(mesh_binary.shape + (3,))
    for cc in np.setdiff1d(np.unique(protrusion_labels_work), 0):
        volume_color[volume_protrusion_labels == cc] = protrude_colors[int(cc) % len(protrude_colors)]
    skio.imsave(
        str(save_dir / 'protrusions_prop_color.tif'),
        np.uint8(255 * volume_color).transpose(2, 1, 0, 3),
    )

    output_paths = {
        'cell_binary_tif': save_dir / 'volumize_cell_binary.tif',
        'basal_binary_tif': save_dir / 'volumize_basal_cell_binary.tif',
        'protrusion_labels_tif': save_dir / 'volumize_protrusion_cell_labels.tif',
        'color_tif': save_dir / 'protrusions_prop_color.tif',
        'shrinkwrap_obj': save_dir / 'no_protrusion_mesh_shrinkwrap_ds.obj',
    }

    return VolumeResult(
        volume_labels=volume_protrusion_labels,
        cell_binary=mesh_binary,
        basal_binary=mesh_ref_binary,
        surface_labels=protrusion_labels_work,
        output_paths=output_paths,
    )
