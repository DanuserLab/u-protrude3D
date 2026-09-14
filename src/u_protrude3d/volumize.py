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
        genus0_tol=cfg.mesh_sw_genus0_tol,
        deltaL=cfg.mesh_sw_deltaL,
        alpha=cfg.mesh_sw_alpha,
        beta=cfg.mesh_sw_beta,
        solver=cfg.mesh_sw_solver,
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
    # Zero-protrusion short-circuit: all labels are 0 → no protrusions
    # to volumize.  Voxelise the cell and return a zero label volume with
    # cell_binary == basal_binary.
    # ------------------------------------------------------------------
    if np.all(protrusion_labels == 0):
        print('[volumize_protrusions] All protrusion labels are zero — skipping shrinkwrap/advection.')
        import skimage.morphology as skmorph
        voxel_size = cfg.voxel_size
        vertices = np.array(mesh.vertices)
        v_min = vertices.min(axis=0)
        v_max = vertices.max(axis=0)
        pad = 4
        shape = tuple(
            int(np.ceil((v_max[i] - v_min[i]) / voxel_size)) + 2 * pad
            for i in range(3)
        )
        # Rasterise mesh surface into a binary volume
        import skimage.draw as skdraw
        cell_binary = np.zeros(shape, dtype=np.uint8)
        from unwrap3D.Segmentation import segmentation as unwrap3D_segmentation
        cell_binary = unwrap3D_segmentation.mesh_to_binary(
            mesh, voxel_size=voxel_size, pad=pad
        ) if hasattr(unwrap3D_segmentation, 'mesh_to_binary') else cell_binary

        volume_labels = np.zeros(shape, dtype=np.uint16)
        import skimage.io as skio_local
        skio_local.imsave(str(save_dir / 'volumize_cell_binary.tif'),
                          np.uint8(255 * (cell_binary > 0)))
        skio_local.imsave(str(save_dir / 'volumize_basal_cell_binary.tif'),
                          np.uint8(255 * (cell_binary > 0)))
        skio_local.imsave(str(save_dir / 'volumize_protrusion_cell_labels.tif'), volume_labels)
        out_paths = {
            'cell_binary_tif':        save_dir / 'volumize_cell_binary.tif',
            'basal_binary_tif':       save_dir / 'volumize_basal_cell_binary.tif',
            'protrusion_labels_tif':  save_dir / 'volumize_protrusion_cell_labels.tif',
        }
        return VolumeResult(
            volume_labels=volume_labels,
            cell_binary=cell_binary,
            basal_binary=cell_binary,   # identical to cell binary when no protrusions
            surface_labels=protrusion_labels + 1,
            output_paths={k: str(v) for k, v in out_paths.items()},
        )

    # ------------------------------------------------------------------
    # 2. Build basal sub-mesh (remove protrusion faces with k-ring erosion)
    # ------------------------------------------------------------------
    protrusion_labels_shrink = protrusion_labels.copy()
    for _ in range(max(1, cfg.label_erosion_rings)):
        _lf = spstats.mode(protrusion_labels_shrink[mesh.faces] * 1, axis=-1)[0]
        if len(_lf.shape) == 2:
            _lf = _lf[:, 0]
        _bverts = []
        for cc in np.setdiff1d(np.unique(protrusion_labels_shrink), 0):
            b_loop = igl.boundary_loop(mesh.faces[_lf == cc])
            if len(b_loop) > 0:
                _bverts.append(b_loop)
        if not _bverts:
            break
        protrusion_labels_shrink[np.hstack(_bverts)] = 0

    protrusion_labels_faces = spstats.mode(
        protrusion_labels_shrink[mesh.faces] * 1, axis=-1
    )[0]
    if len(protrusion_labels_faces.shape) == 2:
        protrusion_labels_faces = protrusion_labels_faces[:, 0]

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
            genus0_alpha_frac=cfg.gvf_sw_genus0_alpha_frac,
            genus0_alpha_auto=cfg.gvf_sw_genus0_alpha_auto,
            genus0_tol=cfg.gvf_sw_genus0_tol,
            total_shrinkwrap_iters=cfg.total_shrinkwrap_iters,
            decay_rate=cfg.decay_rate,
            remesh_iters=10,
            conformalize=cfg.gvf_sw_conformalize,
            min_size=cfg.gvf_sw_min_size,
            upsample=cfg.gvf_sw_upsample,
            min_lr=cfg.min_lr,
            make_manifold=cfg.gvf_sw_make_manifold,
            watertight_fraction=cfg.gvf_sw_watertight_fraction,
            deltaL=cfg.gvf_sw_deltaL,
            alpha=cfg.gvf_sw_alpha,
            beta=cfg.gvf_sw_beta,
            solver=cfg.gvf_sw_solver,
            curvature_weighting=cfg.gvf_sw_curvature_weighting,
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
                total_shrinkwrap_iters=cfg.mesh_sw_punchout_total_iters,
                decay_rate=cfg.decay_rate,
                remesh_iters=10,
                min_size=cfg.mesh_sw_min_size,
                min_lr=cfg.min_lr,
                genus0_alpha_frac=cfg.mesh_sw_punchout_genus0_alpha_frac,
                genus0_alpha_auto=cfg.mesh_sw_punchout_genus0_alpha_auto,
                genus0_tol=cfg.mesh_sw_punchout_genus0_tol,
                deltaL=cfg.mesh_sw_deltaL,
                alpha=cfg.mesh_sw_alpha,
                beta=cfg.mesh_sw_beta,
                solver=cfg.mesh_sw_solver,
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
                genus0_alpha_frac=cfg.gvf_sw_genus0_alpha_frac,
                genus0_alpha_auto=cfg.gvf_sw_genus0_alpha_auto,
                genus0_tol=cfg.gvf_sw_punchout_genus0_tol,
                total_shrinkwrap_iters=cfg.gvf_sw_punchout_total_iters,
                decay_rate=cfg.decay_rate, remesh_iters=10,
                conformalize=cfg.gvf_sw_conformalize, min_size=cfg.gvf_sw_min_size,
                upsample=cfg.gvf_sw_upsample, min_lr=cfg.min_lr,
                make_manifold=cfg.gvf_sw_make_manifold,
                watertight_fraction=cfg.gvf_sw_watertight_fraction,
                deltaL=cfg.gvf_sw_deltaL, alpha=cfg.gvf_sw_alpha, beta=cfg.gvf_sw_beta,
                solver=cfg.gvf_sw_solver,
                curvature_weighting=cfg.gvf_sw_curvature_weighting,
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

    # Loss used to pick the final shrinkwrap mesh from the iteration sequence
    # (always computed, regardless of shrinkwrap_iter_select, so it can be plotted).
    all_chamfer_norm = (all_chamfer_dists - all_chamfer_dists.min()) / (
        all_chamfer_dists.max() - all_chamfer_dists.min() + 1e-20
    )
    gauss_norm = (gauss_steps - gauss_steps.min()) / (
        gauss_steps.max() - gauss_steps.min() + 1e-20
    )
    loss = 0.5 * all_chamfer_norm + 0.5 * gauss_norm

    if cfg.shrinkwrap_iter_select == 'min_loss':
        selected_idx = int(np.argmin(loss))
    else:  # 'last'
        selected_idx = len(all_meshes_iter_flat) - 1
    final_mesh = meshtools.largest_component_mesh(all_meshes_iter_flat[selected_idx])
    final_mesh.export(str(save_dir / 'min_loss_mesh_ds.obj'))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(all_chamfer_norm, label='chamfer (norm)')
    ax.plot(gauss_norm, label='Gauss curvature (norm)')
    ax.plot(loss, label='combined loss', linewidth=2)
    ax.axvline(selected_idx, color='k', linestyle='--', label='selected')
    ax.set_xlabel('shrinkwrap iteration')
    ax.set_ylabel('loss')
    ax.set_title(f"shrinkwrap_iter_select='{cfg.shrinkwrap_iter_select}'")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(save_dir / 'shrinkwrap_loss.png'))
    plt.close(fig)

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
        'shrinkwrap_loss_plot': save_dir / 'shrinkwrap_loss.png',
    }

    return VolumeResult(
        volume_labels=volume_protrusion_labels,
        cell_binary=mesh_binary,
        basal_binary=mesh_ref_binary,
        surface_labels=protrusion_labels_work,
        output_paths=output_paths,
    )
