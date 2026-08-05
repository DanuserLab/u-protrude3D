"""Per-step (mesh-based) curvature-gated protrusion erosion.

Unlike the constant-image cMCF in ``segment.py`` step 4 (whose external field is
computed once from the original binary and therefore never lets a region "rest"),
this flow **recomputes the mesh curvature every iteration**.  As a protrusion
flattens its curvature drops toward the basal level, its drive goes to zero, and
the flow self-terminates on the discovered basal surface.

Each step:
  1. recompute per-vertex mean curvature ``H`` (igl principal curvature) and outward
     normals on the *current* mesh;
  2. discover / hold a basal curvature level ``tau`` (Otsu on ``H``);
  3. drive only convex-above-basal vertices inward: ``speed = max(H - tau, 0)``,
     ``force = -speed * normal``  (concave and near-basal get zero force);
  4. implicit Laplacian solve ``(M - deltaL*L) x = M (x + step*force)`` for stability;
  5. optionally project any vertex that left the binary back onto its surface.
"""

from __future__ import annotations

import numpy as np
import igl


def _otsu(x):
    import skimage.filters as skf
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size < 3 or np.unique(x).size < 3:
        return float(np.mean(x)) if x.size else 0.0
    return float(skf.threshold_otsu(x))


def _project_into_binary(V, mesh_binary, sdf_vol, sdf_vol_normal):
    """Push vertices outside the binary back to its zero level set (in place)."""
    shape = np.array(mesh_binary.shape) - 1
    c = np.clip(V.astype(int), 0, shape)
    out = mesh_binary[c[:, 0], c[:, 1], c[:, 2]] == 0
    if np.any(out):
        d = sdf_vol[c[out, 0], c[out, 1], c[out, 2]]
        n = sdf_vol_normal[:, c[out, 0], c[out, 1], c[out, 2]].T
        V[out] = V[out] - d[:, None] * n
    return V


def erode_protrusions_meshflow(
    mesh,
    mesh_binary=None,
    sdf_vol=None,
    sdf_vol_normal=None,
    niters: int = 20,
    step_size: float = 0.25,
    deltaL: float = 5e-4,
    curvature_radius: int = 5,
    tau_mode: str = 'fixed',
    project: bool = True,
    solver: str = 'pardiso',
):
    """Self-terminating curvature-gated inward erosion (per-step recomputed).

    Parameters
    ----------
    mesh : trimesh.Trimesh -- input surface (in voxel coordinates if projecting).
    mesh_binary, sdf_vol, sdf_vol_normal : the containing binary and its signed
        distance / outward-normal fields (from ``segmentation.mean_curvature_binary``).
        Required only when ``project=True``.
    niters : int -- max iterations (the flow self-terminates before this).
    step_size : float -- inward advection step.
    deltaL : float -- implicit Laplacian smoothing stiffness.
    curvature_radius : int -- neighbourhood radius for ``igl.principal_curvature``.
    tau_mode : {'fixed', 'per_step'} -- ``'fixed'`` discovers the basal curvature
        level once (step 0) and holds it (clean convergence); ``'per_step'``
        re-discovers it each iteration.
    project : bool -- clamp every step inside ``mesh_binary`` (containment).
    solver : {'pardiso', 'scipy'}.

    Returns
    -------
    Usteps : (N, 3, niters+1) array -- vertex positions at every step.
    tau : float -- the discovered basal curvature level.
    incr : (niters,) array -- mean per-step displacement (convergence trace).
    """
    import scipy.sparse.linalg as sla
    spsolve = sla.spsolve
    if solver == 'pardiso':
        try:
            import pypardiso
            spsolve = pypardiso.spsolve
        except Exception:
            pass

    V = np.array(mesh.vertices, dtype=np.float64)
    F = np.array(mesh.faces, dtype=np.int32)
    Usteps = np.zeros((len(V), 3, niters + 1))
    Usteps[..., 0] = V
    incr = np.zeros(niters)
    tau = None

    for it in range(niters):
        _, _, k1, k2 = igl.principal_curvature(V, F, radius=curvature_radius)
        Hmean = np.nan_to_num(0.5 * (k1 + k2))   # positive at convex protrusion tips
        N = np.nan_to_num(igl.per_vertex_normals(V, F))

        if tau is None or tau_mode == 'per_step':
            tau = _otsu(Hmean)

        speed = np.clip(Hmean - tau, 0.0, None)  # erode only convex above basal
        force = -N * speed[:, None]              # inward

        # robust (mollified) Laplacian -> stable on degenerate / thin triangles.
        # robust_laplacian returns L positive semi-definite, so the implicit
        # smoothing solve is (M + deltaL*L) x = M (x + step*force).
        import robust_laplacian
        L, M = robust_laplacian.mesh_laplacian(np.ascontiguousarray(V),
                                               np.ascontiguousarray(F))
        S = (M + deltaL * L).tocsc()
        b = M @ (V + step_size * force)
        try:
            Vn = np.asarray(spsolve(S, b))
        except Exception:
            Vn = np.asarray(sla.spsolve(S, b))

        if not np.all(np.isfinite(Vn)):     # numerical safety: hold last good state
            Vn = V.copy()

        if project and mesh_binary is not None:
            Vn = _project_into_binary(Vn, mesh_binary, sdf_vol, sdf_vol_normal)

        incr[it] = float(np.mean(np.linalg.norm(Vn - V, axis=1)))
        V = Vn
        Usteps[..., it + 1] = V

    return Usteps, tau, incr
