import numpy as np
import igl


def compute_curvatures(V, F, radius=5):
    """Compute principal and derived curvatures on a triangle mesh.

    Parameters
    ----------
    V : (N, 3) float array – vertex positions
    F : (M, 3) int array – face indices
    radius : int – neighbourhood radius for :func:`igl.principal_curvature`

    Returns
    -------
    dict with keys:
        k1, k2       – principal curvatures (N,)
        H_mean       – mean curvature 0.5*(k1+k2) (N,)
        H_gauss_norm – normalised Gaussian curvature k1*k2/norm (N,)
        ridgeness    – max(k1,k2) - |min(k1,k2)|, clipped ≥0 (N,)
        valleys      – -|max(k1,k2)| - min(k1,k2), clipped ≥0 (N,)
        norm         – normalising constant used for H_gauss_norm (float)
    """
    _, _, k1, k2 = igl.principal_curvature(V, F, radius=radius)[:4]

    norm = np.nanmean(0.5 * (np.abs(k1) + np.abs(k2))) + 1e-20
    H_gauss_norm = k1 * k2 / norm
    H_mean = 0.5 * (k1 + k2)

    ridgeness = np.maximum(k1, k2) - np.abs(np.minimum(k1, k2))
    ridgeness = np.clip(ridgeness, 0, np.inf)

    valleys = -np.abs(np.maximum(k1, k2)) - np.minimum(k1, k2)
    valleys = np.clip(valleys, 0, np.inf)

    # Koenderink Shape Index — scale-invariant dome vs cylinder classifier.
    # SI = (2/π) * arctan((k1+k2) / (k1−k2)), defined where k1 ≠ k2.
    #   SI ≈ +1 : dome / bleb
    #   SI ≈ +0.5 : cylinder / ridge
    #   SI ≈  0  : saddle
    #   SI ≈ -1  : cup / pit
    # Where k1 == k2 (umbilical pts), sign(k1) determines SI = ±1.
    denom = k1 - k2
    umbilical = np.abs(denom) < 1e-12
    shape_index = np.where(
        umbilical,
        np.sign(k1).astype(float),
        (2.0 / np.pi) * np.arctan2(k1 + k2, denom),
    )

    return {
        'k1': k1,
        'k2': k2,
        'H_mean': H_mean,
        'H_gauss_norm': H_gauss_norm,
        'ridgeness': ridgeness,
        'valleys': valleys,
        'shape_index': shape_index,
        'norm': norm,
    }
