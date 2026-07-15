import numpy as np
import skimage.filters as skfilters
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import expm_multiply


def binary_threshold(values, method='multiotsu', n_otsu_levels=3, level=-1):
    """Adaptive threshold returning (binary_mask, threshold_value).

    Parameters
    ----------
    values : array-like
        Scalar values to threshold.
    method : {'multiotsu', 'mean'}
        Thresholding strategy.
    n_otsu_levels : int
        Number of Otsu classes (only used when method='multiotsu').
    level : int
        Which Otsu threshold level to pick (index into sorted thresholds).

    Returns
    -------
    mask : ndarray of bool
    threshold : float
    """
    values = np.asarray(values)
    if method == 'multiotsu':
        threshold = skfilters.threshold_multiotsu(values, classes=n_otsu_levels)[level]
    elif method == 'mean':
        threshold = np.nanmean(values)
    else:
        threshold = skfilters.threshold_multiotsu(values, classes=n_otsu_levels)[level]
    return values >= threshold, threshold


def _mesh_laplacian_unweighted(n_vertices, faces):
    """Unweighted graph Laplacian (internal helper for adaptive thresholding)."""
    rows, cols = [], []
    for tri in faces:
        i, j, k = tri
        for a, b in [(i, j), (j, i), (j, k), (k, j), (k, i), (i, k)]:
            rows.append(a)
            cols.append(b)
    data = np.ones(len(rows))
    A = coo_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices)).tocsr()
    A.data[:] = 1.0
    A.eliminate_zeros()
    degree = np.array(A.sum(axis=1)).ravel()
    return diags(degree) - A


def adaptive_threshold_heat(values, faces, t=1.0, method='sauvola', k=0.3):
    """Adaptive thresholding via heat diffusion on the mesh graph.

    Parameters
    ----------
    values : (N,) array
        Scalar field on vertices.
    faces : (M, 3) int array
        Triangle connectivity.
    t : float
        Diffusion time.
    method : {'mean', 'niblack', 'sauvola'}
    k : float
        Threshold parameter for Niblack/Sauvola.

    Returns
    -------
    binary : uint8 array
    threshold : float array
    mean : float array
    std : float array
    """
    values = np.asarray(values, dtype=float)
    L = _mesh_laplacian_unweighted(len(values), faces)
    mean = expm_multiply(-t * L, values)
    second_moment = expm_multiply(-t * L, values ** 2)
    variance = np.maximum(second_moment - mean ** 2, 0.0)
    std = np.sqrt(variance)

    if method == 'mean':
        threshold = mean
    elif method == 'niblack':
        threshold = mean + k * std
    elif method == 'sauvola':
        R = np.max(std)
        threshold = mean * (1 + k * (std / (R + 1e-12) - 1))
    else:
        raise ValueError(f'Unknown method: {method}')

    binary = (values > threshold).astype(np.uint8)
    return binary, threshold, mean, std


def adaptive_threshold_libigl(V, F, f, t=1.0, method='sauvola', k=0.3):
    """Adaptive thresholding using the cotangent Laplacian (libigl).

    Parameters
    ----------
    V : (N, 3) float array – vertex positions
    F : (M, 3) int array – face indices
    f : (N,) float array – scalar field on vertices
    t, method, k : same as :func:`adaptive_threshold_heat`

    Returns
    -------
    binary, threshold, mu, sigma
    """
    import igl
    from .mesh_ops import cotangent_laplacian_operator

    A = cotangent_laplacian_operator(V, F)
    mu = expm_multiply(-t * A, f)
    mu2 = expm_multiply(-t * A, f ** 2)
    sigma = np.sqrt(np.maximum(mu2 - mu ** 2, 0.0))

    if method == 'mean':
        T = mu
    elif method == 'niblack':
        T = mu + k * sigma
    elif method == 'sauvola':
        R = np.max(sigma)
        T = mu * (1 + k * (sigma / (R + 1e-12) - 1))
    else:
        raise ValueError(f'Unknown method: {method}')

    binary = (f > T).astype(np.uint8)
    return binary, T, mu, sigma
