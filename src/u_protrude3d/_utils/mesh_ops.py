import numpy as np
import scipy.sparse as spsparse
import igl
import unwrap3D.Mesh.meshtools as meshtools


def mesh_laplacian(n_vertices, faces):
    """Unweighted graph Laplacian L = D - A."""
    from scipy.sparse import coo_matrix, diags
    rows, cols = [], []
    for tri in np.asarray(faces):
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


def cotangent_laplacian_operator(V, F):
    """Mass-normalised cotangent Laplacian M^{-1} L as a sparse matrix."""
    L = igl.cotmatrix(V, F)
    M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI)
    M_diag = np.asarray(M.sum(axis=1)).ravel()
    Minv = 1.0 / (M_diag + 1e-12)
    A = L.tocsr().copy()
    A.data *= np.repeat(Minv, np.diff(A.indptr))
    return A


def mesh_smooth_scalar(mesh, values, delta=0.5, n_iters=50):
    """Laplacian smoothing of a scalar field defined on mesh vertices.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    values : (N,) float array
    delta : float – diffusion step size
    n_iters : int – number of smoothing iterations

    Returns
    -------
    smoothed : (N,) float array
    """
    result = meshtools.smooth_scalar_function_mesh(
        mesh,
        scalar_fn=np.asarray(values)[:, None],
        delta=delta,
        exact=False,
        n_iters=n_iters,
        weights=None,
        return_weights=False,
        alpha=0,
    )
    return result[:, 0]


def extract_connected_components(mesh, binary_mask, min_size=20):
    """Extract connected components of the sub-mesh selected by *binary_mask*.

    Applies `scipy.stats.mode` to map vertex binary to face binary, builds the
    sub-mesh, finds connected components, and discards components smaller than
    *min_size* faces.

    Parameters
    ----------
    mesh : trimesh.Trimesh – full surface mesh
    binary_mask : (N_vertices,) bool/int array
    min_size : int – minimum number of faces per component

    Returns
    -------
    cc_labels : list of ndarray – face index arrays for each component
    vertex_label_array : (N_vertices,) int array – 0 = background, 1..K = labels
    """
    import scipy.stats as spstats

    binary_faces = spstats.mode(
        (np.asarray(binary_mask) > 0)[mesh.faces] * 1, axis=1
    )[0]
    binary_faces = np.squeeze(binary_faces)

    sub_mesh = meshtools.create_mesh(
        mesh.vertices, faces=mesh.faces[binary_faces > 0]
    )
    cc_labels = meshtools.connected_components_mesh(
        sub_mesh,
        original_face_indices=np.where(binary_faces > 0)[0],
    )
    cc_labels = [cc for cc in cc_labels if len(cc) >= min_size]

    vertex_label_array = np.zeros(len(mesh.vertices), dtype=np.int32)
    for ii, cc in enumerate(cc_labels):
        verts = np.unique(mesh.faces[cc].ravel())
        vertex_label_array[verts] = ii + 1

    return cc_labels, vertex_label_array
