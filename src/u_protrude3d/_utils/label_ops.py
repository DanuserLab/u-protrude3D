import numpy as np
import skimage.segmentation as sksegmentation
import unwrap3D.Mesh.meshtools as meshtools


def propagate_labels_on_surface(mesh, initial_labels, n_iters=1, rebinarize_prob=0.25):
    """Spread a binary label on the mesh surface via affinity-matrix diffusion.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    initial_labels : (N_vertices,) int/bool array – 1 = foreground, 0 = background
    n_iters : int – number of label-spreading iterations
    rebinarize_prob : float – probability threshold for re-binarizing

    Returns
    -------
    propagated : (N_vertices,) int array
    """
    W = meshtools.vertex_geometric_affinity_matrix(
        mesh, gamma=None, eps=1e-12, alpha=0.5, normalize=True
    )
    propagated, _ = meshtools.labelspreading_mesh_binary(
        mesh.vertices,
        mesh.faces,
        (np.asarray(initial_labels) == 1) * 1,
        W=W,
        niters=n_iters,
        return_proba=True,
        thresh=rebinarize_prob,
    )
    return np.asarray(propagated)


def remove_small_label_components(mesh, labels, min_size=5):
    """Zero out label IDs whose connected component has fewer than *min_size* vertices.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    labels : (N_vertices,) int array
    min_size : int

    Returns
    -------
    cleaned : (N_vertices,) int array
    """
    return meshtools.remove_small_mesh_components_labels(
        mesh.vertices,
        mesh.faces,
        bg_label=0,
        labels=np.asarray(labels),
        vertex_labels_bool=True,
        physical_size=True,
        minsize=min_size,
        keep_largest_only=True,
    )


def relabel_sequential(labels):
    """Relabel integer array so IDs are contiguous starting at 1."""
    return sksegmentation.relabel_sequential(np.asarray(labels))[0]
