import numpy as np
import unwrap3D.Mesh.meshtools as meshtools


def pca_rotate_mesh(mesh, contour_level=0.5):
    """PCA-align a mesh patch and return the model and mean.

    Parameters
    ----------
    mesh : trimesh.Trimesh – patch sub-mesh
    contour_level : float – passed to :func:`meshtools.PCA_rotate_mesh`

    Returns
    -------
    pca_model : fitted sklearn PCA
    mean_pts : (3,) float array – centroid used for centering
    """
    pca_model, mean_pts = meshtools.PCA_rotate_mesh(
        binary=None, mesh=mesh, mesh_contour_level=contour_level
    )
    return pca_model, mean_pts


def compute_aspect_ratio(pts):
    """Aspect ratio (max_range / min_range) of a point set projected to 2D.

    Parameters
    ----------
    pts : (N, 2) or (N, 3) float array – the first two columns are used.

    Returns
    -------
    stretch : float ≥ 1
    """
    pts = np.asarray(pts)
    range_x = pts[:, 0].max() - pts[:, 0].min()
    range_y = pts[:, 1].max() - pts[:, 1].min()
    range_xy = np.array([range_x, range_y])
    return (np.max(range_xy) + 0.1) / (np.min(range_xy) + 0.1)


def compute_planarity(pca_img_pts, boundary_indices, dilate_ksize=2):
    """Check whether interior vertices lie within the PCA-projected boundary.

    Parameters
    ----------
    pca_img_pts : (N, 2) float array – 2D image-space positions of all vertices
    boundary_indices : (B,) int array – indices of boundary vertices
    dilate_ksize : int – dilation radius applied to rasterised boundary polygon

    Returns
    -------
    frac_outside : float – fraction of interior vertices outside the polygon
    canvas : (H, W) bool array – rasterised boundary mask
    """
    import skimage.morphology as skmorph
    from skimage.draw import polygon

    pca_img_pts = np.asarray(pca_img_pts)
    H = int(pca_img_pts[:, 0].max()) + 2
    W = int(pca_img_pts[:, 1].max()) + 2
    canvas = np.zeros((H, W), dtype=bool)

    bpts = pca_img_pts[boundary_indices]
    for offset in [(0, 0), (0.5, 0.5)]:
        rr, cc = polygon(
            (bpts[:, 0] + offset[0]).astype(int),
            (bpts[:, 1] + offset[1]).astype(int),
            shape=(H, W),
        )
        canvas[rr, cc] = True

    canvas = skmorph.binary_dilation(canvas, skmorph.disk(dilate_ksize))

    inner = np.setdiff1d(np.arange(len(pca_img_pts)), boundary_indices)
    test_bool = canvas[
        pca_img_pts[inner, 0].astype(int),
        pca_img_pts[inner, 1].astype(int),
    ]
    frac_outside = np.sum(test_bool == 0) / float(len(test_bool)) if len(test_bool) > 0 else 0.0
    return frac_outside, canvas
