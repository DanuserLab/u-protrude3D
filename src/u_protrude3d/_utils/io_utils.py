import os
from pathlib import Path

import numpy as np
import scipy.io as spio
import unwrap3D.Mesh.meshtools as meshtools
import unwrap3D.Utility_Functions.file_io as fio


def load_mesh(path):
    """Load a triangle mesh from an .obj or .mat file.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    mesh : trimesh.Trimesh
    """
    import h5py

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == '.obj':
        return meshtools.read_mesh(
            str(path), process=True, validate=True, keep_largest_only=True
        )
    elif suffix == '.mat':
        try:
            mat = spio.loadmat(str(path))
            verts = mat['vertices']
            faces = mat['faces'].astype(int) - 1
        except Exception:
            with h5py.File(str(path), 'r') as f:
                var = f['surface']
                faces = np.array(var['faces']).astype(int) - 1
                verts = np.array(var['vertices'])
            verts = verts.T
            faces = faces.T
        return meshtools.create_mesh(vertices=verts, faces=faces)
    else:
        raise ValueError(f'Unsupported mesh format: {suffix}')


def load_mat_labels(path, key='protrusion_labels'):
    """Load a label array from a .mat file.

    Parameters
    ----------
    path : str or Path
    key : str – variable name inside the .mat file

    Returns
    -------
    labels : (N,) int array
    """
    return np.squeeze(spio.loadmat(str(path))[key])


def ensure_dir(path):
    """Create directory (and parents) if it does not exist.

    Returns the Path object.
    """
    path = Path(path)
    fio.mkdir(str(path))
    return path


def build_save_paths(save_dir, basename):
    """Return a dict of standard output file paths for a single cell.

    Parameters
    ----------
    save_dir : str or Path – per-cell save directory
    basename : str – cell identifier (no extension)

    Returns
    -------
    dict[str, Path]
    """
    d = Path(save_dir)
    return {
        'mesh_obj': d / 'mesh.obj',
        'mesh_gt_color_obj': d / 'mesh_gt_protrusion_color.obj',
        'gt_labels_mat': d / 'protrusion_labels_GT_surface.mat',
        'raw_stats_mat': d / 'raw_surface_stats.mat',
        'smooth_stats_mat': d / 'smooth_surface_stats.mat',
        'instance_stats_mat': d / 'instance_protrusion_segmentation_stats.mat',
        'height_color_obj': d / 'surface_mesh_height-color.obj',
        'curvature_color_obj': d / 'surface_mesh_curvature-color.obj',
        'height_binary_obj': d / 'surface_mesh_height_binary-color.obj',
        'basal_binary_obj': d / 'surface_basal_binary-color.obj',
        'final_labels_obj': d / 'full_global_protrude_segment_multilevel_method3_expanded_v1.obj',
        'cMCF_ref_obj': d / 'ext_cMCF_reference_height.obj',
        'cMCF_plot_svg': d / 'external_MCF_iteration_determination.svg',
        'protrude_submesh_dir': d / 'protrusions_to_rethreshold',
    }
