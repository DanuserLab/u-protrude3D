import numpy as np
import seaborn as sns
import unwrap3D.Mesh.meshtools as meshtools
import unwrap3D.Visualisation.colors as vol_colors


def get_vertex_colors(labels, palette=None, n_colors=24, random_seed=1232):
    """Map integer vertex labels to RGB colours.

    Parameters
    ----------
    labels : (N,) int array – 0 = background (grey)
    palette : (K, 3) float array or None – if None, Spectral palette is generated
    n_colors : int – palette size when palette=None
    random_seed : int – for shuffling the palette

    Returns
    -------
    colors : (N, 3) float array in [0, 1]
    """
    labels = np.asarray(labels)
    if palette is None:
        rng = np.random.default_rng(random_seed)
        palette = np.vstack(sns.color_palette('Spectral', n_colors))
        rng.shuffle(palette)

    colors = np.ones((len(labels), 3)) * 0.75  # default grey
    for cc in np.setdiff1d(np.unique(labels), 0):
        colors[labels == cc] = palette[int(cc) % len(palette)]
    return colors


def scalar_to_vertex_colors(values, colormap, vmin=None, vmax=None):
    """Map a scalar field on vertices to RGB colours using a matplotlib colormap.

    Parameters
    ----------
    values : (N,) float array
    colormap : matplotlib colormap
    vmin, vmax : float or None – clipping range

    Returns
    -------
    colors : (N, 3) float array in [0, 1]
    """
    return vol_colors.get_colors(values, colormap=colormap, vmin=vmin, vmax=vmax)[:, :3]


def export_colored_obj(mesh, vertex_colors_rgb, path):
    """Write a mesh with per-vertex RGB colours (float [0,1]) to an .obj file.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    vertex_colors_rgb : (N, 3) float array in [0, 1]
    path : str or Path
    """
    colored = meshtools.create_mesh(
        mesh.vertices,
        mesh.faces,
        vertex_colors=np.uint8(255 * np.asarray(vertex_colors_rgb)),
    )
    if colored.volume < 0:
        colored.faces = colored.faces[:, ::-1]
    colored.export(str(path))
