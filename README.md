# u-Protrude3D

**3D cell protrusion segmentation, benchmarking, and volumization for fluorescence microscopy derived meshes.**

`u-Protrude3D` takes a triangulated cell surface mesh and automatically detects, and labels individual protrusions (e.g. blebs, filopodia, microvilli, lamellipodia) by inferring an optimal basal reference surface. It can optionally benchmark predictions against ground-truth labels and volumize surface labels into 3-D voxel volumes. The latter effectively decomposes the cell volume = protrusion volumes + basal volume.    

---

## Installation

```bash
pip install u-Protrude3D
```

> **Solver note.** The default sparse solver is `pardiso` (Intel MKL). If you do not have MKL installed, change `cfg.cmcf.solver` to `'scipy'` (slower but dependency-free).

Pre-compile the numba JIT functions once before your first run to avoid a 10–30 s delay on the first call:

```python
import u_protrude3d
u_protrude3d.warmup()
```

---

## Quick-start

```python
import u_protrude3d as up3d

# 1. Segment protrusions on a single cell mesh
result = up3d.segment_protrusions(
    mesh_path="data/cell01.obj",
    save_dir="output/cell01/",
)
print(result.vertex_labels)   # integer label per vertex (0 = background)

# 2. Benchmark predicted labels against ground truth
bench = up3d.benchmark_segmentation(
    pred_dirs=["output/cell01/", "output/cell02/"],
    gt_dirs=["gt/cell01/",       "gt/cell02/"],
    save_dir="output/benchmark/",
)
print(bench.ap)  # shape (n_cells, n_thresholds)

# 3. Volumize: map surface labels into a 3-D voxel array
vol = up3d.volumize_protrusions(
    mesh_path="data/cell01.obj",
    protrusion_labels=result.vertex_labels,
    tif_path="data/cell01.tif",
    save_dir="output/cell01_vol/",
    cMCF_steps=result.cMCF_steps,
)
print(vol.volume_labels.shape)  # (Z, Y, X)
```

---

## Segment protrusions

### Full example

```python
import u_protrude3d as up3d

cfg = up3d.SegmentConfig()

# Adjust the most commonly tuned parameters
cfg.voxel_size = 0.160          # physical voxel size of the microscope (µm)
cfg.cmcf.n_iters = 25           # more iterations → smoother basal reference
cfg.std_patch.sure_ridge_threshold = 0.03   # more aggressive filopodium detection

result = up3d.segment_protrusions(
    mesh_path="data/cell01.obj",
    save_dir="output/cell01/",
    cfg=cfg,
    tif_path="data/cell01.tif",       # optional: needed for SDF-based binary
    gt_label_path="gt/cell01.tif",    # optional: compute inline AP score
)

# Result fields
result.vertex_labels   # (N,) int — 0 = background, 1..K = protrusion instances
result.cMCF_steps      # (N, 3, T) float — mesh vertex positions at each cMCF step
result.basal_binary    # (N,) bool — vertices classified as basal (non-protrusion)
result.dists           # (N,) float — signed-distance height field
result.H               # (N,) float — mean curvature at each vertex
result.output_paths    # dict of Path objects for every saved file
result.ap              # AP scores (only if gt_label_path was provided)
```

### Chaining into volumize

```python
result = up3d.segment_protrusions("data/cell01.obj", "output/cell01/", cfg=cfg)

vol = up3d.volumize_protrusions(
    mesh_path="data/cell01.obj",
    protrusion_labels=result.vertex_labels,
    tif_path="data/cell01.tif",
    save_dir="output/cell01_vol/",
    cMCF_steps=result.cMCF_steps,
)
```

---

## Benchmark segmentation

### Full example

```python
import glob
import u_protrude3d as up3d

pred_dirs = sorted(glob.glob("output/cell*/"))
gt_dirs   = sorted(glob.glob("gt/cell*/"))

cfg = up3d.BenchmarkConfig()
cfg.figure_format = "png"   # change output format
cfg.figure_dpi    = 300

bench = up3d.benchmark_segmentation(
    pred_dirs=pred_dirs,
    gt_dirs=gt_dirs,
    save_dir="output/benchmark/",
    cfg=cfg,
)

# Result fields
bench.ap               # (n_cells, n_thresholds) average precision
bench.tp               # (n_cells, n_thresholds) true positives
bench.fp               # (n_cells, n_thresholds) false positives
bench.fn               # (n_cells, n_thresholds) false negatives
bench.iou_thresholds   # (n_thresholds,) the IoU cutoffs used
bench.per_cell_names   # list of folder names, one per cell
bench.figures          # dict of matplotlib Figure objects
bench.output_paths     # dict of saved file paths
```

The `.mat` files inside each `pred_dir` must contain the key `protrusion_labels` (per-vertex integer array). Ground-truth `.mat` files must contain the key `protrude_labels`.

---

## Volumize protrusions

### Full example

```python
import numpy as np
import u_protrude3d as up3d

cfg = up3d.VolumeConfig()
cfg.voxel_size          = 0.160   # must match the acquisition voxel size (µm)
cfg.total_shrinkwrap_iters = 120  # more iterations → tighter basal fit
cfg.gvf_iters           = 20      # more GVF smoothing → cleaner force field

vol = up3d.volumize_protrusions(
    mesh_path="data/cell01.obj",
    protrusion_labels=np.load("labels.npy"),   # (N_vertices,) int
    tif_path="data/cell01.tif",
    save_dir="output/cell01_vol/",
    cfg=cfg,
)

# Result fields
vol.volume_labels   # (Z, Y, X) int16 — voxel-space protrusion labels
vol.cell_binary     # (Z, Y, X) bool  — voxel cell mask
vol.basal_binary    # (Z, Y, X) bool  — voxel basal region mask
vol.surface_labels  # (N,) int        — final per-vertex labels after refinement
vol.output_paths    # dict of saved file paths
```

---

## Interactive GUI

Open a point-and-click parameter editor that returns the updated config objects:

```python
import u_protrude3d as up3d

# Launch with defaults
seg_cfg, bench_cfg, vol_cfg = up3d.launch_config_gui()

# Or pre-populate from existing configs
seg_cfg, bench_cfg, vol_cfg = up3d.launch_config_gui(
    cfg_segment=seg_cfg,
    cfg_benchmark=bench_cfg,
    cfg_volume=vol_cfg,
)

# Use the edited configs immediately
result = up3d.segment_protrusions("data/cell01.obj", "output/", cfg=seg_cfg)
```

The GUI window has three tabs (Segment / Benchmark / Volume). Click **Apply** to validate and commit values, **Export JSON** to save a config file, and **Load JSON** to restore one.

---

## Parameter reference

### SegmentConfig — general pipeline settings

| Parameter | Default | Explanation |
|---|---|---|
| `voxel_size` | `0.160` | Physical size of one voxel in micrometers. Must match your microscope acquisition settings. Used when converting the mesh into a signed-distance height field. |
| `n_smooth_scalar_fn_iters` | `50` | How many Laplacian smoothing passes are applied to the height-field scalar before thresholding. More passes reduce noise but may blur sharp protrusion boundaries. |
| `offset_ref_ind` | `0` | Which cMCF iteration is used as the "reference" flat baseline for height calculation. Increase if the very first inflation step produces a noisy reference. |
| `min_size_comps_initial` | `20` | Minimum number of vertices a connected region must have to survive the initial foreground detection. Smaller values keep more detail; larger values discard more noise. |
| `min_size_comps_protrude_patch` | `5` | Same size filter applied to sub-patches inside each candidate protrusion region. |
| `sdf_binary_dilate_ksize` | `2` | Dilation radius (in voxels) applied to the binary cell mask before computing the height field. Increase if protrusion bases are getting clipped. |
| `sdf_binary_erode_ksize` | `1` | Erosion radius after dilation. Balances out the dilation to avoid inflating the mask. |
| `curvature_radius` | `5` | Neighbourhood radius (in mesh hops) used when computing principal curvatures. Larger values smooth the curvature estimate and are better for coarse meshes; smaller values preserve local detail. |
| `n_protrude_colors` | `24` | Size of the colour palette used when saving coloured `.obj` visualisation files. Increase if you have more than 24 protrusions per cell. |
| `random_seed` | `1232` | Seeds the random-colour palette so colours are reproducible across runs. |
| `debug_viz` | `False` | When `True`, saves extra intermediate visualisations. Useful for diagnosing segmentation failures but produces many files. |

---

### cMCFConfig — basal-surface extraction

The pipeline "inflates" the mesh inward using conformal Mean Curvature Flow (cMCF) to find the smooth, protrusion-free basal surface. The height of each vertex above this inflated reference is the main signal used for protrusion detection.

| Parameter | Default | Explanation |
|---|---|---|
| `n_iters` | `20` | Number of inflation steps. More iterations → a smoother, more convex basal reference. Increase for highly irregular or spiky cells; decrease if the reference over-smooths and loses anatomically relevant curvature. |
| `step_size` | `0.25` | How far the mesh moves per inflation step (as a fraction of the Laplacian update). Larger values converge faster but can cause mesh tangling. |
| `extra_smooth` | `False` | Apply a small amount of Laplacian mesh smoothing between each cMCF step. Helps stabilise convergence on meshes with very irregular triangles. |
| `deltaL_smooth` | `5e-5` | Strength of the inter-step Laplacian smoothing (only used when `extra_smooth=True`). Very small by default to avoid distorting the mesh. |
| `deltaL` | `5e-4` | Laplacian step size multiplier for the cMCF update itself. Rarely needs changing; reduce if you see mesh instability. |
| `mollify_factor` | `1e-5` | Regularisation constant for the robust Laplacian, which guards against numerical errors in near-degenerate triangles. Only needs increasing if you see linear-algebra warnings. |
| `solver` | `'pardiso'` | Sparse linear solver used at each cMCF step. `'pardiso'` (Intel MKL) is fastest. Use `'scipy'` if MKL is not available. |

---

### InitialHeightConfig — protrusion binary mask

After computing the height field, these settings control how it is binarised into "protrusion" vs. "background".

| Parameter | Default | Explanation |
|---|---|---|
| `use_auto` | `True` | Automatically select a threshold from the height-field distribution. Set to `False` to use `manual_threshold` instead. |
| `use_mean` | `True` | When `use_auto=True`, threshold at the mean height value. Simple and robust for most cells. |
| `use_otsu` | `False` | When `use_auto=True`, use multi-class Otsu's method instead of the mean. More adaptive but can be fooled by bimodal distributions. Only one of `use_mean` / `use_otsu` should be `True`. |
| `manual_threshold` | `3.0` | Height value (in µm) used as the threshold when `use_auto=False`. Increase to be more conservative (fewer, larger protrusions detected). |
| `use_lower_threshold` | `True` | When using Otsu, pick the lower threshold level (`True`) or the higher one (`False`). The lower level gives a more inclusive binary mask. |
| `prop_iters` | `1` | How many diffusion steps are used to propagate the binary label across the surface. More iterations smooth out ragged boundaries. |
| `prop_rebinarize` | `0.25` | After diffusion, vertices whose propagated probability is above this value are kept as foreground. Lower = more inclusive. |

---

### LargePatchConfig — oversized patch handling

Patches larger than `min_max_area` faces are treated differently because they often represent lamellipodia or other flat, broad structures rather than discrete protrusions.

| Parameter | Default | Explanation |
|---|---|---|
| `min_max_area` | `10000` | Face count above which a candidate patch is classified as "large". Increase if you want the algorithm to treat more patches as discrete protrusions rather than large flat structures. |
| `max_area_thresh_factor` | `0.0` | Adds `factor × median_patch_area` to `min_max_area` as an adaptive component. Useful when patch sizes vary widely across cells. |
| `ridgeness_threshold` | `0.125` | Mean ridgeness above this value → the large patch is classified as ridge-like (filopodium/lamellopodium edge); below → bleb-like (dome). |
| `occ_threshold` | `0.4` | If the fraction of the patch that is "occupied" by the primary binary mask exceeds this value, a simpler single-step segmentation is used; otherwise a two-step approach is applied. |
| `H_segment_method` | `'mean'` | How to threshold mean curvature within a bleb-like large patch. `'mean'` = threshold at the mean value; `'multiotsu'` = use Otsu multi-class thresholding for a more data-driven cut. |
| `H_segment_otsu_n_levels` | `3` | Number of Otsu classes when `H_segment_method='multiotsu'`. More classes = finer curvature bins. |
| `H_segment_otsu_level` | `-1` | Which Otsu threshold to use, with Python negative indexing: `-1` = highest (most conservative), `-2` = second-highest (more inclusive). |
| `second_seg_H_method` | `'multiotsu'` | Method for the second-pass H segmentation (applied to complex patches that the first pass could not cleanly resolve). |
| `second_seg_H_otsu_n_levels` | `2` | Otsu levels for the second-pass segmentation. |
| `second_seg_H_otsu_level` | `-1` | Which Otsu threshold for the second pass. |
| `ridge_segment_method` | `'mean'` | How to threshold ridgeness within a ridge-like large patch. |
| `ridge_otsu_n_levels` | `3` | Otsu levels for ridge segmentation. |
| `ridge_otsu_level` | `-1` | Which Otsu threshold for ridge segmentation. |

---

### StdPatchConfig — standard patch classification

Standard-sized patches (below `LargePatchConfig.min_max_area`) go through a more detailed classification pipeline that separates blebs (dome-shaped), filopodia (thin ridges), and lamellipodia (flat sheets).

#### Protrusion-type thresholds

| Parameter | Default | Explanation |
|---|---|---|
| `sure_noridge_threshold` | `0.02` | Patches whose mean ridgeness is *below* this are confidently classified as bleb-like. Increase to classify more patches as blebs. |
| `sure_ridge_threshold` | `0.025` | Patches whose mean ridgeness is *above* this are confidently classified as ridge-like (filopodium/lamellipodium). Decrease to classify more patches as ridges. Patches between the two thresholds are classified by a secondary shape test. |

#### Mean-curvature (H) segmentation

These parameters control how the height within each patch is binarised using mean curvature.

| Parameter | Default | Explanation |
|---|---|---|
| `H_segment_method` | `'multiotsu'` | `'mean'`: threshold at the patch mean — fast but less adaptive. `'multiotsu'`: data-driven threshold using Otsu classes — better when curvature is bimodal. |
| `H_segment_otsu_n_levels` | `2` | Number of Otsu classes. `2` = one threshold (foreground/background). `3` = two thresholds (add an intermediate class). |
| `H_segment_otsu_level` | `-1` | Which threshold to use. `-1` = highest (most conservative, fewest foreground vertices). |
| `H_segment_erode_steps` | `2` | Number of morphological erosion passes applied to the H binary on the mesh. Erosion shrinks the foreground, removing thin connections and isolated specks. |
| `H_segment_use_local_adaptive` | `True` | Compute the H threshold locally (per-patch adaptive baseline) rather than globally. Recommended: handles illumination or curvature gradients across the cell surface. |
| `H_local_adaptive_smooth_iters` | `50` | Number of Laplacian smoothing steps used to estimate the local H baseline. More iterations → a smoother, more global baseline. |
| `apply_power_H_correct` | `True` | Apply a power-law correction (`H^power_H_correct`) before thresholding. This compresses the high end of the curvature range, which can make thresholding more robust when a few vertices have extreme values. |
| `power_H_correct` | `0.5` | Exponent for the power correction (default `0.5` = square-root). Values < 1 compress the high end; values > 1 amplify it. |

#### Ridgeness segmentation

The same structure as H segmentation but applied to the ridgeness scalar field, which is high along the crests of narrow protrusions.

| Parameter | Default | Explanation |
|---|---|---|
| `ridge_segment_method` | `'multiotsu'` | Threshold method for ridgeness. Same options as `H_segment_method`. |
| `ridge_otsu_n_levels` | `3` | Otsu classes for ridgeness. |
| `ridge_otsu_level` | `-1` | Which Otsu threshold to use. |
| `ridge_segment_erode_steps` | `1` | Erosion passes on the ridge binary. |
| `ridge_use_local_adaptive` | `True` | Use a per-patch adaptive ridgeness baseline. |
| `ridge_local_adaptive_smooth_iters` | `50` | Smoothing iterations for the local ridge baseline. |
| `apply_power_ridge_correct` | `False` | Apply power correction to ridgeness before thresholding. Disabled by default because ridgeness is already well-scaled. |
| `power_ridge_correct` | `0.5` | Exponent if power correction is enabled. |

#### Planarity test (filopodium detection)

| Parameter | Default | Explanation |
|---|---|---|
| `planarity_check_frac` | `1.0` | Maximum fraction of interior vertices allowed to lie *outside* the projected boundary hull for the patch to be considered planar (filopodium-like). `1.0` = any planarity passes; lower values enforce stricter planarity. |
| `planarity_check_dilate_binary` | `2` | Dilation radius (pixels) used when rasterising the boundary for the planarity test. Increase for coarser meshes where boundaries look jagged in projection. |

#### Multi-bleb splitting

If a single large foreground region contains multiple dome-shaped sub-structures, these parameters control whether it is split into separate bleb labels.

| Parameter | Default | Explanation |
|---|---|---|
| `multibleb_min_recovered_frac` | `0.4` | Watershed must recover at least this fraction of the region area for splitting to proceed. Prevents over-splitting of noisy, heterogeneous patches. |
| `multibleb_min_occ_area_fraction` | `0.1` | Each candidate sub-bleb must occupy at least this fraction of the total patch area. Eliminates very small fragments. |
| `multibleb_max_mean_aspect_ratio` | `2.0` | Sub-blebs whose average aspect ratio exceeds this value are rejected (too elongated to be a bleb — likely a filopodium). |
| `multibleb_check_neck_neg_curvature` | `False` | Additionally require negative curvature at the boundary between sub-blebs (a "neck"). Stricter splitting criterion; rarely needed. |
| `multibleb_neck_neg_curvature_thresh` | `-0.05` | Curvature value that counts as a neck when `multibleb_check_neck_neg_curvature=True`. More negative = stricter. |

---

### BenchmarkConfig

| Parameter | Default | Explanation |
|---|---|---|
| `iou_thresholds` | `linspace(0, 1, 21)` | List of IoU cutoffs at which AP/TP/FP/FN are evaluated. The default gives a sweep from 0 to 1 in steps of 0.05. Narrow to `[0.5, 0.75]` for a COCO-style summary. |
| `save_figures` | `True` | Save mean-AP and median-AP curve figures alongside the summary `.mat` file. |
| `figure_dpi` | `600` | Resolution of saved figures in dots-per-inch. Use `300` for drafts, `600` for publication. |
| `figure_format` | `'svg'` | Output format: `'svg'` (lossless, scalable), `'pdf'`, or `'png'`. |

---

### VolumeConfig — volumization

| Parameter | Default | Explanation |
|---|---|---|
| `voxel_size` | `0.160` | Physical voxel size (µm). Must match `SegmentConfig.voxel_size` and your acquisition settings. |
| `gvf_mu` | `0.01` | Diffusion coefficient for the Gradient Vector Field (GVF). GVF is a smoothed version of the signed-distance gradient used to guide the inward mesh advection. Larger `mu` → more diffusion, which helps the force field reach into deep concavities but blurs boundaries. |
| `gvf_iters` | `15` | Number of GVF diffusion iterations. More iterations → a smoother force field that extends further from the original gradient. Increase for cells with deep narrow protrusions. |
| `total_shrinkwrap_iters` | `100` | Total number of steps the basal surface mesh takes as it shrinks inward to fit the cell body. More iterations → tighter basal fit, but also more computation time. |
| `n_punchout_refinements` | `2` | Number of topology-repair passes after shrinkwrapping. Each pass removes mesh self-intersections ("punch-outs"). Rarely needs changing. |
| `decay_rate` | `0.75` | Rate at which the shrinkwrap step size decays each iteration. Values closer to 1 maintain a larger step size for longer (faster but less stable); values closer to 0 slow down more aggressively. |
| `min_lr` | `0.24` | Minimum step size floor for shrinkwrapping. Prevents the step size from decaying to near zero before convergence. |
| `voxelize_dilate_ksize` | `2` | Morphological dilation kernel size applied to the voxelised cell binary. Closes small holes in the voxel mask. |
| `voxelize_erode_ksize` | `2` | Erosion after dilation, restoring the boundary while keeping the interior filled. |
| `voxelize_padsize` | `50` | Zero-padding (in voxels) added around the cell binary before volumizing. Ensures the mesh does not intersect the array boundary. |
| `extra_pad` | `25` | Additional padding for numerical stability in the GVF computation. |
| `label_erosion_rings` | `1` | Number of erosion rings applied to the basal label before inpainting. Prevents basal labels from bleeding into protrusion bases. |
| `min_size_comps` | `20` | Minimum voxel count for a connected component in the volume labels. Removes isolated specks. |
| `n_smooth_iters` | `50` | Laplacian smoothing passes on the volumized label surfaces. Produces cleaner isosurfaces for visualisation. |
| `random_seed` | `1232` | Reproducibility seed for random operations inside the volumization. |
| `n_protrude_colors` | `24` | Colour palette size for volumized `.obj` visualisation files. |

#### Spherical mapping (advanced)

These parameters control the internal quasi-conformal parameterization used to build the GVF reference sphere. They rarely need adjustment.

| Parameter | Default | Explanation |
|---|---|---|
| `spherical_map_delta` | `5e-3` | Step size for the iterative spherical map solver. Reduce if the solver diverges. |
| `spherical_map_min_iter` | `10` | Minimum solver iterations even if convergence is reached earlier. |
| `spherical_map_max_iter` | `25` | Maximum solver iterations. Increase for very complex mesh topologies. |
| `spherical_map_mollify_factor` | `1e-5` | Robust Laplacian regularisation for the spherical solver. Matches `cMCFConfig.mollify_factor`. |

---

## Tips and common adjustments

**Fewer false-positive protrusions.** Increase `std_patch.sure_ridge_threshold` and `std_patch.sure_noridge_threshold` together (e.g. to `0.04` each), and raise `min_size_comps_initial` to filter more small patches.

**Blebs are being split into multiple fragments.** Reduce `cmcf.n_iters` to keep the basal reference closer to the original shape, or lower `std_patch.H_segment_erode_steps` to prevent over-erosion.

**Filopodia are not being detected.** Lower `std_patch.sure_ridge_threshold` (e.g. to `0.015`) so more patches are treated as ridge-like. If they are thin, also check that `std_patch.planarity_check_frac` is set to `1.0` (permissive).

**Pardiso not available.** Set `cfg.cmcf.solver = 'scipy'`. Expect a 3–5× slowdown per cell.

**Consistent colours across cells.** Keep `random_seed` the same value in both `SegmentConfig` and `VolumeConfig`, and do not change `n_protrude_colors` mid-dataset.

**Batch processing a folder of cells.**
```python
import glob
import u_protrude3d as up3d

cfg = up3d.SegmentConfig()
cfg.voxel_size = 0.160

for mesh_path in sorted(glob.glob("data/*.obj")):
    name = Path(mesh_path).stem
    result = up3d.segment_protrusions(
        mesh_path=mesh_path,
        save_dir=f"output/{name}/",
        cfg=cfg,
    )
    print(f"{name}: {result.vertex_labels.max()} protrusions found")
```
