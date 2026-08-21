from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# Sub-configs for SegmentConfig
# ---------------------------------------------------------------------------

@dataclass
class cMCFConfig:
    """External conformal Mean Curvature Flow parameters."""
    n_iters: int = 20
    step_size: float = 0.25
    extra_smooth: bool = False
    deltaL_smooth: float = 5e-5
    deltaL: float = 5e-4
    mollify_factor: float = 1e-5
    solver: str = 'pardiso'
    # Gaussian smoothing applied to the binary mask before SDF computation.
    # Larger values produce a smoother SDF normal field (less noisy cMCF force).
    sdf_smooth: float = 1.0
    # Gaussian smoothing applied to the SDF gradient field after computation.
    sdf_smooth_gradient: float = 1.0
    # Erosion-only variant: drive the flow only where curvature exceeds an
    # auto-discovered basal level (protrusions erode, basal is preserved).
    erosion_only: bool = False
    # Strictly keep every intermediate shape inside the binary (project any
    # vertex that left the mask back onto its surface each step).
    erosion_project_binary: bool = False


@dataclass
class InitialHeightConfig:
    """Initial height field binarization parameters."""
    use_auto: bool = True
    use_mean: bool = True
    use_otsu: bool = False
    manual_threshold: float = 3.0
    otsu_n_levels: int = 3
    otsu_level: int = -1
    prop_iters: int = 1
    prop_rebinarize: float = 0.25
    # Absolute minimum height a vertex must exceed to be considered protrusive,
    # applied on top of the auto threshold.  Default 0 (no extra filter).
    min_height_threshold: float = 0.0
    # Local-adaptive binarization: subtract a Laplacian-smoothed baseline from
    # the height field before thresholding so that local protrusions are detected
    # relative to their neighbourhood rather than the global mean.
    use_local_adaptive: bool = False
    local_adaptive_smooth_iters: int = 50


@dataclass
class LargePatchConfig:
    """Parameters for over-sized (lamellipodia-like) patch processing."""
    curv_ridge_ratio_threshold: float = 0.5
    # Half-width of the hysteresis dead-zone around the threshold.
    # Patches with ratio in [threshold-offset, threshold+offset] are returned
    # as a single label rather than being forced into either branch.
    curv_ridge_ratio_offset: float = 0.0
    # When True, use the Koenderink Shape Index (SI) instead of curv_ridge_ratio
    # for bleb/ridge branch selection.  SI is scale-invariant: large shallow blebs
    # and small tight blebs both score near +1, cylinders/ridges near +0.5.
    # The same curv_ridge_ratio_threshold is reused as the SI cut-off.
    use_shape_index_classifier: bool = False
    # When True, the ridge branch returns the whole patch as one label instead
    # of splitting it into individual ridge CCs.  The bleb branch is unaffected.
    ridge_no_split: bool = False
    # Multi-ridge guard: after ridge segmentation, any CC whose mean Shape Index
    # exceeds this value is considered dome-like (misclassified bleb) and is
    # reclassified through the bleb branch.  Set to 1.1 to disable.
    ridge_reclass_si_threshold: float = 1.1
    occ_threshold: float = 0.4
    H_segment_method: Literal['mean', 'multiotsu'] = 'mean'
    H_segment_otsu_n_levels: int = 3
    H_segment_otsu_level: int = -1
    apply_power_H_correct: bool = False
    power_H_correct: float = 0.5
    H_segment_use_local_adaptive: bool = True
    H_local_adaptive_smooth_iters: int = 50
    H_segment_erode_steps: int = 0
    ridge_segment_method: Literal['mean', 'multiotsu'] = 'mean'
    ridge_otsu_n_levels: int = 3
    ridge_otsu_level: int = -1
    min_max_area: int = 10000
    max_area_thresh_factor: float = 0.0


@dataclass
class InvariantConfig:
    """Parameters for :func:`segment_protrusions_invariant` SI-based thresholding."""
    si_segment_method: Literal['mean', 'multiotsu'] = 'multiotsu'
    si_segment_otsu_n_levels: int = 2
    si_segment_otsu_level: int = -1
    n_diffusion_iters: int = 5
    # Local-adaptive SI thresholding within each patch: subtract a Laplacian-smoothed
    # SI baseline before thresholding so dome tips are detected relative to local background.
    si_use_local_adaptive: bool = False
    si_local_adaptive_smooth_iters: int = 50
    # Seeding strategy: None → SI binarisation; 'height'/'shape_index' → geodesic watershed
    watershed_seeding: Optional[Literal['height', 'shape_index']] = None
    ws_min_peak_dist: int = 3              # min 1-ring hops between kept peaks
    ws_smooth_iters: int = 10             # seed-scalar smoothing before NMS; 0 = skip
    ws_peak_eps: float = 0.0              # plateau tolerance for peak finding; 0 = strict
    ws_saddle_merge: bool = True          # applies to ALL seeding paths
    ws_saddle_criterion: Literal['height', 'shape_index', 'adaptive'] = 'height'
    ws_saddle_depth_threshold: float = 0.2  # 'height'/'adaptive' elongated path: fractional valley depth (0–1)
    ws_saddle_si_threshold: float = 0.5    # 'shape_index'/'adaptive' compact path: merge if mean(SI[boundary]) > this
    ws_saddle_ar_threshold: float = 0.5   # 'adaptive': mean SI per label > this → compact (bleb); ≤ → elongated (ridge)


@dataclass
class StdPatchConfig:
    """Parameters for standard-size patch classification."""
    curv_ridge_ratio_threshold: float = 0.5
    # Half-width of the hysteresis dead-zone around the threshold (same
    # semantics as LargePatchConfig.curv_ridge_ratio_offset).
    curv_ridge_ratio_offset: float = 0.0
    # Shape Index classifier — same semantics as LargePatchConfig.
    use_shape_index_classifier: bool = False
    # When True, the ridge branch returns the whole patch as one label.
    # The bleb branch (multi-bleb check and splitting) is unaffected.
    ridge_no_split: bool = False
    # Multi-ridge guard: ridge CCs with mean SI above this are reclassified as
    # blebs.  Set to 1.1 to disable.
    ridge_reclass_si_threshold: float = 1.1
    # Ridge-fragment merge: after ridge CC finding, merge adjacent CCs whose
    # shared boundary height exceeds this fraction of the lower CC mean height.
    # Same height-valley logic as the global merge_saddle_factor but applied
    # locally within each patch before label-spreading.  Set to 0.0 to disable.
    ridge_merge_saddle_factor: float = 0.6
    H_segment_method: Literal['mean', 'multiotsu'] = 'multiotsu'
    H_segment_otsu_n_levels: int = 2
    H_segment_otsu_level: int = -1
    H_segment_erode_steps: int = 2
    H_segment_use_local_adaptive: bool = True
    H_local_adaptive_smooth_iters: int = 50
    apply_power_H_correct: bool = True
    power_H_correct: float = 0.5
    ridge_segment_method: Literal['mean', 'multiotsu'] = 'multiotsu'
    ridge_otsu_n_levels: int = 3
    ridge_otsu_level: int = -1
    ridge_segment_erode_steps: int = 1
    ridge_use_local_adaptive: bool = True
    ridge_local_adaptive_smooth_iters: int = 50
    apply_power_ridge_correct: bool = False
    power_ridge_correct: float = 0.5
    planarity_check_frac: float = 1.0
    planarity_check_dilate_binary: int = 2
    multibleb_min_recovered_frac: float = 0.4
    multibleb_min_occ_area_fraction: float = 0.1
    multibleb_max_mean_aspect_ratio: float = 2.0
    multibleb_check_neck_neg_curvature: bool = False
    multibleb_neck_neg_curvature_thresh: float = -0.05
    # H-saddle valley check: if the mean |H| on the shared boundary between any
    # adjacent pair of H-CCs exceeds this fraction of the lower CC's mean |H|,
    # the two CCs are topographically connected (one dome) and the split is vetoed.
    # Set to 0.0 to disable.
    multibleb_valley_factor: float = 0.6
    # Debug/diagnostic: bypass ALL multi-bleb guard conditions and split every
    # bleb-classified patch into its raw H-CC labels whenever > 1 CC exists.
    # Use this to check whether splitting is geometrically possible before
    # deciding which guard is wrongly blocking it.
    multibleb_force_split: bool = False


# ---------------------------------------------------------------------------
# Top-level configs for the three public functions
# ---------------------------------------------------------------------------

@dataclass
class SegmentConfig:
    """Full parameter set for :func:`segment_protrusions`."""
    voxel_size: float = 0.160
    n_smooth_scalar_fn_iters: int = 50
    offset_ref_ind: int = 0
    min_size_comps_initial: int = 20
    min_size_comps_protrude_patch: int = 5
    sdf_binary_dilate_ksize: int = 2
    sdf_binary_erode_ksize: int = 1
    n_protrude_colors: int = 24
    random_seed: int = 1232
    debug_viz: bool = False
    curvature_radius: int = 5
    # Remeshing strategy for step 1.  When True, attempts incremental isotropic
    # remeshing (uniform triangles, better curvature accuracy) before falling
    # back to predecimate+resample if it raises an exception.
    use_isotropic_remesh: bool = False
    isotropic_remesh_edge_length_factor: float = 1.0
    # Greedy post-merge: after all patches are labelled, adjacent instance pairs
    # that increase the area-weighted signal score when merged are greedily
    # combined.  Set merge_adjacent_blebs=False to skip.
    # Neck filter for the uncovered-face fill step: uncovered connected components
    # whose mean height is below this fraction of the mean height of the already-
    # labelled protrusion vertices are treated as necks/connectors and skipped.
    # Set to 0.0 to disable (label everything basal_binary covers).
    uncovered_neck_height_frac: float = 0.5
    merge_adjacent_blebs: bool = True
    merge_max_iters: int = 100
    # Height-based post-merge: adjacent instance pairs where the mean HEIGHT on
    # their shared boundary exceeds this fraction of the lower instance's mean
    # height are considered same-dome fragments and merged.  Height is used
    # rather than curvature because |H| varies strongly tip→flank→rim on a
    # single bleb, making curvature comparisons unreliable.  A genuine
    # inter-protrusion saddle sits in a height valley; a within-dome fragment
    # boundary stays elevated.  Set to 0.0 to disable.  Typical range: 0.5–0.9.
    merge_saddle_factor: float = 0.7
    cmcf: cMCFConfig = field(default_factory=cMCFConfig)
    initial_height: InitialHeightConfig = field(default_factory=InitialHeightConfig)
    large_patch: LargePatchConfig = field(default_factory=LargePatchConfig)
    std_patch: StdPatchConfig = field(default_factory=StdPatchConfig)
    invariant: InvariantConfig = field(default_factory=InvariantConfig)


@dataclass
class BenchmarkConfig:
    """Parameter set for :func:`benchmark_segmentation`."""
    iou_thresholds: list = field(
        default_factory=lambda: list(__import__('numpy').linspace(0.0, 1, 21))
    )
    save_figures: bool = True
    figure_dpi: int = 600
    figure_format: str = 'svg'


@dataclass
class VolumeConfig:
    """Parameter set for :func:`volumize_protrusions`."""
    voxel_size: float = 0.160
    spherical_map_delta: float = 5e-3
    spherical_map_min_iter: int = 10
    spherical_map_max_iter: int = 25
    spherical_map_mollify_factor: float = 1e-5
    gvf_mu: float = 0.01
    gvf_iters: int = 15
    gvf_vfc_sigma: float = 0.0   # VFC sigma for GVF path; 0 = disabled (pure GVF)
    gvf_vfc_blend: float = 1.0   # blend weight: 0 = pure VFC, 1 = pure GVF
    total_shrinkwrap_iters: int = 100
    # Mesh-based shrinkwrap (alternative to GVF — no voxelization required)
    use_mesh_based_shrinkwrap: bool = False
    mesh_sw_enable_hole_masking: bool = False        # zero forces over open holes to form a meniscus
    mesh_sw_hole_sq_dist_threshold: Optional[float] = None  # explicit sq-dist hole threshold (mesh units²); None → auto from edge length
    mesh_sw_hole_edge_length_factor: float = 3.0    # auto threshold = (factor × mean_edge_len)²; larger = more aggressive masking
    mesh_sw_anchor_factor: float = 3.0
    mesh_sw_concavity_boost: float = 1.0
    mesh_sw_force_sigma: float = 5.0
    mesh_sw_convergence_sq_dist: float = 0.5
    mesh_sw_topology_repair_alpha_frac: float = 0.05
    mesh_sw_min_size: int = 10000           # target vertex count for internal remeshing
    mesh_sw_genus0_alpha_auto: bool = False # auto-select alpha for genus-0 topology repair
    mesh_sw_genus0_alpha_frac: float = 0.2  # alpha = frac × mean mesh extent (when alpha_auto=False)
    mesh_sw_vfc_sigma: float = 0.0        # VFC Gaussian sigma; 0 = disabled (default off)
    mesh_sw_vfc_blend: float = 0.5        # 0 = pure direct force, 1 = pure VFC
    mesh_sw_smooth_iters: int = 0         # post-force Laplacian smooth iters (0 = off)
    mesh_sw_balloon_factor: float = 0.0   # outward balloon force (0 = off)
    # How to pick the final shrinkwrap mesh from the iteration sequence.
    # 'last'     — use the final iteration (default; most tightly fitted).
    # 'min_loss' — use the iteration minimising 0.5*chamfer + 0.5*|Gauss curvature|,
    #              both normalised to [0,1].  More conservative; avoids over-tightening.
    shrinkwrap_iter_select: Literal['last', 'min_loss'] = 'last'
    n_punchout_refinements: int = 2
    decay_rate: float = 0.75
    min_lr: float = 0.24
    voxelize_dilate_ksize: int = 2
    voxelize_erode_ksize: int = 2
    voxelize_padsize: int = 50
    extra_pad: int = 25
    label_erosion_rings: int = 1
    min_size_comps: int = 20
    n_smooth_iters: int = 50
    random_seed: int = 1232
    n_protrude_colors: int = 24
