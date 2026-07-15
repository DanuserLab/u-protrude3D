from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


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


@dataclass
class InitialHeightConfig:
    """Initial height field binarization parameters."""
    use_auto: bool = True
    use_mean: bool = True
    use_otsu: bool = False
    manual_threshold: float = 3.0
    use_lower_threshold: bool = True
    prop_iters: int = 1
    prop_rebinarize: float = 0.25


@dataclass
class LargePatchConfig:
    """Parameters for over-sized (lamellipodia-like) patch processing."""
    ridgeness_threshold: float = 0.125
    occ_threshold: float = 0.4
    H_segment_method: Literal['mean', 'multiotsu'] = 'mean'
    H_segment_otsu_n_levels: int = 3
    H_segment_otsu_level: int = -1
    second_seg_H_method: Literal['mean', 'multiotsu'] = 'multiotsu'
    second_seg_H_otsu_n_levels: int = 2
    second_seg_H_otsu_level: int = -1
    ridge_segment_method: Literal['mean', 'multiotsu'] = 'mean'
    ridge_otsu_n_levels: int = 3
    ridge_otsu_level: int = -1
    min_max_area: int = 10000
    max_area_thresh_factor: float = 0.0


@dataclass
class StdPatchConfig:
    """Parameters for standard-size patch classification."""
    sure_noridge_threshold: float = 0.02
    sure_ridge_threshold: float = 0.025
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
    cmcf: cMCFConfig = field(default_factory=cMCFConfig)
    initial_height: InitialHeightConfig = field(default_factory=InitialHeightConfig)
    large_patch: LargePatchConfig = field(default_factory=LargePatchConfig)
    std_patch: StdPatchConfig = field(default_factory=StdPatchConfig)


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
    total_shrinkwrap_iters: int = 100
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
