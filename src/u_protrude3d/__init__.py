"""u-Protrude3D – 3D cell protrusion segmentation, benchmarking, and volumization."""

from .segment import segment_protrusions, segment_protrusions_invariant, SegmentResult, InvariantResult
from .benchmark import benchmark_segmentation, BenchmarkResult
from .volumize import volumize_protrusions, VolumeResult
from .config import SegmentConfig, BenchmarkConfig, VolumeConfig
from .gui import launch_config_gui
from .optimize import optimize_segmentation, score_segmentation

__version__ = "0.1.0"

__all__ = [
    "segment_protrusions",
    "segment_protrusions_invariant",
    "InvariantResult",
    "benchmark_segmentation",
    "volumize_protrusions",
    "SegmentConfig",
    "BenchmarkConfig",
    "VolumeConfig",
    "SegmentResult",
    "BenchmarkResult",
    "VolumeResult",
    "launch_config_gui",
    "optimize_segmentation",
    "score_segmentation",
]


def warmup():
    """Pre-compile numba JIT functions to avoid first-call latency."""
    import numpy as np
    from ._utils.metrics import _label_overlap

    dummy = np.zeros(10, dtype=np.int64)
    _label_overlap(dummy, dummy)
