"""u-Protrude3D – 3D cell protrusion segmentation, benchmarking, and volumization."""

from .segment import segment_protrusions, SegmentResult
from .benchmark import benchmark_segmentation, BenchmarkResult
from .volumize import volumize_protrusions, VolumeResult
from .config import SegmentConfig, BenchmarkConfig, VolumeConfig
from .gui import launch_config_gui

__version__ = "0.1.0"

__all__ = [
    "segment_protrusions",
    "benchmark_segmentation",
    "volumize_protrusions",
    "SegmentConfig",
    "BenchmarkConfig",
    "VolumeConfig",
    "SegmentResult",
    "BenchmarkResult",
    "VolumeResult",
    "launch_config_gui",
]


def warmup():
    """Pre-compile numba JIT functions to avoid first-call latency."""
    import numpy as np
    from ._utils.metrics import _label_overlap

    dummy = np.zeros(10, dtype=np.int64)
    _label_overlap(dummy, dummy)
