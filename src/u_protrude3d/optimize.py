"""Biologically-motivated parameter optimisation for :func:`segment_protrusions`.

The quality score rewards segmentations where each instance has strong
type-specific signal concentration (mean curvature for blebs, ridgeness for
ridges/ruffles, height for all) while penalising fragmentation into many small
instances.
"""
from __future__ import annotations

import copy
import itertools
import os
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import pandas as pd
    from .segment import SegmentResult
    from .config import SegmentConfig


# ---------------------------------------------------------------------------
# Quality score
# ---------------------------------------------------------------------------

def score_segmentation(
    result: "SegmentResult",
    cfg: "SegmentConfig",
    w_height: float = 0.4,
    fragmentation_beta: float = 0.5,
    small_area_thresh: float | None = None,
) -> dict:
    """Compute a biological quality score for a segmentation result.

    Parameters
    ----------
    result : SegmentResult
    cfg : SegmentConfig
        Used for ``curv_ridge_ratio_threshold`` and ``min_max_area``.
    w_height : float
        Weight given to the height signal vs the type-specific shape signal.
        ``1 - w_height`` goes to curvature (bleb) or ridgeness (ridge/ruffle).
    fragmentation_beta : float
        Exponent on the instance count in the fragmentation penalty.  ``0.5``
        means doubling the count costs √2; ``1.0`` means it costs 2×.
    small_area_thresh : float or None
        Instances with area below this are counted as fragments.  Defaults to
        ``cfg.large_patch.min_max_area / 4``.

    Returns
    -------
    dict with keys ``n_instances``, ``n_small``, ``signal_quality``, ``frag``, ``Q``.
    """
    import igl

    mesh_V = result.mesh_V
    mesh_F = result.mesh_F
    vertex_labels = result.vertex_labels
    H_mean = result.H_mean
    ridgeness = result.ridgeness
    height = result.height

    labels = np.setdiff1d(np.unique(vertex_labels), 0)
    n = len(labels)
    if n == 0:
        return dict(n_instances=0, n_small=0, signal_quality=0.0, frag=1.0, Q=0.0)

    # Per-vertex areas (from face doublearea, averaged onto vertices)
    face_areas = igl.doublearea(mesh_V, mesh_F) / 2.0
    vertex_areas = igl.average_onto_vertices(
        mesh_V, mesh_F,
        np.vstack([face_areas] * 3).T,
    )[:, 0]

    # Global normalisation constants (99th pct over protrusive vertices)
    pro_mask = vertex_labels > 0
    eps = 1e-8
    h99 = float(np.percentile(height[pro_mask], 99)) + eps
    H99 = float(np.percentile(np.abs(H_mean[pro_mask]), 99)) + eps
    r99 = float(np.percentile(ridgeness[pro_mask], 99)) + eps

    from .segment import _instance_signal

    thr = cfg.std_patch.curv_ridge_ratio_threshold
    if small_area_thresh is None:
        # Use 10 % of the average protrusive-vertex area as the "small" cutoff.
        # cfg.large_patch.min_max_area is in face count, not physical area, so
        # dividing it by 4 would be meaningless in mesh-unit space.
        avg_instance_area = float(np.sum(vertex_areas[pro_mask])) / max(n, 1)
        small_area_thresh = 0.10 * avg_instance_area

    area_per = []
    signal_per = []

    for lbl in labels:
        mask = vertex_labels == lbl
        s_k = _instance_signal(mask, H_mean, ridgeness, height, cfg, h99, H99, r99,
                               w_height=w_height)
        area_k = float(np.sum(vertex_areas[mask]))
        area_per.append(area_k)
        signal_per.append(s_k)

    areas = np.array(area_per)
    signals = np.array(signal_per)
    total_area = float(np.sum(areas))

    signal_quality = float(np.sum(areas * signals) / total_area)
    n_small = int(np.sum(areas < small_area_thresh))
    frag = float(n ** fragmentation_beta * (1.0 + n_small / max(n, 1)))
    Q = signal_quality / frag

    return dict(
        n_instances=n,
        n_small=n_small,
        signal_quality=signal_quality,
        frag=frag,
        Q=Q,
    )


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

def _set_nested(cfg: "SegmentConfig", dotted_key: str, value) -> None:
    """Set a (possibly nested) config field via a dotted path like 'cmcf.n_iters'."""
    parts = dotted_key.split(".")
    obj = cfg
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def optimize_segmentation(
    mesh_path: str | os.PathLike,
    save_dir: str | os.PathLike,
    param_grid: dict | None = None,
    base_cfg: "SegmentConfig | None" = None,
    w_height: float = 0.4,
    fragmentation_beta: float = 0.5,
    small_area_thresh: float | None = None,
    n_jobs: int = 1,
    tif_path: str | os.PathLike | None = None,
) -> "tuple[SegmentConfig, pd.DataFrame]":
    """Sweep ``param_grid`` over :func:`segment_protrusions` and return the
    parameter combination that maximises the biological quality score.

    Parameters
    ----------
    mesh_path : path-like
    save_dir : path-like
        Root directory; each combination is saved to ``save_dir/sweep/<idx>/``.
    param_grid : dict or None
        Keys are dotted ``SegmentConfig`` paths (e.g. ``'cmcf.n_iters'``),
        values are lists of values to try.  Default sweeps
        ``n_smooth_scalar_fn_iters`` × ``cmcf.n_iters``.
    base_cfg : SegmentConfig or None
        Starting config; all non-swept fields keep their values.
    w_height : float
        Weight given to the height signal in :func:`score_segmentation`.
    fragmentation_beta : float
        Fragmentation exponent in :func:`score_segmentation`.
    small_area_thresh : float or None
        Passed through to :func:`score_segmentation`.
    n_jobs : int
        Parallel workers (``joblib``).  ``1`` = sequential.
    tif_path : path-like or None
        Optional TIFF volume for GT-free height refinement, forwarded to
        :func:`segment_protrusions`.

    Returns
    -------
    best_cfg : SegmentConfig
    sweep_df : pandas.DataFrame
        Columns: all param names + ``n_instances``, ``n_small``,
        ``signal_quality``, ``frag``, ``Q``, ``run_save_dir``.
    """
    import pandas as pd
    from .segment import segment_protrusions
    from .config import SegmentConfig as _SC

    if param_grid is None:
        param_grid = {
            "n_smooth_scalar_fn_iters": [20, 40, 60, 80, 100],
            "cmcf.n_iters": [15, 20, 25],
        }

    if base_cfg is None:
        base_cfg = _SC()

    save_dir = Path(save_dir)
    sweep_root = save_dir / "sweep"
    sweep_root.mkdir(parents=True, exist_ok=True)

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    def _run_one(idx, combo):
        cfg = copy.deepcopy(base_cfg)
        for k, v in zip(keys, combo):
            _set_nested(cfg, k, v)

        run_dir = sweep_root / f"{idx:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        result = segment_protrusions(
            mesh_path, run_dir, cfg=cfg, tif_path=tif_path,
        )
        scores = score_segmentation(
            result, cfg,
            w_height=w_height,
            fragmentation_beta=fragmentation_beta,
            small_area_thresh=small_area_thresh,
        )
        row = {k: v for k, v in zip(keys, combo)}
        row.update(scores)
        row["run_save_dir"] = str(run_dir)
        return row, cfg

    if n_jobs == 1 or len(combinations) == 1:
        rows_cfgs = [_run_one(i, combo) for i, combo in enumerate(combinations)]
    else:
        try:
            from joblib import Parallel, delayed
            rows_cfgs = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_run_one)(i, combo)
                for i, combo in enumerate(combinations)
            )
        except ImportError:
            rows_cfgs = [_run_one(i, combo) for i, combo in enumerate(combinations)]

    rows = [r for r, _ in rows_cfgs]
    cfgs = [c for _, c in rows_cfgs]

    df = pd.DataFrame(rows)
    best_idx = int(df["Q"].idxmax())
    best_cfg = cfgs[best_idx]

    _plot_sweep(df, keys, save_dir / "sweep" / "sweep_scores.svg")

    return best_cfg, df


# ---------------------------------------------------------------------------
# Sweep visualisation
# ---------------------------------------------------------------------------

def _plot_sweep(df: "pd.DataFrame", param_keys: list[str], save_path: Path) -> None:
    n_params = len(param_keys)
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params + 1, 4), squeeze=False)

    for ax, key in zip(axes[0], param_keys):
        grouped = df.groupby(key)["Q"].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            grouped[key], grouped["mean"], yerr=grouped["std"].fillna(0),
            marker="o", linewidth=1.5, capsize=4, color="#1f77b4",
        )
        best_val = df.loc[df["Q"].idxmax(), key]
        ax.axvline(best_val, color="#d62728", linewidth=1.2, linestyle="--",
                   label=f"best = {best_val}")
        ax.set_xlabel(key, fontsize=10)
        ax.set_ylabel("Quality score Q", fontsize=10)
        ax.set_title(key.split(".")[-1], fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    fig.suptitle("Segmentation parameter sweep", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
