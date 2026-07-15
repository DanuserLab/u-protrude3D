from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt

from .config import BenchmarkConfig
from ._utils.metrics import average_precision, _relabel_sequential_vertex_labels


@dataclass
class BenchmarkResult:
    """Return value of :func:`benchmark_segmentation`."""
    ap: np.ndarray
    tp: np.ndarray
    fp: np.ndarray
    fn: np.ndarray
    iou_thresholds: np.ndarray
    per_cell_names: list
    figures: dict
    output_paths: dict


def benchmark_segmentation(
    pred_dirs: list,
    gt_dirs: Optional[list] = None,
    pred_mat_filename: str = 'instance_protrusion_segmentation_stats.mat',
    gt_mat_filename: str = 'protrusion_labels_GT_surface.mat',
    save_dir: Optional[str | os.PathLike] = None,
    cfg: Optional[BenchmarkConfig] = None,
) -> BenchmarkResult:
    """Compute AP/TP/FP/FN vs IoU threshold for one or more segmented cells.

    Parameters
    ----------
    pred_dirs : list of str or Path
        One folder per cell containing *pred_mat_filename*.
    gt_dirs : list of str or Path or None
        Matching ground-truth folders.  If None, assumed same as *pred_dirs*.
    pred_mat_filename : str
        Name of the .mat file with predicted labels inside each pred folder.
        The key ``'protrusion_labels'`` is read from this file.
    gt_mat_filename : str
        Name of the .mat file with ground-truth labels inside each gt folder.
        The key ``'protrude_labels'`` is read from this file.
    save_dir : str, Path or None
        Where figures and summary .mat are written.  Skipped if None.
    cfg : BenchmarkConfig or None
        Algorithm parameters.  Uses defaults when None.

    Returns
    -------
    BenchmarkResult
        Fields: ap, tp, fp, fn (n_cells × n_thresholds), iou_thresholds,
        per_cell_names, figures, output_paths.
    """
    if cfg is None:
        cfg = BenchmarkConfig()

    if gt_dirs is None:
        gt_dirs = pred_dirs

    if len(pred_dirs) != len(gt_dirs):
        raise ValueError('pred_dirs and gt_dirs must have the same length')

    iou_thresholds = np.asarray(cfg.iou_thresholds)
    all_ap, all_tp, all_fp, all_fn = [], [], [], []
    per_cell_names = []

    for pred_dir, gt_dir in zip(pred_dirs, gt_dirs):
        pred_dir = Path(pred_dir)
        gt_dir = Path(gt_dir)

        pred_mat = spio.loadmat(str(pred_dir / pred_mat_filename))
        pred_labels = np.squeeze(pred_mat['protrusion_labels'])

        gt_mat = spio.loadmat(str(gt_dir / gt_mat_filename))
        gt_labels = np.squeeze(gt_mat['protrude_labels'])
        
        print(gt_labels)
        
        gt_ = _relabel_sequential_vertex_labels(gt_labels)
        pred_ = _relabel_sequential_vertex_labels(pred_labels)
        
        print(gt_)
        print(iou_thresholds)
        
        ap_, tp_, fp_, fn_ = average_precision(
            gt_[None,:], pred_[None,:], threshold=list(iou_thresholds)
        )
        
        print(ap_)

        all_ap.append(ap_)
        all_tp.append(tp_)
        all_fp.append(fp_)
        all_fn.append(fn_)
        
        print(pred_dir.name)
        # per_cell_names.append(pred_dir.name)

    all_ap = np.vstack(all_ap)
    all_tp = np.vstack(all_tp)
    all_fp = np.vstack(all_fp)
    all_fn = np.vstack(all_fn)
    
    print(all_ap.shape)
    print(all_ap)

    # figures = {}
    # output_paths = {}

    # if save_dir is not None:
    #     save_dir = Path(save_dir)
    #     save_dir.mkdir(parents=True, exist_ok=True)

    #     # Mean AP curve
    #     fig_mean, ax = plt.subplots(figsize=(5, 5))
    #     ax.plot(iou_thresholds, np.nanmean(all_ap, axis=0), 'ko-')
    #     ax.set_xlim([0.5, 1.0])
    #     ax.set_ylim([0, 1])
    #     ax.set_ylabel('Average Precision', fontsize=18)
    #     ax.set_xlabel('IoU', fontsize=18)
    #     ax.tick_params(right=True, length=10)
    #     mean_path = save_dir / f'AP_metric_curve.{cfg.figure_format}'
    #     fig_mean.savefig(str(mean_path), dpi=cfg.figure_dpi, bbox_inches='tight')
    #     figures['mean_ap'] = fig_mean
    #     output_paths['mean_ap_fig'] = mean_path

    #     # Median AP curve
    #     fig_med, ax2 = plt.subplots(figsize=(5, 5))
    #     ax2.plot(iou_thresholds, np.nanmedian(all_ap, axis=0), 'ko-')
    #     ax2.set_xlim([0.5, 1.0])
    #     ax2.set_ylim([0, 1])
    #     ax2.set_ylabel('Average Precision (median)', fontsize=18)
    #     ax2.set_xlabel('IoU', fontsize=18)
    #     ax2.tick_params(right=True, length=10)
    #     med_path = save_dir / f'AP_metric_median_curve.{cfg.figure_format}'
    #     fig_med.savefig(str(med_path), dpi=cfg.figure_dpi, bbox_inches='tight')
    #     figures['median_ap'] = fig_med
    #     output_paths['median_ap_fig'] = med_path

    #     # Summary .mat
    #     mat_path = save_dir / 'AP_metrics_protrusion-detection-surface.mat'
    #     spio.savemat(
    #         str(mat_path),
    #         {
    #             'all_ap': all_ap,
    #             'all_tp': all_tp,
    #             'all_fp': all_fp,
    #             'all_fn': all_fn,
    #             'iou_thresholds': iou_thresholds,
    #             'per_cell_names': per_cell_names,
    #         },
    #     )
    #     output_paths['summary_mat'] = mat_path

    #     plt.close('all')

    # return BenchmarkResult(
    #     ap=all_ap,
    #     tp=all_tp,
    #     fp=all_fp,
    #     fn=all_fn,
    #     iou_thresholds=iou_thresholds,
    #     per_cell_names=per_cell_names,
    #     figures=figures,
    #     output_paths=output_paths,
    # )
