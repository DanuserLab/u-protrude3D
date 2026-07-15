import numpy as np
from numba import jit
from scipy.optimize import linear_sum_assignment
import skimage.segmentation as sksegmentation


# @jit(nopython=True)
def _label_overlap(x, y):
    """Pixel-level overlap matrix between two integer label arrays.

    Parameters
    ----------
    x, y : int ndarray – label arrays (0 = background, 1..N = instances)

    Returns
    -------
    overlap : (x.max()+1, y.max()+1) uint array
    """
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint64)
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap


def _intersection_over_union(masks_true, masks_pred):
    """IoU matrix for all (true, pred) label pairs.

    Returns
    -------
    iou : (n_true+1, n_pred+1) float array  (row/col 0 = background)
    """
    masks_true = np.asarray(masks_true)
    masks_pred = np.asarray(masks_pred)
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou


def _true_positive(iou, th):
    """Number of true positives at IoU threshold *th* via Hungarian matching."""
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(np.float32) - iou / (2 * n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    return int(match_ok.sum())


def _relabel_sequential_vertex_labels(labels):
    """Relabel integer label array so IDs are contiguous starting at 1."""
    return sksegmentation.relabel_sequential(labels)[0]


def average_precision(masks_true, masks_pred, threshold=(0.5, 0.75, 0.9)):
    """Average precision AP = TP / (TP + FP + FN) at one or more IoU thresholds.

    Parameters
    ----------
    masks_true, masks_pred : list of int ndarray or single ndarray
        Ground-truth and predicted label maps (0 = background).
    threshold : list of float or ndarray
        IoU thresholds to evaluate at.

    Returns
    -------
    ap, tp, fp, fn : each shape (n_images, n_thresholds) or (n_thresholds,)
        when a single image is passed.
    """
    not_list = not isinstance(masks_true, list)
    if not_list:
        masks_true = [masks_true]
        masks_pred = [masks_pred]
    if not isinstance(threshold, (list, np.ndarray)):
        threshold = [threshold]

    if len(masks_true) != len(masks_pred):
        raise ValueError('average_precision requires len(masks_true)==len(masks_pred)')

    ap = np.zeros((len(masks_true), len(threshold)), np.float32)
    tp = np.zeros_like(ap)
    fp = np.zeros_like(ap)
    fn = np.zeros_like(ap)
    n_true = np.array([np.max(m) for m in masks_true])
    n_pred = np.array([np.max(m) for m in masks_pred])

    for n in range(len(masks_true)):
        if n_pred[n] > 0:
            iou = _intersection_over_union(masks_true[n], masks_pred[n])[1:, 1:]
            for k, th in enumerate(threshold):
                tp[n, k] = _true_positive(iou, th)
        fp[n] = n_pred[n] - tp[n]
        fn[n] = n_true[n] - tp[n]
        ap[n] = tp[n] / (tp[n] + fp[n] + fn[n])

    if not_list:
        return ap[0], tp[0], fp[0], fn[0]
    return ap, tp, fp, fn


def mask_ious(masks_true, masks_pred):
    """Return best-matched IoU per true instance and their matched pred indices."""
    iou = _intersection_over_union(masks_true, masks_pred)[1:, 1:]
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= 0.5).astype(np.float32) - iou / (2 * n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    iout = np.zeros(masks_true.max())
    iout[true_ind] = iou[true_ind, pred_ind]
    preds = np.zeros(masks_true.max(), int)
    preds[true_ind] = pred_ind + 1
    return iout, preds


def aggregated_jaccard_index(masks_true, masks_pred):
    """AJI = total intersection of matched pairs / total union."""
    aji = np.zeros(len(masks_true))
    for n in range(len(masks_true)):
        iout, preds = mask_ious(masks_true[n], masks_pred[n])
        inds = np.arange(0, masks_true[n].max(), 1, int)
        overlap = _label_overlap(masks_true[n], masks_pred[n])
        union = np.logical_or(masks_true[n] > 0, masks_pred[n] > 0).sum()
        overlap = overlap[inds[preds > 0] + 1, preds[preds > 0].astype(np.int32)]
        aji[n] = overlap.sum() / union
    return aji
