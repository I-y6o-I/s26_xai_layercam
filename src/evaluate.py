import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from tqdm import tqdm


def collect_predictions(model, dataloader, device):
    """Run inference and collect predictions, targets, and masks."""
    model.eval()
    all_preds, all_targets, all_masks = [], [], []

    with torch.no_grad():
        for images, targets, valid_mask in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            logits = model(images)
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.numpy())
            all_masks.append(valid_mask.numpy())

    return (
        np.concatenate(all_preds),
        np.concatenate(all_targets),
        np.concatenate(all_masks),
    )


def compute_per_label_auc(predictions, targets, masks, label_names):
    """Compute ROC-AUC for each label, respecting the valid mask."""
    results = {}
    for i, name in enumerate(label_names):
        valid = masks[:, i]
        if valid.sum() < 2:
            results[name] = float('nan')
            continue
        y_true = targets[valid, i]
        y_score = predictions[valid, i]
        if len(np.unique(y_true)) < 2:
            results[name] = float('nan')
        else:
            results[name] = roc_auc_score(y_true, y_score)
    return results


def find_optimal_thresholds(predictions, targets, masks, label_names):
    """
    Find per-label decision threshold that maximises Youden's J
    (sensitivity + specificity - 1) on the provided data.

    Returns a dict {label: threshold}.
    """
    thresholds = {}
    for i, name in enumerate(label_names):
        valid = masks[:, i]
        if valid.sum() < 2:
            thresholds[name] = 0.5
            continue
        y_true = targets[valid, i]
        y_score = predictions[valid, i]
        if len(np.unique(y_true)) < 2:
            thresholds[name] = 0.5
            continue
        fpr, tpr, thresh = roc_curve(y_true, y_score)
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        thresholds[name] = float(thresh[best_idx])
    return thresholds


def evaluate_with_thresholds(predictions, targets, masks, label_names, thresholds):
    """Compute F1 per label using the supplied thresholds."""
    f1_scores = {}
    for i, name in enumerate(label_names):
        valid = masks[:, i]
        if valid.sum() < 2:
            f1_scores[name] = float('nan')
            continue
        y_true = targets[valid, i]
        y_score = predictions[valid, i]
        y_pred = (y_score >= thresholds[name]).astype(int)
        f1_scores[name] = f1_score(y_true, y_pred, zero_division=0)
    return f1_scores


def print_evaluation_table(auc_scores, thresholds, f1_scores):
    """Print a formatted summary table."""
    header = f"{'Label':<35} {'AUC':>6}  {'Threshold':>9}  {'F1':>6}"
    print(header)
    print('-' * len(header))

    valid_aucs = [v for v in auc_scores.values() if not np.isnan(v)]
    valid_f1s  = [v for v in f1_scores.values()  if not np.isnan(v)]

    for name in auc_scores:
        auc = auc_scores[name]
        thr = thresholds[name]
        f1  = f1_scores[name]
        auc_str = f'{auc:.4f}' if not np.isnan(auc) else '   N/A'
        f1_str  = f'{f1:.4f}'  if not np.isnan(f1)  else '   N/A'
        print(f'{name:<35} {auc_str:>6}  {thr:>9.4f}  {f1_str:>6}')

    print('-' * len(header))
    print(f"{'Mean (valid labels)':<35} {np.mean(valid_aucs):>6.4f}  {'':>9}  {np.mean(valid_f1s):>6.4f}")
