"""
Evaluation metrics module
Contains various evaluation metric calculation functions
"""
import torch
import numpy as np


def bce_pos_weight(Y):
    """Calculate positive sample weights for BCE loss"""
    pos = Y.sum(0).clamp_(min=1.0)
    neg = (Y.size(0) - pos).clamp_(min=1.0)
    return (neg / pos)


def _topk_indices(y_prob: np.ndarray, k: int):
    """Return top-k prediction indices for each sample (N, k)"""
    N, L = y_prob.shape
    k_eff = min(k, L)
    return np.argpartition(-y_prob, kth=k_eff-1, axis=1)[:, :k_eff]


def precision_at_k_visit(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    """
    For each visit t: P@k(t) = hits_t / min(k, |Y_t|), then average over samples
    """
    topk = _topk_indices(y_prob, k)  # shape: [N, k]
    N = y_true.shape[0]
    precs = []

    for i in range(N):
        true_idx = np.where(y_true[i] > 0.5)[0]
        m = true_idx.size
        if m == 0:
            precs.append(0.0) 
            continue
        hit = len(set(topk[i].tolist()) & set(true_idx.tolist()))
        denom = float(min(k, m))
        precs.append(hit / denom)

    return float(np.mean(precs)) if len(precs) > 0 else 0.0


def accuracy_at_k_code(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    """
    Code-level Acc@k (fine-grained): Treat each "true code occurrence" as an instance,
    count whether it is hit by top-k; equivalent to micro Recall@k
    = (total true codes hit across all samples) / (total true codes across all samples)
    """
    topk = _topk_indices(y_prob, k)
    hits_total = 0
    true_total = int(y_true.sum())
    
    if true_total == 0:
        return 0.0

    N = y_true.shape[0]
    for i in range(N):
        true_idx = set(np.where(y_true[i] > 0.5)[0].tolist())
        pred_idx = set(topk[i].tolist())
        hits_total += len(true_idx & pred_idx)
    
    return hits_total / float(true_total)


def recall_at_k_micro(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    """Same as accuracy_at_k_code, exposed separately for easy alignment with paper appendix"""
    return accuracy_at_k_code(y_true, y_prob, k)


@torch.no_grad()
def evaluate(model, X, Y, ks=(10, 20, 30)):
    """
    Return metrics consistent with the paper:
    - Visit-level P@k
    - Code-level Acc@k (= micro Recall@k)
    Also return Recall@k as alias for easy comparison with appendix
    """
    model.eval()
    logits = model(X)
    probs = torch.sigmoid(logits).cpu().numpy()
    y_true = Y.cpu().numpy()

    metrics = {}
    for k in ks:
        p_at_k = precision_at_k_visit(y_true, probs, k)
        acc_at_k = accuracy_at_k_code(y_true, probs, k)
        r_at_k = recall_at_k_micro(y_true, probs, k)
        metrics[f"P@{k}"] = p_at_k
        metrics[f"Acc@{k}"] = acc_at_k
        metrics[f"Recall@{k}"] = r_at_k  # Same as Acc@k (micro average)
    
    return metrics

