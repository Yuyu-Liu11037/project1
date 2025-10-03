"""
评估指标模块
包含各种评估指标的计算函数
"""
import torch
import numpy as np


def bce_pos_weight(Y):
    """计算BCE损失的正样本权重"""
    pos = Y.sum(0).clamp_(min=1.0)
    neg = (Y.size(0) - pos).clamp_(min=1.0)
    return (neg / pos)


def _topk_indices(y_prob: np.ndarray, k: int):
    """返回每个样本的 top-k 预测下标 (N, k)"""
    N, L = y_prob.shape
    k_eff = min(k, L)
    return np.argpartition(-y_prob, kth=k_eff-1, axis=1)[:, :k_eff]


def precision_at_k_visit(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    """
    Visit-level P@k（粗粒度）：对每个样本 i，计算 |TopK_i ∩ True_i| / k，最后对样本取平均
    """
    topk = _topk_indices(y_prob, k)
    N = y_true.shape[0]
    precs = []
    
    for i in range(N):
        true_idx = np.where(y_true[i] > 0.5)[0]
        if true_idx.size == 0:
            # 若该次就诊没有真标签，则该样本对 P@k 贡献 0（常见做法）
            precs.append(0.0)
            continue
        hit = len(set(topk[i].tolist()) & set(true_idx.tolist()))
        precs.append(hit / float(k))
    
    return float(np.mean(precs)) if len(precs) > 0 else 0.0


def accuracy_at_k_code(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    """
    Code-level Acc@k（细粒度）：把每个"真代码出现"当作一次实例，
    统计它是否被 top-k 命中；等价于 micro Recall@k
    = (所有样本命中的真代码数) / (所有样本真代码总数)
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
    """与 accuracy_at_k_code 完全相同，单独暴露便于和论文附录对齐显示"""
    return accuracy_at_k_code(y_true, y_prob, k)


@torch.no_grad()
def evaluate(model, X, Y, ks=(10, 20)):
    """
    返回与论文一致的指标：
    - Visit-level P@k
    - Code-level Acc@k(= micro Recall@k)
    同时也返回 Recall@k 作为别名，便于对照附录
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
        metrics[f"Recall@{k}"] = r_at_k  # 与 Acc@k 相同（微平均）
    
    return metrics

