from collections import defaultdict, Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import warnings
from pyhealth.data import  Patient
from pyhealth.medcode import CrossMap
warnings.filterwarnings('ignore')

from pyhealth.datasets import MIMIC4Dataset
# from pyhealth.tasks import readmission_prediction_mimic4_fn

mapping = CrossMap("ICD10CM", "CCSCM")

def diag_prediction_mimic4_fn(patient: Patient):
    samples = []
    visit_ls = list(patient.visits.keys())
    for i in range(len(visit_ls)):
        visit = patient.visits[visit_ls[i]]
        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]
        cond_ccs = []
        for con in conditions:
            if mapping.map(con):
                cond_ccs.append(mapping.map(con)[0]) 

        if len(cond_ccs) * len(procedures) * len(drugs) == 0:
            continue
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": cond_ccs,
                "procedures": procedures,
                "adm_time" : visit.encounter_time.strftime("%Y-%m-%d %H:%M"),
                "drugs": drugs,
                "cond_hist": conditions,
            }
        )
    # exclude: patients with less than 2 visit
    if len(samples) < 2:
        return []
    # add history
    samples[0]["cond_hist"] = [samples[0]["cond_hist"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs"] = [samples[0]["drugs"]]
    samples[0]["adm_time"] = [samples[0]["adm_time"]]

    for i in range(1, len(samples)):
        samples[i]["drugs"] = samples[i - 1]["drugs"] + [
            samples[i]["drugs"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["cond_hist"] = samples[i - 1]["cond_hist"] + [
            samples[i]["cond_hist"]
        ]
        samples[i]["adm_time"] = samples[i - 1]["adm_time"] + [
            samples[i]["adm_time"]
        ]

    for i in range(len(samples)):
        samples[i]["cond_hist"][i] = []

    return samples

# ============ 1) 工具函数：患者级排序、构造“下一次标签” ============
def sort_samples_within_patient(samples):
    # 按 patient_id 分组，并按 adm_time 排序（你已有 adm_time 字符串，如"2131-06-19 21:32"）
    by_pid = defaultdict(list)
    for s in samples:
        by_pid[s["patient_id"]].append(s)
    for pid in by_pid:
        # 若 adm_time 是字符串，直接排序即可；如有异常可转 datetime 再排
        by_pid[pid] = sorted(by_pid[pid], key=lambda x: x["adm_time"][-1])
    return by_pid

def build_pairs(samples_by_pid, task="current"):
    """
    task="current": 用样本自己的 conditions 做标签（你当前的构造）
    task="next":    严格按论文，特征来自第 t 次，标签用第 t+1 次的 conditions
    返回 pairs: list of (X_sample_dict, y_codes_list)
    """
    pairs = []
    for pid, seq in samples_by_pid.items():
        if task == "current":
            for s in seq:
                pairs.append((s, s["conditions"]))
        elif task == "next":
            # 至少要有 t 和 t+1
            for i in range(len(seq) - 1):
                s_t = seq[i]
                y_next = seq[i + 1]["conditions"]
                pairs.append((s_t, y_next))
        else:
            raise ValueError("task must be 'current' or 'next'")
    return pairs

# ============ 2) 词表：三种模态（ICD 历史、手术码、ATC-3）与标签空间（这里用你的 CCS） ============
def build_vocab_from_pairs(pairs):
    diag_c, proc_c, drug_c, y_c = Counter(), Counter(), Counter(), Counter()
    for s, y in pairs:
        for visit_codes in s["cond_hist"]:   # 历史 ICD 诊断（最后一步为空，已防泄漏）
            diag_c.update(visit_codes)
        for visit_codes in s["procedures"]:  # 每步是一个手术码列表
            proc_c.update(visit_codes)
        for visit_codes in s["drugs"]:       # 每步是一个 ATC3 列表
            drug_c.update(visit_codes)
        y_c.update(y)                        # 标签（你的 CCS）
    def mk_vocab(cnt):
        itos = [c for c, _ in cnt.most_common()]
        stoi = {c:i for i,c in enumerate(itos)}
        return stoi, itos
    return mk_vocab(diag_c), mk_vocab(proc_c), mk_vocab(drug_c), mk_vocab(y_c)

# ============ 3) 向量化：把一条样本变成三模态 multi-hot 并拼接成 X；y 变成多热 ============
def multihot_from_sequence(seq_of_lists, stoi):
    x = torch.zeros(len(stoi), dtype=torch.float32)
    for codes in seq_of_lists:
        for c in codes:
            if c in stoi: x[stoi[c]] = 1.0
    return x

def vectorize_pair(s, y_codes, vocabs, use_current_step=False):
    diag_stoi, proc_stoi, drug_stoi, y_stoi = vocabs
    # 入院预测(admission-time)：不看当前步的 proc/drug；出院预测(discharge-time)可看
    if use_current_step:
        proc_hist = s["procedures"]
        drug_hist = s["drugs"]
    else:
        proc_hist = s["procedures"][:-1] if len(s["procedures"])>0 else []
        drug_hist = s["drugs"][:-1] if len(s["drugs"])>0 else []

    x_diag = multihot_from_sequence(s["cond_hist"], diag_stoi)  # 历史 ICD（当前步已置空）
    x_proc = multihot_from_sequence(proc_hist, proc_stoi)
    x_drug = multihot_from_sequence(drug_hist, drug_stoi)
    X = torch.cat([x_diag, x_proc, x_drug], dim=0)

    y = torch.zeros(len(y_stoi), dtype=torch.float32)
    for c in y_codes:
        if c in y_stoi: y[y_stoi[c]] = 1.0
    return X, y

# ============ 4) 简单的多标签 MLP ============
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, out_dim)  # logits
        )
    def forward(self, x): return self.net(x)

# ============ 5) 数据准备 + 训练/评估 ============
def prepare_XY(pairs, vocabs, use_current_step=False):
    Xs, Ys = [], []
    for s, y_codes in pairs:
        X, y = vectorize_pair(s, y_codes, vocabs, use_current_step=use_current_step)
        Xs.append(X); Ys.append(y)
    return torch.stack(Xs), torch.stack(Ys)

def split_by_patient(pairs, test_size=0.2, val_size=0.1, seed=42):
    # 按 patient_id 分桶，避免同一患者泄漏到不同划分
    pid2pairs = defaultdict(list)
    for s, y in pairs:
        pid2pairs[s["patient_id"]].append((s, y))
    pids = list(pid2pairs.keys())
    tr_pids, te_pids = train_test_split(pids, test_size=test_size, random_state=seed)
    tr_pids, va_pids = train_test_split(tr_pids, test_size=val_size, random_state=seed)
    def collect(pid_list):
        out=[]
        for pid in pid_list: out.extend(pid2pairs[pid])
        return out
    return collect(tr_pids), collect(va_pids), collect(te_pids)

def bce_pos_weight(Y):
    pos = Y.sum(0).clamp_(min=1.0)
    neg = (Y.size(0) - pos).clamp_(min=1.0)
    return (neg / pos)

def topk_acc(y_true, y_prob, k):
    # y_true, y_prob: numpy arrays (N, L)
    N, L = y_true.shape
    topk = np.argpartition(-y_prob, kth=min(k, L-1), axis=1)[:, :k]
    hits = []
    for i in range(N):
        y_set = set(np.where(y_true[i] > 0.5)[0].tolist())
        pred_set = set(topk[i].tolist())
        denom = max(1, min(k, len(y_set)))  # 与论文定义一致
        hits.append(len(y_set & pred_set) / denom)
    return float(np.mean(hits))

@torch.no_grad()
def evaluate(model, X, Y, ks=(15,20,30)):
    model.eval()
    logits = model(X)
    probs = torch.sigmoid(logits).cpu().numpy()
    y_true = Y.cpu().numpy()

    # AUPRC (micro)
    # 注意：average_precision_score 的 micro 需要把二维展平成一维
    ap_micro = average_precision_score(y_true.ravel(), probs.ravel())

    # F1 (micro，阈值0.5；可在验证集调优阈值)
    y_hat = (probs >= 0.5).astype(np.int32)
    f1_micro = f1_score(y_true, y_hat, average="micro", zero_division=0)

    accks = {f"Acc@{k}": topk_acc(y_true, probs, k) for k in ks}

    metrics = {"AUPRC": ap_micro, "F1": f1_micro}
    metrics.update(accks)
    return metrics

def train_mlp_on_samples(samples,
                         task="next",          # "next" 对齐论文；"current" 用你已有标注
                         use_current_step=False, # 入院预测(False) or 出院预测(True)
                         hidden=512, lr=1e-3, wd=1e-5,
                         epochs=10, seed=42):
    # 1) 排序 + 组装 (current/next)
    by_pid = sort_samples_within_patient(samples)
    pairs = build_pairs(by_pid, task=task)

    # 2) 患者级划分
    train_pairs, val_pairs, test_pairs = split_by_patient(pairs, seed=seed)

    # 3) 词表
    (diag_stoi,_), (proc_stoi,_), (drug_stoi,_), (y_stoi, y_itos) = build_vocab_from_pairs(train_pairs)
    vocabs = (diag_stoi, proc_stoi, drug_stoi, y_stoi)

    # 4) 向量化
    Xtr, Ytr = prepare_XY(train_pairs, vocabs, use_current_step=use_current_step)
    Xva, Yva = prepare_XY(val_pairs,   vocabs, use_current_step=use_current_step)
    Xte, Yte = prepare_XY(test_pairs,   vocabs, use_current_step=use_current_step)

    # 5) 模型与损失（多标签）
    model = MLP(Xtr.size(1), hidden=hidden, out_dim=Ytr.size(1))
    pw = bce_pos_weight(Ytr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # 6) 训练
    for ep in range(1, epochs+1):
        model.train()
        logits = model(Xtr)
        loss = criterion(logits, Ytr)
        opt.zero_grad(); loss.backward(); opt.step()

        if ep % 1 == 0:
            mtr = evaluate(model, Xva, Yva)
            print(f"Epoch {ep:02d} | loss={loss.item():.4f} | "
                  f"val AUPRC={mtr['AUPRC']:.4f} F1={mtr['F1']:.4f} "
                  f"Acc@20={mtr['Acc@20']:.4f}")

    # 7) 测试集评估（与论文一致：AUPRC、F1、Acc@15/20/30）
    test_metrics = evaluate(model, Xte, Yte, ks=(15,20,30))
    print("[TEST]", test_metrics)
    return model, vocabs, y_itos, test_metrics

if __name__ == "__main__":
    torch.manual_seed(42); np.random.seed(42)

    print("Loading MIMIC-IV dataset...")
    mimic4_base = MIMIC4Dataset(
        root="/data/yuyu/data/MIMIC_IV/hosp",
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
    )

    mimic4_prediction = mimic4_base.set_task(diag_prediction_mimic4_fn)
    print(f"Total samples: {len(mimic4_prediction.samples)}")

    # 训练一个小模型跑通流程
    model, vocabs, y_itos, test_metrics = train_mlp_on_samples(
        mimic4_prediction.samples,
        task="next",             # 严格“下一次就诊”预测
        use_current_step=False,  # 入院时可用信息（不看当前处方/手术）
        hidden=512,
        lr=1e-3,
        wd=1e-5,
        epochs=10,
        seed=42,
    )
    print("[DONE] Test metrics:", test_metrics)