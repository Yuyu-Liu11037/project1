# ehr_hlm/train_hlm.py
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm
from models.hyperbolic_lm import HyperbolicLM
from data_utils import build_vocabs_from_samples, vectorize_sequences

def split_by_patient(pairs, test_size=0.2, val_size=0.1, seed=42):
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

class SimpleMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for multi-label classification
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128], dropout=0.2):
        super(SimpleMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x, y_multi_hot=None):
        logits = self.network(x)
        
        if y_multi_hot is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y_multi_hot)
            return logits, loss
        else:
            return logits

def precision_at_k(y_true, y_prob, k):
    """
    Visit-level Precision@k (P@k)
    衡量在预测的前 k 个候选 CCS code 中，多少比例与实际下次就诊的 CCS code 匹配。
    代表"患者就诊层面"的准确性（粗粒度）。
    """
    N, L = y_true.shape
    topk = np.argpartition(-y_prob, kth=min(k, L-1), axis=1)[:, :k]
    precisions = []
    for i in range(N):
        y_set = set(np.where(y_true[i] > 0.5)[0].tolist())
        pred_set = set(topk[i].tolist())
        if len(pred_set) > 0:
            precision = len(y_set & pred_set) / len(pred_set)
        else:
            precision = 0.0
        precisions.append(precision)
    return float(np.mean(precisions))

def accuracy_at_k(y_true, y_prob, k):
    """
    Code-level Accuracy@k (Acc@k)
    衡量在预测的前 k 个 CCS code 中，是否包含了真实的 CCS code。
    代表"具体诊断编码层面"的准确性（细粒度）。
    """
    N, L = y_true.shape
    topk = np.argpartition(-y_prob, kth=min(k, L-1), axis=1)[:, :k]
    accuracies = []
    for i in range(N):
        y_set = set(np.where(y_true[i] > 0.5)[0].tolist())
        pred_set = set(topk[i].tolist())
        # 如果真实标签集合为空，则准确率为1（没有错误预测）
        if len(y_set) == 0:
            accuracy = 1.0
        else:
            # 检查是否所有真实标签都在预测的top-k中
            accuracy = 1.0 if y_set.issubset(pred_set) else 0.0
        accuracies.append(accuracy)
    return float(np.mean(accuracies))

@torch.no_grad()
def evaluate(model, batch, Y, ks=(10,20)):
    model.eval()
    logits, _ = model(batch, y_multi_hot=None)
    probs = torch.sigmoid(logits).cpu().numpy()
    y_true = Y.cpu().numpy()

    # micro AUPRC
    from sklearn.metrics import average_precision_score
    ap_micro = average_precision_score(y_true.ravel(), probs.ravel())

    # Visit-level Precision@k 和 Code-level Accuracy@k
    p_at_k = {f"P@{k}": precision_at_k(y_true, probs, k) for k in ks}
    acc_at_k = {f"Acc@{k}": accuracy_at_k(y_true, probs, k) for k in ks}

    m = {"AUPRC": ap_micro}
    m.update(p_at_k)
    m.update(acc_at_k)
    return m

def convert_sequences_to_multihot(batch, vocabs):
    """
    Convert sequence-based batch to multi-hot vectors for MLP
    """
    diag_stoi, proc_stoi, drug_stoi, y_stoi = vocabs
    
    # Calculate dimensions
    diag_dim = len(diag_stoi)
    proc_dim = len(proc_stoi)
    drug_dim = len(drug_stoi)
    total_dim = diag_dim + proc_dim + drug_dim
    
    batch_size = len(batch["diag_seqs"])
    X = torch.zeros(batch_size, total_dim, dtype=torch.float32)
    
    for i in range(batch_size):
        # Process diagnosis sequences
        for seq in batch["diag_seqs"][i]:
            for idx in seq:
                if idx < diag_dim:
                    X[i, idx] = 1.0
        
        # Process procedure sequences
        for seq in batch["proc_seqs"][i]:
            for idx in seq:
                if idx < proc_dim:
                    X[i, diag_dim + idx] = 1.0
        
        # Process drug sequences
        for seq in batch["drug_seqs"][i]:
            for idx in seq:
                if idx < drug_dim:
                    X[i, diag_dim + proc_dim + idx] = 1.0
    
    return X

@torch.no_grad()
def evaluate_mlp(model, X, Y, ks=(10,20)):
    """
    Evaluate MLP model (simplified version of evaluate function)
    """
    model.eval()
    logits = model(X)
    probs = torch.sigmoid(logits).cpu().numpy()
    y_true = Y.cpu().numpy()

    # micro AUPRC
    from sklearn.metrics import average_precision_score
    ap_micro = average_precision_score(y_true.ravel(), probs.ravel())

    # Visit-level Precision@k 和 Code-level Accuracy@k
    p_at_k = {f"P@{k}": precision_at_k(y_true, probs, k) for k in ks}
    acc_at_k = {f"Acc@{k}": accuracy_at_k(y_true, probs, k) for k in ks}

    m = {"AUPRC": ap_micro}
    m.update(p_at_k)
    m.update(acc_at_k)
    return m

def train_model(samples, model_type="hyperbolic", task="next", use_current_step=False, dim=32, lr=1e-3, wd=0.0, epochs=10, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)

    # 1) 构对：current/next
    def sort_samples_within_patient(samples):
        by_pid = defaultdict(list)
        for s in samples:
            by_pid[s["patient_id"]].append(s)
        for pid in by_pid:
            by_pid[pid] = sorted(by_pid[pid], key=lambda x: x["adm_time"][-1])
        return by_pid

    def build_pairs(samples_by_pid, task="current"):
        pairs = []
        for pid, seq in samples_by_pid.items():
            if task == "current":
                for s in seq: pairs.append((s, s["conditions"]))
            else:
                for i in range(len(seq) - 1):
                    s_t = seq[i]; y_next = seq[i + 1]["conditions"]
                    pairs.append((s_t, y_next))
        return pairs

    by_pid = sort_samples_within_patient(samples)
    pairs = build_pairs(by_pid, task=task)

    # 2) 按患者划分
    train_pairs, val_pairs, test_pairs = split_by_patient(pairs, seed=seed)

    # 3) 词表
    (diag_stoi,_), (proc_stoi,_), (drug_stoi,_), (y_stoi,y_itos) = build_vocabs_from_samples(train_pairs)
    vocabs = (diag_stoi, proc_stoi, drug_stoi, y_stoi)

    # 4) 索引化序列（不再 multihot）
    tr_batch, Ytr = vectorize_sequences(train_pairs, vocabs, use_current_step=use_current_step)
    va_batch, Yva = vectorize_sequences(val_pairs,   vocabs, use_current_step=use_current_step)
    te_batch, Yte = vectorize_sequences(test_pairs,  vocabs, use_current_step=use_current_step)

    # 5) 模型选择
    print(f"Creating {model_type} model...")
    if model_type == "hyperbolic":
        model = HyperbolicLM(
            diag_vocab_size=len(diag_stoi),
            proc_vocab_size=len(proc_stoi),
            drug_vocab_size=len(drug_stoi),
            ccs_vocab_size=len(y_stoi),
            dim=dim,
            frechet_steps=15,
            frechet_lr=0.15,
            lambda_rad=1e-3,
            lambda_ont=0.0,
        )
    elif model_type == "mlp":
        # Calculate input dimension for MLP
        input_dim = len(diag_stoi) + len(proc_stoi) + len(drug_stoi)
        model = SimpleMLP(
            input_dim=input_dim,
            output_dim=len(y_stoi),
            hidden_dims=[256, 128],
            dropout=0.2
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'hyperbolic' or 'mlp'")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 6) 训练（全量 batch，demo 版；后续易改 DataLoader）
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    print(f"Starting training...")
    for ep in tqdm(range(1, epochs+1), desc="Training progress", unit="epoch"):
        model.train()
        opt.zero_grad()
        
        # Handle different input formats for different model types
        if model_type == "hyperbolic":
            logits, loss = model({k:v for k,v in tr_batch.items()}, y_multi_hot=Ytr.to(device))
        elif model_type == "mlp":
            # Convert sequences to multi-hot vectors for MLP
            X_tr = convert_sequences_to_multihot(tr_batch, vocabs)
            logits, loss = model(X_tr.to(device), y_multi_hot=Ytr.to(device))
        
        loss.backward(); opt.step()

        # Validation
        if model_type == "hyperbolic":
            mtr = evaluate(model, {k:v for k,v in va_batch.items()}, Yva.to(device))
        elif model_type == "mlp":
            X_va = convert_sequences_to_multihot(va_batch, vocabs)
            mtr = evaluate_mlp(model, X_va.to(device), Yva.to(device))
            
        print(f"Epoch {ep:02d}/{epochs} | loss={loss.item():.4f} | "
              f"val AUPRC={mtr['AUPRC']:.4f} P@10={mtr['P@10']:.4f} Acc@10={mtr['Acc@10']:.4f}")

    # 7) 测试
    print("\nStarting testing...")
    if model_type == "hyperbolic":
        mte = evaluate(model, {k:v for k,v in te_batch.items()}, Yte.to(device))
    elif model_type == "mlp":
        X_te = convert_sequences_to_multihot(te_batch, vocabs)
        mte = evaluate_mlp(model, X_te.to(device), Yte.to(device))
    print("[TEST]", mte)
    return model, vocabs, y_itos, mte

# Backward compatibility function
def train_hlm(samples, task="next", use_current_step=False, dim=32, lr=1e-3, wd=0.0, epochs=10, seed=42):
    """
    Backward compatibility function - calls train_model with hyperbolic model
    """
    return train_model(samples, model_type="hyperbolic", task=task, use_current_step=use_current_step, 
                      dim=dim, lr=lr, wd=wd, epochs=epochs, seed=seed)
