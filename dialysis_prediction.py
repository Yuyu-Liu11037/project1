"""
AKI to CKD progression prediction main program
"""
import argparse
from collections import defaultdict
from functools import partial
import json
import random
from typing import Counter
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import warnings
import pickle
import geoopt
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')


class TransformerModel(nn.Module):
    def __init__(self, x_vocab_size, hidden, out_dim, *,
                 diag_size, proc_size,
                 num_heads=8, num_layers=3, p=0.3,
                 diag_itos=None, c=1.0, max_diag_len=None):
        super().__init__()
        self.diag_itos = diag_itos
        self.emb_diag  = nn.Embedding(diag_size + 1, hidden, padding_idx=0)
        # self.emb_proc  = nn.Embedding(proc_size + 1, embed_dim, padding_idx=0)
        # self.emb_third = nn.Embedding(x_vocab_size - (diag_size + proc_size) + 1, embed_dim, padding_idx=0)

        enc_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=num_heads, dim_feedforward=hidden * 4, dropout=p, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden))
        self.output_projection = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(p)

    def encode(self, x_diag, x_proc, x_drug):
        device = x_diag.device
        B = x_diag.shape[0]
        
        e_diag = self.emb_diag(x_diag)   # (B, L_diag, E)
        # e_proc = self.emb_proc(x_proc)   # (B, L_proc, E)
        # e_third = self.emb_third(x_drug) # (B, L_drug, E)
        
        diag_mask = (x_diag != 0)   # (B, L_diag)
        # proc_mask = (x_proc != 0)   # (B, L_proc)
        # drug_mask = (x_drug != 0)   # (B, L_drug)
        # padding_mask = torch.cat([diag_mask, proc_mask, drug_mask], dim=1)  # (B, L_total)
    
        padding_mask_expanded = diag_mask.unsqueeze(-1)  # (B, L_total, 1)
        e_diag = e_diag.masked_fill(~padding_mask_expanded, 0.)

        cls_tokens = self.cls_token.expand(B, 1, -1)        # (B, 1, H)
        x_seq = torch.cat([cls_tokens, e_diag], dim=1) # (B, L_total+1, H)
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
        key_padding_mask = torch.cat([cls_mask, diag_mask], dim=1)
        src_key_padding_mask = ~key_padding_mask  # True = mask out

        x_seq = self.transformer(x_seq, src_key_padding_mask=src_key_padding_mask)
        x_cls = self.dropout(x_seq[:, 0, :])  # (B, H)
        return x_cls

    def forward(self, x_diag, x_proc, x_drug, return_binary=False):
        x_cls = self.encode(x_diag, x_proc, x_drug)  # (B, H)
        logits = self.output_projection(x_cls)  # (B, out_dim)
        
        if return_binary:
            # Apply sigmoid and threshold to get binary predictions (0 or 1)
            probs = torch.sigmoid(logits)
            binary_preds = (probs > 0.5).float()
            return binary_preds
        else:
            # Return logits for training (maintains gradients)
            return logits


class CodeTokenizer:
    """Tokenizer for medical codes (diagnosis, procedure, drug)"""
    
    def __init__(self, stoi, offset=0):
        """
        Args:
            stoi: String to index mapping dictionary
            offset: Starting index for this tokenizer (to reserve 0 for padding)
        """
        self.stoi = stoi
        self.itos = {i: code for code, i in stoi.items()}  # Index to string mapping
        self.offset = offset
        self.vocab_size = len(stoi)
    
    def encode(self, seq_of_lists):
        """
        Convert code sequences to indices
        
        Args:
            seq_of_lists: List of lists of codes (e.g., [[code1, code2], [code3]])
        
        Returns:
            List of indices
        """
        indices = []
        for codes in seq_of_lists:
            for c in codes:
                if c not in self.stoi:
                    # print(c)
                    continue
                indices.append(self.stoi[c] + self.offset)
        return indices
    
    def decode(self, indices, remove_padding=True):
        """
        Convert indices back to code strings
        
        Args:
            indices: List of indices
            remove_padding: Whether to remove 0 (padding) indices
        
        Returns:
            List of code strings
        """
        codes = []
        for idx in indices:
            if remove_padding and idx == 0:
                continue
            idx_without_offset = idx - self.offset
            if idx_without_offset in self.itos:
                codes.append(self.itos[idx_without_offset])
        return codes


def indices_from_sequence(seq_of_lists, stoi, offset=0):
    """Convert sequence to list of indices (with optional offset to reserve 0 for padding)
    
    DEPRECATED: Use CodeTokenizer.encode() instead
    """
    indices = []
    for codes in seq_of_lists:
        for c in codes:
            if c in stoi: 
                indices.append(stoi[c] + offset)  # Add offset to reserve 0 for padding
    return indices


def create_tokenizers(vocabs):
    """
    Create tokenizers for diagnosis, procedure, drug codes, and labels
    
    Args:
        vocabs: Tuple of (diag_stoi, proc_stoi, drug_stoi, ccs_stoi)
        - diag_stoi: ICD codes for cond_hist (input)
        - ccs_stoi: CCS codes for labels (output)
    
    Returns:
        Tuple of (diag_tokenizer, proc_tokenizer, drug_tokenizer, ccs_tokenizer)
    """
    diag_stoi, proc_stoi, drug_stoi, ccs_stoi = vocabs
    
    # Reserve 0 for padding, so start at 1
    # With separate embeddings, each type uses local indexing (0-indexed in its own vocab)
    diag_tokenizer = CodeTokenizer(diag_stoi, offset=1)  # ICD codes for input
    ccs_tokenizer = CodeTokenizer(ccs_stoi, offset=1)    # CCS codes for output
    
    # Each code type now uses local indexing
    proc_tokenizer = CodeTokenizer(proc_stoi, offset=1)
    drug_tokenizer = CodeTokenizer(drug_stoi, offset=1)
    
    return diag_tokenizer, proc_tokenizer, drug_tokenizer, ccs_tokenizer


def vectorize_pair(s, y_label, vocabs, use_current_step=False):
    diag_stoi, proc_stoi, drug_stoi, ccs_stoi = vocabs
    
    # Admission prediction: don't look at current step's proc/drug; discharge prediction can look
    cond_hist = s["cond_hist"]
    if use_current_step:
        proc_hist = s["procedures"]
        drug_hist = s["drugs"]
    else:
        proc_hist = s["procedures"][:-1] if len(s["procedures"])>0 else []
        drug_hist = s["drugs"][:-1] if len(s["drugs"])>0 else []

    # Create tokenizers (CCS for labels, ICD for input)
    diag_tokenizer, proc_tokenizer, drug_tokenizer, ccs_tokenizer = create_tokenizers(vocabs)
    
    # Tokenize each type of code
    x_diag_indices = diag_tokenizer.encode(cond_hist) 
    x_proc_indices = proc_tokenizer.encode(proc_hist)
    x_drug_indices = drug_tokenizer.encode(drug_hist)
    
    # Return three separate tensors (use 0 as padding)
    X_diag = torch.tensor(x_diag_indices, dtype=torch.long) if len(x_diag_indices) > 0 else torch.tensor([0], dtype=torch.long)
    X_proc = torch.tensor(x_proc_indices, dtype=torch.long) if len(x_proc_indices) > 0 else torch.tensor([0], dtype=torch.long)
    X_drug = torch.tensor(x_drug_indices, dtype=torch.long) if len(x_drug_indices) > 0 else torch.tensor([0], dtype=torch.long)

    y = torch.tensor([y_label], dtype=torch.float)
    
    return (X_diag, X_proc, X_drug), y


def prepare_XY(pairs, vocabs, use_current_step=False):
    """Prepare training data X and Y as variable-length sequences"""
    Xs_diag, Xs_proc, Xs_drug, Ys = [], [], [], []
    for s, y_label in pairs:
        (X_diag, X_proc, X_drug), y = vectorize_pair(s, y_label, vocabs, use_current_step=use_current_step)
        Xs_diag.append(X_diag)
        Xs_proc.append(X_proc)
        Xs_drug.append(X_drug)
        Ys.append(y)
    return (Xs_diag, Xs_proc, Xs_drug), Ys


def split_by_patient(pairs, test_size=0.2, val_size=0.1, seed=42):
    """Split dataset by patient ID to avoid data leakage"""
    pid2pairs = defaultdict(list)
    for s, y in pairs:
        pid2pairs[s["patient_id"]].append((s, y))
    
    pids = list(pid2pairs.keys())
    tr_pids, te_pids = train_test_split(pids, test_size=test_size, random_state=seed)
    tr_pids, va_pids = train_test_split(tr_pids, test_size=val_size, random_state=seed)
    
    def collect(pid_list):
        out = []
        for pid in pid_list: 
            out.extend(pid2pairs[pid])
        return out
    
    return collect(tr_pids), collect(va_pids), collect(te_pids)


def sort_samples_within_patient(samples):
    """Group by patient ID and sort by admission time"""
    by_pid = defaultdict(list)
    for s in samples:
        by_pid[s["patient_id"]].append(s)
    
    for pid in by_pid:
        # If adm_time is string, sort directly; if needed, convert to datetime for sorting
        by_pid[pid] = sorted(by_pid[pid], key=lambda x: x["adm_time"][-1])
    
    return by_pid


def build_pairs(samples_by_pid):
    pairs = []
    for pid, seq in samples_by_pid.items():
        if len(seq) >= 2:
            early_aki_sample = seq[0]  # Early AKI phase
            followup_sample = seq[1]   # Follow-up phase
                
            # Use early AKI features to predict CKD progression
            ckd_progression_label = followup_sample["ckd_progression"]
            pairs.append((early_aki_sample, ckd_progression_label))
    return pairs


def build_vocab_from_pairs(pairs):
    diag_c, proc_c, drug_c, ccs_c = Counter(), Counter(), Counter(), Counter()
    with open('/data/yuyu/project1/cond_hist_codes.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            codes = line.split()
            diag_c.update(codes)
    
    for s, y in pairs:
        for visit_codes in s["procedures"]:  # Each step is a procedure code list
            proc_c.update(visit_codes)
        for visit_codes in s["drugs"]:       # Each step is an ATC3 list
            drug_c.update(visit_codes) 
    
    def mk_vocab(cnt):
        itos = [c for c, _ in cnt.most_common()]
        stoi = {c:i for i,c in enumerate(itos)}
        return stoi, itos
    
    # Separate vocabularies: ICD for input, CCS for output
    diag_stoi, diag_itos = mk_vocab(diag_c)  # ICD codes for cond_hist
    proc_stoi, proc_itos = mk_vocab(proc_c)
    drug_stoi, drug_itos = mk_vocab(drug_c)
    
    ccs_stoi, ccs_itos = {}, []
    
    return (diag_stoi, diag_itos), (proc_stoi, proc_itos), (drug_stoi, drug_itos), (ccs_stoi, ccs_itos)


def load_preprocessed_data(data_path):
    data_path = Path(data_path)
    
    # Load the main CSV file with patient data
    csv_file = data_path / "full_preprocessing_sample_977.csv"
    
    print(f"Loading data from {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Load mapping files
    with open(data_path / "diag_code_to_idx.json", 'r') as f:
        diag_code_to_idx = json.load(f)
    with open(data_path / "med_code_to_idx.json", 'r') as f:
        med_code_to_idx = json.load(f)
    with open(data_path / "proc_code_to_idx.json", 'r') as f:
        proc_code_to_idx = json.load(f)
    
    samples = []
    aki_patients = 0
    ckd_progression_patients = 0
    
    for _, row in df.iterrows():
        patient_id = row['patient_id']
        
        # Parse diagnosis codes
        diag_codes = []
        if pd.notna(row['diagnosis_codes']) and row['diagnosis_codes']:
            diag_codes = [code.strip() for code in str(row['diagnosis_codes']).split(';')]
        
        # Parse medication codes  
        med_codes = []
        if pd.notna(row['medication_codes']) and row['medication_codes']:
            med_codes = [code.strip() for code in str(row['medication_codes']).split(';')]
        
        # Parse procedure codes
        proc_codes = []
        if pd.notna(row['procedure_codes']) and row['procedure_codes']:
            proc_codes = [code.strip() for code in str(row['procedure_codes']).split(';')]
        
        # Skip if any code list is empty
        if len(diag_codes) == 0 or len(med_codes) == 0 or len(proc_codes) == 0:
            continue
        
        # Check for AKI codes (N17.x series)
        has_aki = any(code.startswith('N17.') for code in diag_codes)
        
        # Only include patients with AKI for AKI to CKD progression prediction
        if not has_aki:
            continue
            
        aki_patients += 1
        
        # Check for CKD progression (N18.x series indicating chronic kidney disease)
        has_ckd = any(code.startswith('N18.') for code in diag_codes)
        
        # Binary label: 1 if AKI progressed to CKD, 0 if not
        ckd_progression_label = 1 if has_ckd else 0
        if has_ckd:
            ckd_progression_patients += 1
        
        # For AKI to CKD progression prediction, we'll create two visits per patient:
        # Visit 1: Early AKI phase (input features) 
        # Visit 2: Follow-up phase with CKD progression outcome
        
        # Split diagnosis codes into early AKI and follow-up
        mid_point = len(diag_codes) // 2 if len(diag_codes) > 1 else 1
        early_diag = diag_codes[:mid_point]
        followup_diag = diag_codes[mid_point:] if len(diag_codes) > 1 else diag_codes
        
        # Split other codes similarly
        mid_med = len(med_codes) // 2 if len(med_codes) > 1 else 1
        early_med = med_codes[:mid_med]
        followup_med = med_codes[mid_med:] if len(med_codes) > 1 else med_codes
        
        mid_proc = len(proc_codes) // 2 if len(proc_codes) > 1 else 1
        early_proc = proc_codes[:mid_proc]
        followup_proc = proc_codes[mid_proc:] if len(proc_codes) > 1 else proc_codes
        
        # Create early AKI visit (input features)
        early_sample = {
            "visit_id": f"{patient_id}_early_aki",
            "patient_id": patient_id,
            "conditions": early_diag,
            "procedures": early_proc,
            "drugs": early_med,
            "cond_hist": early_diag,
            "adm_time": "2023-01-01",
            "ckd_progression": 0,  # Early phase - no progression yet
            "lab_egfr_min": row.get('lab_egfr_min', 0.0),
            "lab_creatinine_max": row.get('lab_creatinine_max', 0.0),
            "lab_bun_max": row.get('lab_bun_max', 0.0),
            "lab_potassium_max": row.get('lab_potassium_max', 0.0),
        }
        
        # Create follow-up visit (target for progression prediction)
        followup_sample = {
            "visit_id": f"{patient_id}_followup",
            "patient_id": patient_id,
            "conditions": followup_diag,
            "procedures": followup_proc,
            "drugs": followup_med,
            "cond_hist": followup_diag,
            "adm_time": "2023-06-01",
            "ckd_progression": ckd_progression_label,  # Binary outcome: 1 if progressed to CKD, 0 if not
            "lab_egfr_min": row.get('lab_egfr_min', 0.0),
            "lab_creatinine_max": row.get('lab_creatinine_max', 0.0),
            "lab_bun_max": row.get('lab_bun_max', 0.0),
            "lab_potassium_max": row.get('lab_potassium_max', 0.0),
        }
        
        samples.extend([early_sample, followup_sample])
    
    print(f"Loaded {len(samples)} samples from {aki_patients} AKI patients")
    print(f"AKI to CKD progression: {ckd_progression_patients}/{aki_patients} patients ({ckd_progression_patients/aki_patients*100:.1f}%)")
    return samples


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


class VariableLengthDataset(Dataset):
    """Dataset for variable-length tensors with three separate code types"""
    def __init__(self, X_list_diag, X_list_proc, X_list_drug, Y_list):
        self.X_list_diag = X_list_diag
        self.X_list_proc = X_list_proc
        self.X_list_drug = X_list_drug
        self.Y_list = Y_list
    
    def __len__(self):
        return len(self.Y_list)
    
    def __getitem__(self, idx):
        return (self.X_list_diag[idx], self.X_list_proc[idx], self.X_list_drug[idx]), self.Y_list[idx]


def collate_fn(batch, max_diag_len=None, max_proc_len=None, max_drug_len=None):
    """Custom collate function to pad variable-length sequences for three code types"""
    X_batch, Y_batch = zip(*batch)
    
    # Unpack three types of X
    X_diag_batch = [x[0] for x in X_batch]
    X_proc_batch = [x[1] for x in X_batch]
    X_drug_batch = [x[2] for x in X_batch]
    
    # Pad each type separately to their respective max lengths
    if max_diag_len is None:
        max_diag_len = max(len(x) for x in X_diag_batch) if X_diag_batch else 0
    if max_proc_len is None:
        max_proc_len = max(len(x) for x in X_proc_batch) if X_proc_batch else 0
    if max_drug_len is None:
        max_drug_len = max(len(x) for x in X_drug_batch) if X_drug_batch else 0
    
    # Pad each type
    X_diag_padded = torch.nn.utils.rnn.pad_sequence(X_diag_batch, batch_first=True, padding_value=0)
    X_diag_padded = X_diag_padded[:, :max_diag_len]  # Truncate if necessary
    
    X_proc_padded = torch.nn.utils.rnn.pad_sequence(X_proc_batch, batch_first=True, padding_value=0)
    X_proc_padded = X_proc_padded[:, :max_proc_len]
    
    X_drug_padded = torch.nn.utils.rnn.pad_sequence(X_drug_batch, batch_first=True, padding_value=0)
    X_drug_padded = X_drug_padded[:, :max_drug_len]
    
    # Handle different Y types (binary vs multi-label)
    if len(Y_batch[0].shape) == 1 and Y_batch[0].shape[0] == 1:
        # Binary classification: Y_batch contains tensors of shape (1,)
        Y_padded = torch.stack(Y_batch)  # Shape: (batch_size, 1)
    else:
        # Multi-label classification: Y_batch contains multi-hot vectors
        Y_padded = torch.stack(Y_batch)
    
    return (X_diag_padded, X_proc_padded, X_drug_padded), Y_padded


def train_model_on_samples(samples,
                           model_type="transformer", 
                           use_current_step=False, # Admission prediction(False) or discharge prediction(True)
                           hidden=512, lr=1e-3, wd=1e-5,
                           epochs=10, seed=42, 
                           batch_size=32,         # Batch size for training
                           patience=10,           # Number of epochs to wait before stopping
                           min_delta=0.001,      # Minimum change to qualify as improvement
                           monitor_metric='Acc@10', # Metric to monitor for early stopping
                           **model_kwargs):
    # 1) Sort and assemble
    by_pid = sort_samples_within_patient(samples) 
    pairs = build_pairs(by_pid)  
    pairs = [(s, y_codes) for s, y_codes in pairs if s.get('cond_hist', []) and len([x for x in s['cond_hist'] if x]) > 0]

    # 2) Patient-level split
    train_pairs, val_pairs, test_pairs = split_by_patient(pairs, seed=seed)

    # 3) Vocabulary
    (diag_stoi, diag_itos), (proc_stoi, proc_itos), (drug_stoi, drug_itos), (ccs_stoi, ccs_itos) = build_vocab_from_pairs(pairs) # diag_stoi={code: index}, diag_itos=[code1, code2, ...]
    vocabs = (diag_stoi, proc_stoi, drug_stoi, ccs_stoi)

    # 4) Vectorization
    (Xtr_diag, Xtr_proc, Xtr_drug), Ytr = prepare_XY(train_pairs, vocabs, use_current_step=use_current_step)
    (Xva_diag, Xva_proc, Xva_drug), Yva = prepare_XY(val_pairs,   vocabs, use_current_step=use_current_step)
    (Xte_diag, Xte_proc, Xte_drug), Yte = prepare_XY(test_pairs,   vocabs, use_current_step=use_current_step)

    # Calculate max sequence lengths for each code type
    max_diag_len = max(max(len(x) for x in Xtr_diag), max(len(x) for x in Xva_diag), max(len(x) for x in Xte_diag))
    max_proc_len = max(max(len(x) for x in Xtr_proc), max(len(x) for x in Xva_proc), max(len(x) for x in Xte_proc))
    max_drug_len = max(max(len(x) for x in Xtr_drug), max(len(x) for x in Xva_drug), max(len(x) for x in Xte_drug))
    
    print(f"Max sequence lengths - Diag: {max_diag_len}, Proc: {max_proc_len}, Drug: {max_drug_len}")

    # 5) Create DataLoaders for batch training with custom collate function
    train_dataset = VariableLengthDataset(Xtr_diag, Xtr_proc, Xtr_drug, Ytr)
    val_dataset = VariableLengthDataset(Xva_diag, Xva_proc, Xva_drug, Yva)
    test_dataset = VariableLengthDataset(Xte_diag, Xte_proc, Xte_drug, Yte)
    
    # Create collate function with max lengths
    collate_fn_with_max = partial(collate_fn, max_diag_len=max_diag_len, max_proc_len=max_proc_len, max_drug_len=max_drug_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_with_max)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_with_max)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_with_max)

    # 6) Model and loss (binary classification for AKI to CKD progression)
    # Calculate vocabulary sizes for embedding layers
    diag_stoi, proc_stoi, drug_stoi, ccs_stoi = vocabs
    x_vocab_size = len(diag_stoi) + len(proc_stoi) + len(drug_stoi) + 1  # +1 for padding
    y_vocab_size = 1  # Binary classification
    
    device = torch.device('cuda')
    print(f"Using device: {device}")
    print(f"X vocab size: {x_vocab_size}, Y vocab size: {y_vocab_size}")
    
    model_kwargs = {**model_kwargs, 
                             'diag_size': len(diag_stoi),
                             'proc_size': len(proc_stoi),
                             'diag_itos': diag_itos,
                             'max_diag_len': max_diag_len,
                            }
    model = TransformerModel(x_vocab_size, args.hidden, y_vocab_size, **model_kwargs)
    model = model.to(device) 
    opt = geoopt.optim.RiemannianAdam(model.parameters(), lr=lr, weight_decay=wd)

    # 7) Training loop with batches and early stopping
    best_metric = -float('inf')
    patience_counter = 0
    best_model_state = None
        
    print(f"Training for {epochs} epochs")
    for ep in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
            
        # Training phase
        for batch_X, batch_Y in train_loader:
            batch_X_diag, batch_X_proc, batch_X_drug = batch_X
            batch_X_diag = batch_X_diag.to(device) 
            batch_X_proc = batch_X_proc.to(device)
            batch_X_drug = batch_X_drug.to(device)
            batch_Y = batch_Y.to(device)
            
            logits = model(batch_X_diag, batch_X_proc, batch_X_drug, return_binary=False) 
            loss = nn.functional.binary_cross_entropy_with_logits(logits, batch_Y)
                
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        # Validation phase
        if ep % 1 == 0:
            val_metrics = evaluate_binary_classification(model, val_loader, device=device)
            current_metric = val_metrics['accuracy']  # Use accuracy for binary classification
            print(f"Epoch {ep:02d} | avg_loss={avg_loss:.4f} | val Acc={val_metrics['accuracy']:.4f} AUC={val_metrics['auc']:.4f} F1={val_metrics['f1']:.4f}")
            print(f"Logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}, std={logits.std().item():.4f}")
            
            if current_metric > best_metric + min_delta:
                best_metric = current_metric
                patience_counter = 0
                # Save best model state
                best_model_state = model.state_dict().copy()
                print(f"  → New best metric: {best_metric:.4f}")
            else:
                patience_counter += 1
                    
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
                print(f"Restoring best model from epoch {ep - patience_counter}")
                model.load_state_dict(best_model_state)
                break

    test_metrics = evaluate_binary_classification(model, test_loader, device=device)
    print("[TEST] Binary Classification Metrics:", test_metrics)
    
    return model, vocabs, ccs_itos, test_metrics


def evaluate_batched(model, data_loader, ks=(10, 20, 30), device='cuda'):
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_Y in data_loader:
            batch_X_diag, batch_X_proc, batch_X_drug = batch_X
            # For SVM, keep on CPU; for other models, move to device
            if device != 'cpu':
                batch_X_diag = batch_X_diag.to(device)
                batch_X_proc = batch_X_proc.to(device)
                batch_X_drug = batch_X_drug.to(device)
                batch_Y = batch_Y.to(device)
            else:
                batch_Y = batch_Y
            
            # Handle models that return tuple vs single value
            try:
                # For LTransformerDecoder, only pass batch_X_diag
                # The model will automatically handle padding mask
                logits = model(batch_X_diag, batch_X_proc, batch_X_drug)
                if isinstance(logits, tuple):
                    logits = logits[0]
            except RuntimeError as e:
                if "fitted" in str(e):
                    # Model not fitted yet, skip this batch
                    continue
                raise
            
            if device != 'cpu':
                logits = logits.cpu()
            all_logits.append(logits)
            
            # batch_Y is already multi-hot vectors with shape (batch_size, len(ccs_stoi))
            y_multi_hot = batch_Y.float().cpu()
            all_labels.append(y_multi_hot)
    
    # Concatenate all batches
    logits = torch.cat(all_logits, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    
    # Convert logits to probabilities
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    
    # Calculate metrics
    metrics = {}
    for k in ks:
        p_at_k = precision_at_k_visit(labels, probs, k)
        acc_at_k = accuracy_at_k_code(labels, probs, k)
        metrics[f"P@{k}"] = p_at_k
        metrics[f"Acc@{k}"] = acc_at_k
    
    return metrics


def evaluate_binary_classification(model, data_loader, device='cuda'):
    """Evaluate binary classification performance for AKI to CKD progression"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_Y in data_loader:
            batch_X_diag, batch_X_proc, batch_X_drug = batch_X
            if device != 'cpu':
                batch_X_diag = batch_X_diag.to(device)
                batch_X_proc = batch_X_proc.to(device)
                batch_X_drug = batch_X_drug.to(device)
                batch_Y = batch_Y.to(device)
            
            binary_preds = model(batch_X_diag, batch_X_proc, batch_X_drug, return_binary=True)  # (batch_size, 1) - binary 0/1
            
            if device != 'cpu':
                binary_preds = binary_preds.cpu()
                batch_Y = batch_Y.cpu()
            
            all_predictions.append(binary_preds)
            all_labels.append(batch_Y)
    
    # Concatenate all batches
    predictions = torch.cat(all_predictions, dim=0).squeeze()  # (total_samples,)
    labels = torch.cat(all_labels, dim=0).squeeze()  # (total_samples,)
    
    # Convert to numpy
    predictions_np = predictions.numpy().astype(int)
    labels_np = labels.numpy().astype(int)
    
    # Calculate binary classification metrics
    # Accuracy
    accuracy = (predictions_np == labels_np).mean()
    
    # Precision, Recall, F1
    true_positives = ((predictions_np == 1) & (labels_np == 1)).sum()
    false_positives = ((predictions_np == 1) & (labels_np == 0)).sum()
    false_negatives = ((predictions_np == 0) & (labels_np == 1)).sum()
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # For AUC, we need probabilities, but since model outputs binary predictions,
    # we'll use the predictions as probabilities (not ideal, but functional)
    # In practice, you might want to modify the model to also output probabilities
    try:
        auc = roc_auc_score(labels_np, predictions_np.astype(float))
    except ValueError:
        # Handle case where all labels are the same class
        auc = 0.5
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'positive_rate': labels_np.mean(),  # Proportion of positive cases
        'prediction_rate': predictions.mean()  # Proportion of positive predictions
    }
    
    return metrics


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='AKI to CKD progression prediction model training')
    
    parser.add_argument('--use_current_step', action='store_true',
                       help='Whether to use current step information (default: False)')
    parser.add_argument('--hidden', type=int, default=512,
                       help='Hidden layer dimension')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--wd', type=float, default=1e-5,
                       help='Weight decay (default: 1e-5)')
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--batch_size', type=int, default=24,
                       help='Batch size for training')

    # Early stopping parameters
    parser.add_argument('--patience', type=int, default=50,
                       help='Number of epochs to wait before stopping (default: 10)')
    parser.add_argument('--min_delta', type=float, default=0.001,
                       help='Minimum change to qualify as improvement (default: 0.001)')
    parser.add_argument('--monitor_metric', type=str, default='Acc@10',
                       choices=['P@10', 'Acc@10', 'P@20', 'Acc@20', 'P@30', 'Acc@30'],
                       help='Metric to monitor for early stopping (default: Acc@10)')
    
    # Transformer specific parameters
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of Transformer attention heads (default: 8)')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of Transformer layers (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate (default: 0.3)')
    
    # Data path
    parser.add_argument('--data_path', type=str, 
                       default="path/to/your/directory/preprocessed_data",
                       help='Preprocessed data directory path')
    parser.add_argument('--cache_path', type=str, 
                       default=None,
                       help='Path to save/load processed samples cache (default: ./cache/dialysis_prediction_samples.pkl)')
    parser.add_argument('--force_reload', action='store_true',
                       help='Force reload and reprocess data even if cache exists')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Using model: {'transformer'}")
    print(f"Hidden layer dimension: {args.hidden}")
    print(f"Learning rate: {args.lr}")
    print(f"Training epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Patience: {args.patience}, Min delta: {args.min_delta}, Monitor: {args.monitor_metric}")
    
    samples = load_preprocessed_data(args.data_path)

    # Prepare model parameters
    model_kwargs = {'p': args.dropout,}
    model_kwargs.update({
            'num_heads': args.num_heads,
            'num_layers': args.num_layers,
    })
     
    model, vocabs, ccs_itos, test_metrics = train_model_on_samples(
            samples,
            model_type='transformer',
            use_current_step=args.use_current_step,
            hidden=args.hidden,
            lr=args.lr,
            wd=args.wd,
            epochs=args.epochs,
            seed=args.seed,
            batch_size=args.batch_size,
            patience=args.patience,
            min_delta=args.min_delta,
            monitor_metric=args.monitor_metric,
            **model_kwargs
        )
        
    print(f"\n[DONE] Transformer model test results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")