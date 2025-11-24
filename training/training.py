"""
Training module
Contains main model training functions
"""
from itertools import combinations
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import geoopt
from geoopt.manifolds import PoincareBall
from collections import defaultdict
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, Dataset
from functools import partial
import os
from pathlib import Path

from model.models import create_model
from util.data_processing import (
    sort_samples_within_patient, 
    build_pairs, 
    build_vocab_from_pairs,
    prepare_XY, 
    split_by_patient
)
from metrics.metrics import precision_at_k_visit, accuracy_at_k_code, recall_at_k_micro


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
    """Custom collate function to pad variable-length sequences for three code types to global max lengths"""
    X_batch, Y_batch = zip(*batch)
    
    # Unpack three types of X
    X_diag_batch = [x[0] for x in X_batch]
    X_proc_batch = [x[1] for x in X_batch]
    X_drug_batch = [x[2] for x in X_batch]
    
    # Get max lengths (use provided global max, or fallback to batch max)
    if max_diag_len is None:
        max_diag_len = max(len(x) for x in X_diag_batch) if X_diag_batch else 0
    if max_proc_len is None:
        max_proc_len = max(len(x) for x in X_proc_batch) if X_proc_batch else 0
    if max_drug_len is None:
        max_drug_len = max(len(x) for x in X_drug_batch) if X_drug_batch else 0
    
    # Pad each type to global max length (not batch max)
    # First pad to batch max, then pad/truncate to global max
    X_diag_padded = torch.nn.utils.rnn.pad_sequence(X_diag_batch, batch_first=True, padding_value=0)
    if X_diag_padded.size(1) < max_diag_len:
        # Pad to global max length
        padding = torch.zeros(X_diag_padded.size(0), max_diag_len - X_diag_padded.size(1), 
                             dtype=X_diag_padded.dtype, device=X_diag_padded.device)
        X_diag_padded = torch.cat([X_diag_padded, padding], dim=1)
    else:
        # Truncate to global max length
        X_diag_padded = X_diag_padded[:, :max_diag_len]
    
    X_proc_padded = torch.nn.utils.rnn.pad_sequence(X_proc_batch, batch_first=True, padding_value=0)
    if X_proc_padded.size(1) < max_proc_len:
        padding = torch.zeros(X_proc_padded.size(0), max_proc_len - X_proc_padded.size(1), 
                             dtype=X_proc_padded.dtype, device=X_proc_padded.device)
        X_proc_padded = torch.cat([X_proc_padded, padding], dim=1)
    else:
        X_proc_padded = X_proc_padded[:, :max_proc_len]
    
    X_drug_padded = torch.nn.utils.rnn.pad_sequence(X_drug_batch, batch_first=True, padding_value=0)
    if X_drug_padded.size(1) < max_drug_len:
        padding = torch.zeros(X_drug_padded.size(0), max_drug_len - X_drug_padded.size(1), 
                             dtype=X_drug_padded.dtype, device=X_drug_padded.device)
        X_drug_padded = torch.cat([X_drug_padded, padding], dim=1)
    else:
        X_drug_padded = X_drug_padded[:, :max_drug_len]
    
    # For Y (multi-hot vectors), stack directly since they all have the same length (len(ccs_stoi))
    Y_padded = torch.stack(Y_batch)
    
    return (X_diag_padded, X_proc_padded, X_drug_padded), Y_padded


def train_model_on_samples(samples,
                           diag_trie,
                           model_type="transformer", 
                           task="next",          # "next" aligns with paper; "current" uses existing labels
                           use_current_step=False, # Admission prediction(False) or discharge prediction(True)
                           hidden=512, lr=1e-3, wd=1e-5,
                           epochs=10, seed=42, train_percentage=1.0,
                           batch_size=32,         # Batch size for training
                           early_stopping=True,   # Enable early stopping
                           patience=10,           # Number of epochs to wait before stopping
                           min_delta=0.001,      # Minimum change to qualify as improvement
                           monitor_metric='Acc@10', # Metric to monitor for early stopping
                           hierarchical_loss_weight=0.5,
                           **model_kwargs):
    # 1) Sort and assemble (current/next)
    # print(f"\nSamples: {samples[0]}")
    # 每一个sample就是一个病人的一条visit记录
    # 除了 adm_time 和 cond_hist 以外，其他字段都是这个 visit 特有的记录(这两个字段包含了这个病人过往的记录)
    # 每条 visit 里，cond_hist 包含病人过往 conditions 原代码，但是 conditions 字段是 CCS 映射后的代码
    by_pid = sort_samples_within_patient(samples)   # defaultdict(list), {"patient_id": [sample1, sample2, ...]}
    # print(f"\nBy pid: {by_pid['10001401']}")
    # build_pairs有问题。。我们应该是要用病人的所有过往visit记录来预测下一次的诊断，而不是上一次的visit
    # 没事了，cond_hist字段就是之前所有的visits
    pairs = build_pairs(by_pid, task=task)   # (sample_t, label_t+1)
    # print(f"\nPairs: {pairs[10]}")
    pairs = [(s, y_codes) for s, y_codes in pairs if s.get('cond_hist', []) and len([x for x in s['cond_hist'] if x]) > 0]

    # 2) Patient-level split
    train_pairs, val_pairs, test_pairs = split_by_patient(pairs, seed=seed)

    # Apply few-shot sampling to training data
    if train_percentage < 1.0:
        # Set random seed for reproducible sampling
        torch.manual_seed(seed)
        random.seed(seed)
        original_train_size = len(train_pairs)
        sample_size = int(original_train_size * train_percentage)
        train_pairs = random.sample(train_pairs, sample_size)
        print(f"Few-shot training: Using {len(train_pairs)}/{original_train_size} samples ({train_percentage:.1%} of training data)")

    # 3) Vocabulary
    (diag_stoi, diag_itos), (proc_stoi, proc_itos), (drug_stoi, drug_itos), (ccs_stoi, ccs_itos) = build_vocab_from_pairs(pairs) # diag_stoi={code: index}, diag_itos=[code1, code2, ...]
    # vocab_dict = {'diag_itos': diag_itos}
    # with open('vocab.pkl', 'wb') as f:
    #     pickle.dump(vocab_dict, f)
    # exit()
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

    # 6) Model and loss (multi-label)
    # Calculate vocabulary sizes for embedding layers
    # X vocab size = diag_stoi + proc_stoi + drug_stoi + 1 (for padding)
    # Y vocab size = ccs_stoi + 1 (for padding) - now uses CCS codes only for output
    diag_stoi, proc_stoi, drug_stoi, ccs_stoi = vocabs
    x_vocab_size = len(diag_stoi) + len(proc_stoi) + len(drug_stoi) + 1  # +1 for padding
    y_vocab_size = len(ccs_stoi)
    
    device = torch.device('cuda')
    print(f"Using device: {device}")
    print(f"X vocab size: {x_vocab_size}, Y vocab size: {y_vocab_size}")
    
    model_kwargs_with_max = {**model_kwargs, 
                             'diag_size': len(diag_stoi),
                             'proc_size': len(proc_stoi),
                             'diag_itos': diag_itos,
                             'max_diag_len': max_diag_len,
                            }
    model = create_model(model_type, x_vocab_size=x_vocab_size, hidden=hidden, out_dim=y_vocab_size, **model_kwargs_with_max)
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
        epoch_hier_loss = 0.0
        num_batches = 0
            
        # Training phase
        for batch_X, batch_Y in train_loader:
            batch_X_diag, batch_X_proc, batch_X_drug = batch_X
            batch_X_diag = batch_X_diag.to(device)   # (batch_size, max_diag_len)
            batch_X_proc = batch_X_proc.to(device)
            batch_X_drug = batch_X_drug.to(device)
            batch_Y = batch_Y.to(device)
            
            logits = model(batch_X_diag, batch_X_proc, batch_X_drug)  # (batch_size, y_vocab_size)
                
            loss = nn.functional.binary_cross_entropy_with_logits(logits, batch_Y)
            total_loss = loss
            hier_loss_value = 0.0  # Initialize hier_loss value
                
            opt.zero_grad()
            total_loss.backward()
            opt.step()
            epoch_loss += loss.item()
            epoch_hier_loss += hier_loss_value
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        avg_hier_loss = epoch_hier_loss / num_batches

        # Validation phase
        if ep % 1 == 0:
            val_metrics = evaluate_batched(model, val_loader, ks=(10, 20, 30), device=device)
            current_metric = val_metrics[monitor_metric]
                
            print(f"Epoch {ep:02d} | avg_loss={avg_loss:.4f} | hier_loss={avg_hier_loss:.4f} | val P@10={val_metrics['P@10']:.4f} Acc@10={val_metrics['Acc@10']:.4f}")   
            print(f"Logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}, std={logits.std().item():.4f}")
            # print(f"Predictions: {(torch.sigmoid(logits) > 0.5).sum().item()} / {logits.numel()} positive predictions")
            # Early stopping logic
            if early_stopping:
                if current_metric > best_metric + min_delta:
                    best_metric = current_metric
                    patience_counter = 0
                    # Save best model state
                    best_model_state = model.state_dict().copy()
                    print(f"  → New best {monitor_metric}: {best_metric:.4f}")
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered! No improvement in {monitor_metric} for {patience} epochs.")
                    print(f"Restoring best model from epoch {ep - patience_counter}")
                    model.load_state_dict(best_model_state)
                    break

    # 8) Test set evaluation (consistent with paper: Visit-level P@k, Code-level Acc@k)
    test_metrics = evaluate_batched(model, test_loader)
    print("[TEST]", test_metrics)
    
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
