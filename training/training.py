"""
Training module
Contains main model training functions
"""
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import geoopt
from collections import defaultdict
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, Dataset
from functools import partial

from model.models import create_model
from util.data_processing import (
    sort_samples_within_patient, 
    build_pairs, 
    build_vocab_from_pairs,
    prepare_XY, 
    split_by_patient,
    build_icd_hierarchy
)
from metrics.metrics import evaluate, precision_at_k_visit, accuracy_at_k_code, recall_at_k_micro


class VariableLengthDataset(Dataset):
    """Dataset for variable-length tensors with three separate code types"""
    def __init__(self, X_list_diag, X_list_proc, X_list_drug, Y_list):
        """
        Args:
            X_list_diag: list of 1D tensors for diagnosis codes (variable length)
            X_list_proc: list of 1D tensors for procedure codes (variable length)
            X_list_drug: list of 1D tensors for drug codes (variable length)
            Y_list: list of 1D tensors for labels (variable length)
        """
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
    
    # For Y (multi-label), pad to max length
    max_y_len = max(len(y) for y in Y_batch)
    Y_padded_list = []
    for y in Y_batch:
        padded_y = torch.nn.functional.pad(y, (0, max_y_len - len(y)), value=0)
        Y_padded_list.append(padded_y)
    Y_padded = torch.stack(Y_padded_list)
    
    return (X_diag_padded, X_proc_padded, X_drug_padded), Y_padded


def compute_hierarchical_loss(model, icd_hierarchy, ccs_groups, diag_stoi, diag_itos, device='cuda', relevant_codes=None):
    """
    Compute hierarchical constraint loss in hyperbolic space.
    
    Args:
        model: The model with hyperbolic embeddings
        icd_hierarchy: Dictionary {icd_code: [parent]} for prefix hierarchy
        ccs_groups: Dictionary {ccs_code: [icd_code1, icd_code2, ...]} for CCS grouping
        diag_stoi: Dictionary mapping ICD codes to indices
        diag_itos: Dictionary mapping indices to ICD codes
        device: Device for computation
        relevant_codes: Set of codes to compute constraints for (if None, compute for all)
    
    Returns:
        hierarchical_loss: Scalar tensor with hyperbolic distance constraints
    """
    total_loss = 0.0
    count = 0
    
    # Get the hyperbolic embedding layer for diagnosis codes
    emb_diag = model.emb_diag
    ball = emb_diag.ball
    
    # Build set of relevant codes and their ancestors for batch-based optimization
    if relevant_codes is not None:
        # Collect all ancestors of relevant codes
        processed = set()
        codes_to_process = relevant_codes.copy()
        
        while codes_to_process:
            code = codes_to_process.pop()
            if code in processed or code not in icd_hierarchy:
                continue
            processed.add(code)
            # Add parent codes to be processed recursively
            for parent in icd_hierarchy[code]:
                if parent not in processed and parent in diag_stoi:
                    codes_to_process.add(parent)
        
        relevant_codes_set = processed
    else:
        relevant_codes_set = None  # Process all codes
    
    # Constraint 1: ICD tree hierarchy (child close to parent + child deeper than parent)
    for child_code, parent_codes in icd_hierarchy.items():
        if child_code not in diag_stoi:
            continue
        
        # Only process if relevant_codes is None (all) or child_code is in relevant set
        if relevant_codes_set is not None and child_code not in relevant_codes_set:
            continue
        
        child_idx = diag_stoi[child_code]
        child_emb = emb_diag.weight[child_idx]  # Hyperbolic embedding
        
        for parent_code in parent_codes:
            if parent_code not in diag_stoi:
                continue
            
            parent_idx = diag_stoi[parent_code]
            parent_emb = emb_diag.weight[parent_idx]
            
            # Constraint 1a: Child should be close to parent (distance constraint)
            dist_parent_child = ball.dist(child_emb.unsqueeze(0), parent_emb.unsqueeze(0))
            
            # Constraint 1b: Child should be deeper than parent (closer to boundary)
            # In hyperbolic space, distance from origin represents depth in the tree
            dist_child_origin = torch.norm(child_emb)
            dist_parent_origin = torch.norm(parent_emb)
            
            # Child should be further from origin than parent
            # Use relu to penalize when child is NOT deeper than parent
            depth_penalty = torch.relu(dist_parent_origin - dist_child_origin)
            
            # Total constraint loss for this pair
            pair_loss = dist_parent_child + depth_penalty
            total_loss += pair_loss
            count += 1
    
    # Constraint 2: CCS grouping (ICDs mapped to same CCS should be close)
    for ccs_code, icd_codes in ccs_groups.items():
        if len(icd_codes) < 2:  # Need at least 2 codes to form a constraint
            continue
        
        # Compute pairwise distances between all ICDs in this CCS group
        # Only include codes in relevant set if filtering is enabled
        if relevant_codes_set is not None:
            valid_icd_indices = [diag_stoi[code] for code in icd_codes 
                               if code in diag_stoi and code in relevant_codes_set]
        else:
            valid_icd_indices = [diag_stoi[code] for code in icd_codes if code in diag_stoi]
        
        if len(valid_icd_indices) < 2:
            continue
        
        for i in range(len(valid_icd_indices)):
            for j in range(i + 1, len(valid_icd_indices)):
                idx1, idx2 = valid_icd_indices[i], valid_icd_indices[j]
                emb1 = emb_diag.weight[idx1]
                emb2 = emb_diag.weight[idx2]
                
                # Compute hyperbolic distance
                dist = ball.dist(emb1.unsqueeze(0), emb2.unsqueeze(0))
                total_loss += dist
                count += 1
    
    if count > 0:
        return total_loss / count
    else:
        return torch.tensor(0.0, device=device)


def train_model_on_samples(samples,
                           model_type="mlp",      # "mlp" or "transformer"
                           task="next",          # "next" aligns with paper; "current" uses existing labels
                           use_current_step=False, # Admission prediction(False) or discharge prediction(True)
                           hidden=512, lr=1e-3, wd=1e-5,
                           epochs=10, seed=42, train_percentage=1.0,
                           batch_size=32,         # Batch size for training
                           early_stopping=True,   # Enable early stopping
                           patience=10,           # Number of epochs to wait before stopping
                           min_delta=0.001,      # Minimum change to qualify as improvement
                           monitor_metric='Acc@10', # Metric to monitor for early stopping
                           hierarchical_loss_weight=0.1,  # Weight for hierarchical constraint loss
                           **model_kwargs):
    """
    Train model for diagnosis prediction
    
    Args:
        samples: Sample data
        model_type: "mlp" or "transformer", model type
        task: "next" or "current", prediction task type
        use_current_step: Whether to use current step information
        hidden: Hidden layer dimension
        lr: Learning rate
        wd: Weight decay
        epochs: Number of training epochs
        seed: Random seed
        train_percentage: Percentage of training data to use (0.01-1.0), for few-shot training
        batch_size: Batch size for training (default: 32)
        early_stopping: Enable early stopping (default: True)
        patience: Number of epochs to wait before stopping (default: 10)
        min_delta: Minimum change to qualify as improvement (default: 0.001)
        monitor_metric: Metric to monitor for early stopping (default: 'Acc@10')
        hierarchical_loss_weight: Weight for hierarchical constraint loss (default: 0.1)
        **model_kwargs: Model-specific parameters (e.g., num_heads, num_layers, etc.)
    
    Returns:
        model: Trained model
        vocabs: Vocabulary dictionaries
        ccs_itos: Label index to string mapping (CCS codes)
        test_metrics: Test set evaluation metrics
    """
    # 1) Sort and assemble (current/next)
    # print(f"\nSamples: {samples[0]}")
    # 每一个sample就是一个病人的一条visit记录
    # 除了 adm_time 和 cond_hist 以外，其他字段都是这个 visit 特有的记录(这两个字段包含了这个病人过往的记录)
    # 每条 visit 里，cond_hist 包含病人过往 conditions 原代码，但是 conditions 字段是 CCS 映射后的代码
    # TODO: redundant
    by_pid = sort_samples_within_patient(samples)   # defaultdict(list), {"patient_id": [sample1, sample2, ...]}
    # print(f"\nBy pid: {by_pid['10001401']}")
    # build_pairs有问题。。我们应该是要用病人的所有过往visit记录来预测下一次的诊断，而不是上一次的visit
    # 没事了，cond_hist字段就是之前所有的visits
    pairs = build_pairs(by_pid, task=task)   # (sample_t, label_t+1)
    # print(f"\nPairs: {pairs[10]}")
    
    # Filter out samples with empty cond_hist (first visits without history)
    original_num_pairs = len(pairs)
    pairs = [(s, y_codes) for s, y_codes in pairs if s.get('cond_hist', []) and len([x for x in s['cond_hist'] if x]) > 0]
    filtered_num_pairs = len(pairs)
    print(f"\nFiltered {original_num_pairs - filtered_num_pairs} samples with empty cond_hist")
    print(f"Remaining samples: {filtered_num_pairs}/{original_num_pairs}")

    # 2) Patient-level split
    train_pairs, val_pairs, test_pairs = split_by_patient(pairs, seed=seed)
    
    # Check cond_hist lengths
    cond_hist_lengths = [len(s.get('cond_hist', [])) for s, _ in train_pairs]
    if cond_hist_lengths:
        print(f"\ncond_hist length stats: min={min(cond_hist_lengths)}, max={max(cond_hist_lengths)}, avg={sum(cond_hist_lengths)/len(cond_hist_lengths):.2f}")
    
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
    (diag_stoi, diag_itos), (proc_stoi,_), (drug_stoi,_), (ccs_stoi, ccs_itos) = build_vocab_from_pairs(train_pairs) # diag_stoi=ICD, ccs_stoi=CCS for labels
    vocabs = (diag_stoi, proc_stoi, drug_stoi, ccs_stoi)
    
    # 3.5) Build ICD hierarchy structure
    icd_hierarchy, ccs_groups, icd_to_ccs, additional_codes = build_icd_hierarchy(diag_stoi, ccs_stoi)
    print(f"\nBuilt ICD hierarchy: {len(icd_hierarchy)} codes with parent-child relationships")
    print(f"CCS groups: {len(ccs_groups)} distinct CCS codes mapped from ICD codes")
    
    # Expand vocab with additional parent codes
    if additional_codes:
        print(f"Expanding vocabulary with {len(additional_codes)} additional parent codes")
        start_idx = len(diag_itos)
        for code in additional_codes:
            diag_itos.append(code)
            diag_stoi[code] = start_idx
            start_idx += 1
        # Update vocabs tuple with expanded diag vocab
        vocabs = (diag_stoi, proc_stoi, drug_stoi, ccs_stoi)
        print(f"Expanded diag vocab size: {len(diag_stoi)}")

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
    y_vocab_size = len(ccs_stoi) + 1  # +1 for padding (Y uses CCS vocab, separate from diag)
    
    device = torch.device('cuda')
    print(f"Using device: {device}")
    print(f"X vocab size: {x_vocab_size}, Y vocab size: {y_vocab_size}")
    
    # Add max sequence lengths and vocab sizes to model kwargs
    model_kwargs_with_max = {**model_kwargs, 
                             'diag_size': len(diag_stoi),
                             'proc_size': len(proc_stoi),
                            }
    model = create_model(model_type, x_vocab_size=x_vocab_size, hidden=hidden, out_dim=y_vocab_size, **model_kwargs_with_max)
    model = model.to(device) 
    
    mani_params = [p for p in model.parameters() if isinstance(p, geoopt.ManifoldParameter)]
    euc_params  = [p for p in model.parameters() if not isinstance(p, geoopt.ManifoldParameter)]

    opt = geoopt.optim.RiemannianAdam([
        {"params": mani_params, "lr": lr, "weight_decay": wd},
        {"params": euc_params,  "lr": lr, "weight_decay": wd},
    ])

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
            # batch_X is a tuple of (batch_X_diag, batch_X_proc, batch_X_drug)
            batch_X_diag, batch_X_proc, batch_X_drug = batch_X
            # print(f"batch_X_diag: {batch_X_diag.shape}")
            # print(f"batch_X_diag: {batch_X_diag[0]}")  # a lot of 0s
            # print(f"batch_X_proc: {batch_X_proc.shape}")
            # print(f"batch_X_proc: {batch_X_proc[0]}")
            # print(f"batch_X_drug: {batch_X_drug.shape}")
            # print(f"batch_X_drug: {batch_X_drug[0]}")
            batch_X_diag = batch_X_diag.to(device)
            batch_X_proc = batch_X_proc.to(device)
            batch_X_drug = batch_X_drug.to(device)
            batch_Y = batch_Y.to(device)
            
            logits = model(batch_X_diag, batch_X_proc, batch_X_drug)  # (batch_size, y_vocab_size)
            
            # Convert Y from padded index sequences to multi-hot vectors
            # batch_Y shape: (batch_size, max_y_len) with indices (0 = padding)
            batch_size = batch_Y.size(0)
            max_y_len = batch_Y.size(1)
            y_multi_hot = torch.zeros(batch_size, logits.size(1), device=device)
            
            for i in range(batch_size):
                # Extract valid labels (non-zero indices) for this sample
                valid_labels = batch_Y[i][batch_Y[i] != 0]  # Remove padding
                y_multi_hot[i, valid_labels] = 1.0  # Set positions to 1
            
            # Calculate loss using BCE
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y_multi_hot)
            
            # Add hierarchical loss if weight > 0
            if hierarchical_loss_weight > 0:
                # Extract relevant codes from batch for optimized hierarchical loss computation
                # batch_X_diag uses offset=1 (indices 1 to len(diag_stoi)), 0 is padding
                unique_indices = torch.unique(batch_X_diag[batch_X_diag != 0])
                # Convert from 1-indexed to 0-indexed (remove offset)
                adjusted_indices = unique_indices - 1
                # Filter valid indices and convert to codes
                batch_codes = set()
                for idx in adjusted_indices:
                    if 0 <= idx < len(diag_itos):
                        code = diag_itos[idx]
                        batch_codes.add(code)
                
                hier_loss = compute_hierarchical_loss(model, icd_hierarchy, ccs_groups, diag_stoi, diag_itos, device, relevant_codes=batch_codes)
                total_loss = loss + hierarchical_loss_weight * hier_loss
            else:
                total_loss = loss
            
            opt.zero_grad()
            total_loss.backward()
            with torch.no_grad():
                model.reproject_hyperbolic_()  # Reproject all three hyperbolic embeddings
            opt.step()
            
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        # Validation phase
        if ep % 1 == 0:
            val_metrics = evaluate_batched(model, val_loader, ks=(10, 20, 30), device=device)
            current_metric = val_metrics[monitor_metric]
            
            print(f"Epoch {ep:02d} | avg_loss={avg_loss:.4f} | "
                  f"val P@10={val_metrics['P@10']:.4f} Acc@10={val_metrics['Acc@10']:.4f} "
                  f"P@20={val_metrics['P@20']:.4f} Acc@20={val_metrics['Acc@20']:.4f} "
                  f"P@30={val_metrics['P@30']:.4f} Acc@30={val_metrics['Acc@30']:.4f}")
            
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
                    print(f"  → No improvement for {patience_counter} epochs (best {monitor_metric}: {best_metric:.4f})")
                
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered! No improvement in {monitor_metric} for {patience} epochs.")
                    print(f"Restoring best model from epoch {ep - patience_counter}")
                    model.load_state_dict(best_model_state)
                    break

    # 8) Test set evaluation (consistent with paper: Visit-level P@k, Code-level Acc@k)
    test_metrics = evaluate_batched(model, test_loader, ks=(10, 20, 30), device=device)
    print("[TEST]", test_metrics)
    
    return model, vocabs, ccs_itos, test_metrics


def evaluate_batched(model, data_loader, ks=(10, 20, 30), device=None):
    """
    Evaluate model using batched data loader
    Compatible with the original evaluate function but works with DataLoader
    
    Args:
        model: Trained model
        data_loader: DataLoader containing (X, Y) batches
        ks: List of k values for evaluation metrics
        device: Device to use for evaluation (if None, uses model's device)
    
    Returns:
        Dictionary containing evaluation metrics
    """
    
    model.eval()
    all_logits = []
    all_labels = []
    
    # Use model's device if device not specified
    if device is None:
        device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch_X, batch_Y in data_loader:
            batch_X_diag, batch_X_proc, batch_X_drug = batch_X
            batch_X_diag = batch_X_diag.to(device)
            batch_X_proc = batch_X_proc.to(device)
            batch_X_drug = batch_X_drug.to(device)
            batch_Y = batch_Y.to(device)
            
            logits = model(batch_X_diag, batch_X_proc, batch_X_drug)
            all_logits.append(logits.cpu())
            
            # Convert padded index sequences to multi-hot vectors
            batch_size = batch_Y.size(0)
            num_labels = logits.size(1)
            y_multi_hot = torch.zeros(batch_size, num_labels)
            
            for i in range(batch_size):
                valid_labels = batch_Y[i][batch_Y[i] != 0].cpu()
                y_multi_hot[i, valid_labels] = 1.0
            
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
        r_at_k = recall_at_k_micro(labels, probs, k)
        metrics[f"P@{k}"] = p_at_k
        metrics[f"Acc@{k}"] = acc_at_k
        metrics[f"Recall@{k}"] = r_at_k
    
    return metrics
