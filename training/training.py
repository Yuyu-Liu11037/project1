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


def xi_poincare(p, c):
    '''双曲锥的开口角'''
    # p, c: shape (d,)
    # 与训练代码中的 angle_Xi 保持一致
    p2 = np.dot(p, p)
    c2 = np.dot(c, c)
    pc = np.dot(p, c)
    
    num = pc * (1 + p2) - p2 * (1 + c2)
    
    # 使用 clamp 保护数值稳定性，与训练代码一致
    p_minus_c = p - c
    p_minus_c_norm = np.linalg.norm(p_minus_c)
    p_minus_c_norm = max(p_minus_c_norm, 1e-15)  # 对应训练代码中的 clamp(min=1e-15)
    
    p_norm = np.sqrt(p2)
    p_norm = max(p_norm, 1e-15)  # 对应训练代码中的 clamp(min=1e-15)
    
    inside = 1 + p2 * c2 - 2 * pc
    inside = max(inside, 1e-15)  # 对应训练代码中的 clamp(min=1e-15)
    
    den = p_norm * p_minus_c_norm * np.sqrt(inside)
    
    # 与训练代码保持一致：clamp 到 [-1.0 + 1e-7, 1.0 - 1e-7]
    val = num / (den + 1e-15)
    val = np.clip(val, -1.0 + 1e-7, 1.0 - 1e-7)
    return np.arccos(val)  # Xi(p,c)


def psi_from_norm_poincare(norm_p, K, eps):
    '''以 p 为顶点的双曲锥体的半角'''
    # psi(p) = arcsin( K * (1 - ||p||^2) / ||p|| )
    # 与训练代码中的 psi 保持一致：使用 eps 来 clamp 范数的最小值
    norm_p = max(norm_p, eps)  # 对应训练代码中的 clamp(min=eps)
    arg = K * (1.0 - norm_p * norm_p) / (norm_p + 1e-15)
    # 与训练代码保持一致：clamp 到 [-1.0 + 1e-7, 1.0 - 1e-7]
    arg = np.clip(arg, -1.0 + 1e-7, 1.0 - 1e-7)
    return np.arcsin(arg)


def compute_xi_psi(p, c, K, eps):
    p2 = torch.sum(p ** 2)
    c2 = torch.sum(c ** 2)
    pc = torch.sum(p * c)
    
    num = pc * (1 + p2) - p2 * (1 + c2)
    
    p_minus_c = p - c
    p_minus_c_norm = torch.norm(p_minus_c).clamp_min(1e-15)
    p_norm = torch.sqrt(p2).clamp_min(1e-15)
    
    inside = 1 + p2 * c2 - 2 * pc
    inside = inside.clamp_min(1e-15)
    
    den = p_norm * p_minus_c_norm * torch.sqrt(inside)
    
    val = num / (den + 1e-15)
    val = torch.clamp(val, -1.0 + 1e-7, 1.0 - 1e-7)
    xi = torch.acos(val)
    
    norm_p = torch.norm(p).clamp_min(eps)
    arg = K * (1.0 - norm_p * norm_p) / (norm_p + 1e-15)
    arg = torch.clamp(arg, -1.0 + 1e-7, 1.0 - 1e-7)
    psi = torch.asin(arg)
    
    return xi, psi


def cone_violation(p, c, K, eps):
    xi, psi = compute_xi_psi(p, c, K, eps)
    violation = torch.clamp(xi - psi, min=0.0)  # Only penalize if xi > psi
    return violation


def poincare_distance(x, y, c=1.0, eps=1e-5):
    x_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True).clamp(max=1.0 - eps)
    y_norm_sq = torch.sum(y ** 2, dim=-1, keepdim=True).clamp(max=1.0 - eps)
    
    # Compute ||x - y||^2
    diff_norm_sq = torch.sum((x - y) ** 2, dim=-1, keepdim=True)
    
    # Poincaré distance formula: d(x,y) = (1/√c) * arccosh(1 + 2 * ||x-y||^2 / ((1-||x||^2)(1-||y||^2)))
    numerator = 2 * diff_norm_sq
    denominator = (1 - x_norm_sq) * (1 - y_norm_sq)
    denominator = denominator.clamp_min(eps)
    
    arg = 1 + numerator / denominator
    arg = arg.clamp_min(1.0 + eps)  # Ensure arg >= 1 for arccosh
    
    sqrt_c = c ** 0.5
    dist = (1.0 / sqrt_c) * torch.acosh(arg)
    
    return dist.squeeze(-1)


def find_parent_child_pairs(valid_indices, diag_itos, diag_trie):
    parent_child_pairs = []
    
    # Create mapping from code string to index (in diag_itos, which is 0-indexed)
    # Note: Z_diag[0] is padding, Z_diag[1] corresponds to diag_itos[0], etc.
    code_to_idx = {code: idx + 1 for idx, code in enumerate(diag_itos)}  # +1 because 0 is padding
    
    # Convert to list and get valid codes
    valid_indices_list = valid_indices.cpu().numpy().tolist()
    valid_codes = [diag_itos[idx - 1] for idx in valid_indices_list]  # -1 because 0 is padding
    
    # For each code, find its ancestors in the trie
    for code in valid_codes:
        all_sequences = diag_trie.get_all_sequences()
        sequences = [seq for seq in all_sequences if code in seq]
        
        # For each sequence, find parent-child relationships
        for seq in sequences:
            if code in seq:
                code_idx_in_seq = seq.index(code)
                if code_idx_in_seq > 0:
                    parent_code = seq[code_idx_in_seq - 1]
                    parent_idx = code_to_idx[parent_code]
                    child_idx = code_to_idx[code]
                    parent_child_pairs.append((parent_idx, child_idx))
    
    return list(set(parent_child_pairs))  # Remove duplicates


def hyperbolic_loss(patient_embeddings, Z_diag, diag_trie, diag_itos, batch_X_diag, 
                    K=1.0, eps=1e-5, num_negatives=10, margin=1.0, cone_weight=0.5):
    """
    Hyperbolic loss function for hierarchical diagnosis code embeddings.
    """
    batch_size = patient_embeddings.shape[0]
    device = 'cuda'
    D = len(diag_itos)  # Number of diagnosis codes (excluding padding)
    
    loss_list = []
    
    for i in range(batch_size):
        diag_indices = batch_X_diag[i].long()
        patient_embedding = patient_embeddings[i]  # (dim,)
        valid_mask = diag_indices > 0
        valid_indices = diag_indices[valid_mask]  # non-zero tokens
        
        # Part1: Contrastive loss
        positive_embeddings = Z_diag[valid_indices]  # (n_valid, dim)

        all_indices = torch.arange(1, D + 1, device=device)  # 1 to D (excluding 0 which is padding)
        negative_mask = torch.ones(len(all_indices), dtype=torch.bool, device=device)
        for idx in valid_indices:
            negative_mask[idx - 1] = False  # -1 because all_indices starts at 1
        negative_candidates = all_indices[negative_mask]
        num_neg = min(num_negatives, negative_candidates.numel())
        negative_indices = negative_candidates[torch.randperm(negative_candidates.numel(), device=device)[:num_neg]]
        negative_embeddings = Z_diag[negative_indices]  # (num_neg, dim)
        
        # Contrastive loss: minimize distance to positives, maximize distance to negatives
        # Average distance to positive samples
        pos_distances = poincare_distance(
            patient_embedding.unsqueeze(0),  # (1, dim)
            positive_embeddings  # (n_valid, dim)
        )  # (n_valid,)
        avg_pos_distance = pos_distances.mean()
        
        # Average distance to negative samples
        neg_distances = poincare_distance(
            patient_embedding.unsqueeze(0),  # (1, dim)
            negative_embeddings  # (num_neg, dim)
        )  # (num_neg,)
        avg_neg_distance = neg_distances.mean()
        
        # Contrastive loss: pull positives closer, push negatives away
        contrastive_loss = avg_pos_distance - avg_neg_distance + margin
        contrastive_loss = torch.clamp(contrastive_loss, min=0.0)
        
        # Part2: Hierarchical loss - cone constraint violation
        cone_loss_list = []
        parent_child_pairs = find_parent_child_pairs(valid_indices, diag_itos, diag_trie)
        
        for parent_idx, child_idx in parent_child_pairs:
            parent_emb = Z_diag[parent_idx]  # (dim,)
            child_emb = Z_diag[child_idx]  # (dim,)
            violation = cone_violation(parent_emb, child_emb, K, eps)
            cone_loss_list.append(violation)
        
        cone_loss = torch.stack(cone_loss_list).mean()
        # cone_loss = torch.tensor(0.0, device=device)
        loss_list.append(contrastive_loss + cone_weight * cone_loss)
    
    total_loss = torch.stack(loss_list).mean()
    return total_loss


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
                           hierarchical_loss_weight=0.1,  # Weight for hierarchical constraint loss
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
    (diag_stoi, diag_itos), (proc_stoi,_), (drug_stoi,_), (ccs_stoi, ccs_itos) = build_vocab_from_pairs(pairs) # diag_stoi={code: index}, diag_itos=[code1, code2, ...]
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
                            }
    model = create_model(model_type, x_vocab_size=x_vocab_size, hidden=hidden, out_dim=y_vocab_size, **model_kwargs_with_max)
    model = model.to(device) 
    
    # Separate parameters: hyperbolic parameters use RiemannianAdam, others use regular Adam
    # Note: In PoincareMap, all parameters (proj.weight, log_c, gamma) are in Euclidean space.
    # The output embeddings are on the Poincaré ball, but they are computed, not parameters.
    # Since c is learnable and changes during training, we cannot use a fixed manifold.
    # Instead, we use RiemannianAdam without manifold for hyperbolic parameters (Euclidean updates),
    # which still provides better optimization for hyperbolic-related parameters.
    
    hyperbolic_params = []
    euclidean_params = []
    
    # Collect hyperbolic parameters from PoincareMap modules
    # These include: proj.weight, proj.bias, log_c, gamma
    hyperbolic_params.extend(list(model.patient_hyp_head.parameters()))
    hyperbolic_params.extend(list(model.diag_hyp_head.parameters()))
    
    # Collect all other parameters
    for name, param in model.named_parameters():
        if 'patient_hyp_head' not in name and 'diag_hyp_head' not in name:
            euclidean_params.append(param)
    
    # Create parameter groups
    # Since c is learnable, we cannot use a fixed manifold.
    # RiemannianAdam without manifold will use Euclidean updates, but it's still
    # beneficial for hyperbolic-related parameters due to its adaptive learning rate.
    param_groups = [
        # Euclidean parameters (standard parameters)
        {'params': euclidean_params, 'lr': lr, 'weight_decay': wd},
        # Hyperbolic parameters (no manifold, but use RiemannianAdam for better optimization)
        {'params': hyperbolic_params, 'lr': lr, 'weight_decay': wd}
    ]
    
    # Use RiemannianAdam which can handle both Euclidean and Riemannian parameters
    # Without manifold specified, it will use Euclidean updates for all parameters
    opt = geoopt.optim.RiemannianAdam(param_groups, lr=lr, weight_decay=wd)

    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save initial model parameters before training
    initial_model_path = checkpoint_dir / "model_initial.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'epoch': 0,
    }, initial_model_path)
    print(f"Saved initial model parameters to {initial_model_path}")

    # 7) Training loop with batches and early stopping
    best_metric = -float('inf')
    patience_counter = 0
    best_model_state = None
    final_epoch = 0  # Track the final epoch number
    
    print(f"Training for {epochs} epochs")
    for ep in range(1, epochs+1):
        final_epoch = ep
        model.train()
        epoch_loss = 0.0
        epoch_hier_loss = 0.0
        num_batches = 0
        
        # Training phase
        for batch_X, batch_Y in train_loader:
            batch_X_diag, batch_X_proc, batch_X_drug = batch_X
            batch_X_diag = batch_X_diag.to(device)
            batch_X_proc = batch_X_proc.to(device)
            batch_X_drug = batch_X_drug.to(device)
            batch_Y = batch_Y.to(device)
            
            logits, z_patient = model(batch_X_diag, batch_X_proc, batch_X_drug)  # (batch_size, y_vocab_size)
            
            loss = nn.functional.binary_cross_entropy_with_logits(logits, batch_Y)
            total_loss = loss
            hier_loss_value = 0.0  # Initialize hier_loss value
            
            if hierarchical_loss_weight > 0:
                Z_diag    = model.get_diag_hyperbolic()
                hier_loss = hyperbolic_loss(z_patient, Z_diag, diag_trie, diag_itos, batch_X_diag)
                total_loss = loss + hierarchical_loss_weight * hier_loss
                hier_loss_value = hier_loss.item()
            
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
            
            # Get log_c values from both PoincareMap modules
            patient_c = model.patient_hyp_head.c.item()
            diag_c = model.diag_hyp_head.c.item()
            
            print(f"Epoch {ep:02d} | avg_loss={avg_loss:.4f} | hier_loss={avg_hier_loss:.4f} | "
                  f"val P@10={val_metrics['P@10']:.4f} Acc@10={val_metrics['Acc@10']:.4f}  | "
                  f"c: patient={patient_c:.4f} diag={diag_c:.4f}")
            
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
    test_metrics = evaluate_batched(model, test_loader, ks=(10, 20, 30), device=device)
    print("[TEST]", test_metrics)
    
    # Save final model parameters after training
    final_model_path = checkpoint_dir / "model_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'test_metrics': test_metrics,
        'epoch': final_epoch,
    }, final_model_path)
    print(f"Saved final model parameters to {final_model_path}")
    
    return model, vocabs, ccs_itos, test_metrics


def evaluate_batched(model, data_loader, ks=(10, 20, 30), device='cuda'):
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_Y in data_loader:
            batch_X_diag, batch_X_proc, batch_X_drug = batch_X
            batch_X_diag = batch_X_diag.to(device)
            batch_X_proc = batch_X_proc.to(device)
            batch_X_drug = batch_X_drug.to(device)
            batch_Y = batch_Y.to(device)
            
            logits, _ = model(batch_X_diag, batch_X_proc, batch_X_drug)
            all_logits.append(logits.cpu())
            
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
