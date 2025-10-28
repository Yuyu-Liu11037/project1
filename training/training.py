"""
Training module
Contains main model training functions
"""
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import defaultdict
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, Dataset

from model.models import create_model
from util.data_processing import (
    sort_samples_within_patient, 
    build_pairs, 
    build_vocab_from_pairs,
    prepare_XY, 
    split_by_patient
)
from metrics.metrics import evaluate


class VariableLengthDataset(Dataset):
    """Dataset for variable-length tensors"""
    def __init__(self, X_list, Y_list):
        """
        Args:
            X_list: list of 1D tensors (variable length)
            Y_list: list of 1D tensors (variable length)
        """
        self.X_list = X_list
        self.Y_list = Y_list
    
    def __len__(self):
        return len(self.X_list)
    
    def __getitem__(self, idx):
        return self.X_list[idx], self.Y_list[idx]


def collate_fn(batch):
    """Custom collate function to pad variable-length sequences"""
    X_batch, Y_batch = zip(*batch)
    
    # Pad X sequences
    X_padded = torch.nn.utils.rnn.pad_sequence(X_batch, batch_first=True, padding_value=0)
    
    # For Y (multi-label), we need a different approach
    # Find max length of Y sequences and pad
    max_y_len = max(len(y) for y in Y_batch)
    Y_padded_list = []
    for y in Y_batch:
        padded_y = torch.nn.functional.pad(y, (0, max_y_len - len(y)), value=0)
        Y_padded_list.append(padded_y)
    Y_padded = torch.stack(Y_padded_list)
    
    return X_padded, Y_padded


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
        **model_kwargs: Model-specific parameters (e.g., num_heads, num_layers, etc.)
    
    Returns:
        model: Trained model
        vocabs: Vocabulary dictionaries
        y_itos: Label index to string mapping
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
    (diag_stoi,_), (proc_stoi,_), (drug_stoi,_), (y_stoi, y_itos) = build_vocab_from_pairs(train_pairs) # y_stoi = diag_stoi
    vocabs = (diag_stoi, proc_stoi, drug_stoi, y_stoi)

    # 4) Vectorization
    Xtr, Ytr = prepare_XY(train_pairs, vocabs, use_current_step=use_current_step)
    Xva, Yva = prepare_XY(val_pairs,   vocabs, use_current_step=use_current_step)
    Xte, Yte = prepare_XY(test_pairs,   vocabs, use_current_step=use_current_step)

    # 5) Create DataLoaders for batch training with custom collate function
    train_dataset = VariableLengthDataset(Xtr, Ytr)
    val_dataset = VariableLengthDataset(Xva, Yva)
    test_dataset = VariableLengthDataset(Xte, Yte)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 6) Model and loss (multi-label)
    # Calculate vocabulary sizes for embedding layers
    # X vocab size = diag_stoi + proc_stoi + drug_stoi + 1 (for padding)
    # Y vocab size = diag_stoi + 1 (for padding) since Y and diag share the same vocab
    diag_stoi, proc_stoi, drug_stoi, y_stoi = vocabs
    x_vocab_size = len(diag_stoi) + len(proc_stoi) + len(drug_stoi) + 1  # +1 for padding
    y_vocab_size = len(diag_stoi) + 1  # +1 for padding (Y shares vocab with diag)
    
    device = torch.device('cuda')
    print(f"Using device: {device}")
    print(f"X vocab size: {x_vocab_size}, Y vocab size: {y_vocab_size}")
    
    model = create_model(model_type, x_vocab_size=x_vocab_size, hidden=hidden, out_dim=y_vocab_size, **model_kwargs)
    model = model.to(device) 
    
    # Loss will be computed in the training loop by converting indices to multi-hot
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # 7) Training loop with batches and early stopping
    best_metric = -float('inf')
    patience_counter = 0
    best_model_state = None
    
    for ep in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Training phase
        for batch_X, batch_Y in train_loader:
            # print(batch_X[0])
            # print(batch_Y[0])
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            logits = model(batch_X)  # (batch_size, y_vocab_size)
            # print(logits[0])
            
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
            
            opt.zero_grad()
            loss.backward()
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
    
    return model, vocabs, y_itos, test_metrics


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
    import torch
    import numpy as np
    from metrics.metrics import precision_at_k_visit, accuracy_at_k_code, recall_at_k_micro
    
    model.eval()
    all_logits = []
    all_labels = []
    
    # Use model's device if device not specified
    if device is None:
        device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch_X, batch_Y in data_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            logits = model(batch_X)
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

