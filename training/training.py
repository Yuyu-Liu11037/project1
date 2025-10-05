"""
Training module
Contains main model training functions
"""
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader, TensorDataset

from model.models import create_model
from util.data_processing import (
    sort_samples_within_patient, 
    build_pairs, 
    build_vocab_from_pairs,
    prepare_XY, 
    split_by_patient
)
from metrics.metrics import bce_pos_weight, evaluate


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
    print(f"\nSamples: {samples[0]}")
    by_pid = sort_samples_within_patient(samples)   # defaultdict(list), {"patient_id": [sample1, sample2, ...]}
    print(f"\nBy pid: {by_pid['10001401']}")
    # build_pairs有问题。。我们应该是要用病人的所有过往visit记录来预测下一次的诊断，而不是上一次的visit
    # 没事了，cond_hist字段就是之前所有的visits
    pairs = build_pairs(by_pid, task=task)   # (sample_t, label_t+1)
    print(f"\nPairs: {pairs[10]}")

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
    (diag_stoi,_), (proc_stoi,_), (drug_stoi,_), (y_stoi, y_itos) = build_vocab_from_pairs(train_pairs)
    vocabs = (diag_stoi, proc_stoi, drug_stoi, y_stoi)

    # 4) Vectorization
    Xtr, Ytr = prepare_XY(train_pairs, vocabs, use_current_step=use_current_step)
    print(f"\nTrain data shape: {Xtr.shape}")
    print(f"\nTrain data: {Xtr[10]}")
    print(f"\nTrain label shape: {Ytr.shape}")
    print(f"\nTrain label: {Ytr[10]}")
    Xva, Yva = prepare_XY(val_pairs,   vocabs, use_current_step=use_current_step)
    Xte, Yte = prepare_XY(test_pairs,   vocabs, use_current_step=use_current_step)

    # 5) Create DataLoaders for batch training
    train_dataset = TensorDataset(Xtr, Ytr)
    print(f"\nTrain dataset: {train_dataset[0]}")
    val_dataset = TensorDataset(Xva, Yva)
    test_dataset = TensorDataset(Xte, Yte)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 6) Model and loss (multi-label)
    # input: (batch_size, in_dim), in_dim = historical diagnosis codes (ICD)len(diag_stoi) + historical procedure codes (Procedures)len(proc_stoi) + historical drug codes (ATC-3)len(drug_stoi)
    # output: (batch_size, out_dim), out_dim = label codes (CCS)len(y_stoi), size of label vocabulary
    model = create_model(model_type, Xtr.size(1), hidden=hidden, out_dim=Ytr.size(1), **model_kwargs)
    pw = bce_pos_weight(Ytr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
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
            logits = model(batch_X)
            loss = criterion(logits, batch_Y)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        # Validation phase
        if ep % 1 == 0:
            val_metrics = evaluate_batched(model, val_loader, ks=(10, 20))
            current_metric = val_metrics[monitor_metric]
            
            print(f"Epoch {ep:02d} | avg_loss={avg_loss:.4f} | "
                  f"val P@10={val_metrics['P@10']:.4f} Acc@10={val_metrics['Acc@10']:.4f} "
                  f"P@20={val_metrics['P@20']:.4f} Acc@20={val_metrics['Acc@20']:.4f}")
            
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
    test_metrics = evaluate_batched(model, test_loader, ks=(10, 20))
    print("[TEST]", test_metrics)
    
    return model, vocabs, y_itos, test_metrics


def train_mlp_on_samples(samples, **kwargs):
    """Backward compatible MLP training function"""
    return train_model_on_samples(samples, model_type="mlp", **kwargs)


def evaluate_batched(model, data_loader, ks=(10, 20)):
    """
    Evaluate model using batched data loader
    Compatible with the original evaluate function but works with DataLoader
    
    Args:
        model: Trained model
        data_loader: DataLoader containing (X, Y) batches
        ks: List of k values for evaluation metrics
    
    Returns:
        Dictionary containing evaluation metrics
    """
    import torch
    import numpy as np
    from metrics.metrics import precision_at_k_visit, accuracy_at_k_code, recall_at_k_micro
    
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_Y in data_loader:
            logits = model(batch_X)
            all_logits.append(logits.cpu())
            all_labels.append(batch_Y.cpu())
    
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

