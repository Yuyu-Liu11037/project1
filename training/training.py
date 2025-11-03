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
from torch.utils.data import DataLoader, TensorDataset

from model.models import create_model, create_model_from_data
from util.data_processing import (
    sort_samples_within_patient, 
    build_pairs, 
    build_vocab_from_pairs,
    prepare_XY, 
    split_by_patient,
    build_dialysis_pairs,
    build_dialysis_vocab_from_pairs,
    prepare_dialysis_XY
)
from metrics.metrics import bce_pos_weight, evaluate


def aggregate_seed_results(seed_results):
    """
    Aggregate results across all seeds
    
    Args:
        seed_results: List of aggregated results from each seed
    
    Returns:
        Dictionary with final statistics across all seeds
    """
    final_results = {}
    
    # Get all metric names
    metric_names = seed_results[0].keys()
    
    for metric in metric_names:
        means = [seed[metric]['mean'] for seed in seed_results if metric in seed]
        stds = [seed[metric]['std'] for seed in seed_results if metric in seed]
        mins = [seed[metric]['min'] for seed in seed_results if metric in seed]
        maxs = [seed[metric]['max'] for seed in seed_results if metric in seed]
        
        if means:
            final_results[metric] = {
                'mean': np.mean(means),
                'std': np.std(means),
                'min': np.min(mins),
                'max': np.max(maxs)
            }
    
    return final_results


def train_diagnosis_model_on_samples(samples,
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
                           use_gpu=True, force_cpu=False,  # GPU control
                           use_hyperbolic_embeddings=False,  # Use hyperbolic embeddings
                           embedding_file="hyperbolic_embeddings.pkl",  # Path to embeddings file
                           max_seq_length=200,    # Maximum sequence length for sequential data
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
    # Each sample is a patient's visit record
    # Except for adm_time and cond_hist, other fields are specific to this visit (these two fields contain the patient's past records)
    # In each visit, cond_hist contains the patient's past condition original codes, but the conditions field is CCS-mapped codes
    # TODO: redundant
    by_pid = sort_samples_within_patient(samples)   # defaultdict(list), {"patient_id": [sample1, sample2, ...]}
    # print(f"\nBy pid: {by_pid['10001401']}")
    # build_pairs has issues... We should use all past visit records of the patient to predict the next diagnosis, not the previous visit
    # Never mind, the cond_hist field contains all previous visits
    pairs = build_pairs(by_pid, task=task)   # (sample_t, label_t+1)
    # print(f"\nPairs: {pairs[10]}")

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
    if use_hyperbolic_embeddings:
        print(f"Using hyperbolic embeddings with variable-length sequences")
        Xtr, Ytr = prepare_XY(train_pairs, vocabs, use_current_step=use_current_step, 
                              use_hyperbolic_embeddings=True, embedding_file=embedding_file)
        Xva, Yva = prepare_XY(val_pairs, vocabs, use_current_step=use_current_step,
                              use_hyperbolic_embeddings=True, embedding_file=embedding_file)
        Xte, Yte = prepare_XY(test_pairs, vocabs, use_current_step=use_current_step,
                             use_hyperbolic_embeddings=True, embedding_file=embedding_file)
        # Xtr is a list of tensors, Ytr is a tensor
        print(f"\nTrain data: list of {len(Xtr)} tensors with variable lengths")
        print(f"Train label shape: {Ytr.shape}")  # (batch_size, num_labels)
        # Create list dataset for variable-length sequences
        train_dataset = list(zip(Xtr, Ytr))
        val_dataset = list(zip(Xva, Yva))
        test_dataset = list(zip(Xte, Yte))
    else:
        Xtr, Ytr = prepare_XY(train_pairs, vocabs, use_current_step=use_current_step, 
                            use_hyperbolic_embeddings=False)
        Xva, Yva = prepare_XY(val_pairs, vocabs, use_current_step=use_current_step,
                            use_hyperbolic_embeddings=False)
        Xte, Yte = prepare_XY(test_pairs, vocabs, use_current_step=use_current_step,
                            use_hyperbolic_embeddings=False)
        print(f"\nTrain data shape: {Xtr.shape}")   # 100% [34972, 19733]
        print(f"\nTrain label shape: {Ytr.shape}")   # 100% [34972, 274]
        train_dataset = TensorDataset(Xtr, Ytr)
        val_dataset = TensorDataset(Xva, Yva)
        test_dataset = TensorDataset(Xte, Yte)

    # 5) Create DataLoaders for batch training
    # Custom collate function for variable-length sequences when using hyperbolic embeddings
    if use_hyperbolic_embeddings:
        def collate_fn(batch):
            """Collate function for variable-length sequences"""
            X_batch, Y_batch = zip(*batch)
            # X_batch is a list of tensors with variable shapes (num_codes_i, embedding_dim)
            # Y_batch is a list of label tensors (num_labels,)
            # Return as list of X tensors and stacked Y tensor
            return list(X_batch), torch.stack(Y_batch)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 6) Model and loss (multi-label)
    # input: (batch_size, in_dim), in_dim = historical diagnosis codes (ICD)len(diag_stoi) + historical procedure codes (Procedures)len(proc_stoi) + historical drug codes (ATC-3)len(drug_stoi)
    # output: (batch_size, out_dim), out_dim = label codes (CCS)len(y_stoi), size of label vocabulary
    
    device = torch.device('cuda')
    
    # Create model with automatic input dimension detection
    # For hyperbolic embeddings, use first sample to determine embedding_dim
    if use_hyperbolic_embeddings:
        # Xtr is a list, get embedding_dim from first sample
        sample_X = Xtr[0]  # (num_codes, embedding_dim)
        embedding_dim = sample_X.shape[1]
        # Create a dummy 3D tensor for model creation
        dummy_X = sample_X.unsqueeze(0)  # (1, num_codes, embedding_dim)
        model = create_model_from_data(model_type, dummy_X, hidden=hidden, out_dim=Ytr.size(1), 
                                      use_hyperbolic_embeddings=True, **model_kwargs)
    else:
        model = create_model_from_data(model_type, Xtr, hidden=hidden, out_dim=Ytr.size(1), 
                                      use_hyperbolic_embeddings=False, **model_kwargs)
    model = model.to(device)  # Move model to device
    
    pw = bce_pos_weight(Ytr).to(device)
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
            batch_Y = batch_Y.to(device)
            # batch_X is either a tensor (regular) or list of tensors (hyperbolic embeddings)
            if use_hyperbolic_embeddings:
                # batch_X is a list of tensors, move each to device
                batch_X = [x.to(device) for x in batch_X]
            else:
                batch_X = batch_X.to(device)
            
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
                    
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered! No improvement in {monitor_metric} for {patience} epochs.")
                    print(f"Restoring best model from epoch {ep - patience_counter}")
                    model.load_state_dict(best_model_state)
                    break

    # 8) Test set evaluation (consistent with paper: Visit-level P@k, Code-level Acc@k)
    test_metrics = evaluate_batched(model, test_loader, ks=(10, 20, 30), device=device)
    print("[TEST]", test_metrics)
    
    return model, vocabs, y_itos, test_metrics


def train_mlp_on_samples(samples, **kwargs):
    """Backward compatible MLP training function"""
    return train_diagnosis_model_on_samples(samples, model_type="mlp", **kwargs)


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
            batch_Y = batch_Y.to(device)
            # batch_X is either a tensor (regular) or list of tensors (hyperbolic embeddings)
            # Check if it's a list (hyperbolic embeddings case)
            if isinstance(batch_X, list):
                batch_X = [x.to(device) for x in batch_X]
            else:
                batch_X = batch_X.to(device)
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


def train_dialysis_model_on_samples(samples,
                                   model_type="mlp",
                                   hidden=512, lr=1e-3, wd=1e-5,
                                   epochs=10, seed=42, train_percentage=1.0,
                                   batch_size=32,
                                   early_stopping=True,
                                   patience=10,
                                   min_delta=0.001,
                                   monitor_metric='accuracy',
                                   use_gpu=True, force_cpu=False,
                                   **model_kwargs):
    """
    Train model for dialysis prediction (binary classification)
    
    Args:
        samples: Sample data from dialysis_prediction_mimic4_fn
        model_type: "mlp" or "transformer", model type
        hidden: Hidden layer dimension
        lr: Learning rate
        wd: Weight decay
        epochs: Number of training epochs
        seed: Random seed
        train_percentage: Percentage of training data to use (0.01-1.0)
        batch_size: Batch size for training
        early_stopping: Enable early stopping
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        monitor_metric: Metric to monitor for early stopping
        use_gpu: Use GPU if available
        force_cpu: Force CPU usage
        **model_kwargs: Model-specific parameters
    
    Returns:
        model: Trained model
        vocabs: Vocabulary dictionaries
        test_metrics: Test set evaluation metrics
    """
    print(f"\nStarting dialysis prediction training...")
    print(f"Total samples: {len(samples)}")
    
    # Count dialysis vs non-dialysis patients
    dialysis_count = sum(1 for s in samples if s['dialysis_label'] == 1)
    non_dialysis_count = len(samples) - dialysis_count
    print(f"Dialysis patients: {dialysis_count}, Non-dialysis patients: {non_dialysis_count}")
    
    # Build pairs for dialysis prediction
    pairs = build_dialysis_pairs(samples)
    print(f"Total pairs: {len(pairs)}")
    
    # Split by patient ID to avoid data leakage
    train_pairs, val_pairs, test_pairs = split_dialysis_by_patient(pairs, seed=seed)
    
    # Apply few-shot sampling to training data
    if train_percentage < 1.0:
        torch.manual_seed(seed)
        random.seed(seed)
        original_train_size = len(train_pairs)
        sample_size = int(original_train_size * train_percentage)
        train_pairs = random.sample(train_pairs, sample_size)
        print(f"Few-shot training: Using {len(train_pairs)}/{original_train_size} samples ({train_percentage:.1%})")
    
    # Build vocabulary
    (med_stoi, med_itos), (cond_stoi, cond_itos), (proc_stoi, proc_itos) = build_dialysis_vocab_from_pairs(train_pairs)
    vocabs = (med_stoi, cond_stoi, proc_stoi)
    
    print(f"Vocabulary sizes - Medications: {len(med_stoi)}, Conditions: {len(cond_stoi)}, Procedures: {len(proc_stoi)}")
    
    # Vectorize data
    Xtr, Ytr = prepare_dialysis_XY(train_pairs, vocabs)
    Xva, Yva = prepare_dialysis_XY(val_pairs, vocabs)
    Xte, Yte = prepare_dialysis_XY(test_pairs, vocabs)
    
    print(f"Train data shape: {Xtr.shape}")
    print(f"Train labels shape: {Ytr.shape}")
    print(f"Train labels distribution: {Ytr.sum().item()}/{len(Ytr)} positive")
    
    # Create DataLoaders
    train_dataset = TensorDataset(Xtr, Ytr)
    val_dataset = TensorDataset(Xva, Yva)
    test_dataset = TensorDataset(Xte, Yte)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup device
    if force_cpu:
        device = torch.device('cpu')
        print("Using device: cpu (forced)")
    elif use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device}")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device} (GPU not available or disabled)")
    
    # Create model for binary classification
    model = create_model(model_type, Xtr.size(1), hidden=hidden, out_dim=1, **model_kwargs)
    model = model.to(device)
    
    # Binary classification loss
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    # Training loop with early stopping
    best_metric = -float('inf')
    patience_counter = 0
    best_model_state = None
    
    for ep in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Training phase
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            logits = model(batch_X).squeeze()
            loss = criterion(logits, batch_Y)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        # Validation phase - check every epoch for early stopping
        if ep % 1 == 0:
            val_metrics = evaluate_dialysis_batched(model, val_loader, device=device)
            current_metric = val_metrics[monitor_metric]
            
            print(f"Epoch {ep}: Loss={avg_loss:.4f}, Val {monitor_metric}={current_metric:.4f}")
            
            # Early stopping logic
            if early_stopping:
                if current_metric > best_metric + min_delta:
                    best_metric = current_metric
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                    print(f"  → New best {monitor_metric}: {best_metric:.4f}")
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    model.load_state_dict(best_model_state)
                    print(f"\nEarly stopping triggered! No improvement in {monitor_metric} for {patience} epochs.")
                    print(f"Restoring best model from epoch {ep - patience_counter}")
                    break
    
    # Final evaluation on test set
    test_metrics = evaluate_dialysis_batched(model, test_loader, device=device)
    
    print(f"\nFinal test results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return model, vocabs, test_metrics


def split_dialysis_by_patient(pairs, test_size=0.2, val_size=0.1, seed=42):
    """Split dialysis prediction dataset by patient ID to avoid data leakage"""
    from sklearn.model_selection import train_test_split
    
    pid2pairs = defaultdict(list)
    for features, label in pairs:
        pid2pairs[features["patient_id"]].append((features, label))
    
    pids = list(pid2pairs.keys())
    tr_pids, te_pids = train_test_split(pids, test_size=test_size, random_state=seed)
    tr_pids, va_pids = train_test_split(tr_pids, test_size=val_size, random_state=seed)
    
    def collect(pid_list):
        out = []
        for pid in pid_list:
            out.extend(pid2pairs[pid])
        return out
    
    return collect(tr_pids), collect(va_pids), collect(te_pids)


def evaluate_dialysis_batched(model, data_loader, device):
    """Evaluate dialysis prediction model on batched data"""
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_Y in data_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            logits = model(batch_X).squeeze()
            
            all_logits.append(logits.cpu().numpy())
            all_labels.append(batch_Y.cpu().numpy())
    
    # Concatenate all batches
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    
    # Convert logits to probabilities
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    
    # Debug: Print prediction probability distribution
    print(f"\n=== Prediction Probability Distribution Debug ===")
    print(f"Prediction probabilities range: {probs.min():.4f} - {probs.max():.4f}")
    print(f"Prediction probabilities mean: {probs.mean():.4f}")
    print(f"Prediction probabilities std: {probs.std():.4f}")
    print(f"Number of predictions > 0.5: {(probs > 0.5).sum()}")
    print(f"Number of predictions > 0.3: {(probs > 0.3).sum()}")
    print(f"Number of predictions > 0.1: {(probs > 0.1).sum()}")
    
    # Debug: Print label distribution
    print(f"True positive rate: {labels.sum() / len(labels):.4f}")
    print(f"Number of positive samples: {labels.sum()}")
    print(f"Number of negative samples: {len(labels) - labels.sum()}")
    
    # Calculate binary classification metrics
    predictions = (probs > 0.5).astype(int)
    
    # Accuracy
    accuracy = np.mean(predictions == labels)
    
    # Precision, Recall, F1
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    
    print(f"Confusion Matrix:")
    print(f"  TP: {tp}, FP: {fp}")
    print(f"  FN: {fn}, TN: {tn}")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # AUC and AUPRC
    from sklearn.metrics import roc_auc_score, average_precision_score
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.5  # Default for single class
    
    # Calculate AUPRC (Area Under Precision-Recall Curve)
    auprc = average_precision_score(labels, probs)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'auprc': auprc
    }

