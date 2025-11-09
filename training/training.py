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
    split_by_patient,
)
from eval_embedding import load_pkl_file
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
                           use_current_step=False, # Admission prediction(False) or discharge prediction(True)
                           hidden=512, lr=1e-3, wd=1e-5,
                           epochs=10, seed=42, train_percentage=1.0,
                           batch_size=32,         # Batch size for training
                           early_stopping=True,   # Enable early stopping
                           patience=10,           # Number of epochs to wait before stopping
                           min_delta=0.0001,      # Minimum change to qualify as improvement
                           monitor_metric='Acc@10', # Metric to monitor for early stopping
                           use_gpu=True, force_cpu=False,  # GPU control
                           use_hyperbolic_embeddings=False,  # Use hyperbolic embeddings
                           embedding_file="hyperbolic_embeddings.pkl",  # Path to embeddings file
                           max_seq_length=300,    # Maximum sequence length for sequential data
                           use_lr_scheduler=True,  # Enable learning rate scheduler
                           lr_scheduler_factor=0.5,  # Factor by which learning rate will be reduced
                           lr_scheduler_patience=25,  # Number of epochs with no improvement after which learning rate will be reduced
                           lr_scheduler_min_lr=1e-6,  # Minimum learning rate
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
        use_lr_scheduler: Enable learning rate scheduler (default: True)
        lr_scheduler_factor: Factor by which learning rate will be reduced (default: 0.5)
        lr_scheduler_patience: Number of epochs with no improvement after which learning rate will be reduced (default: 5)
        lr_scheduler_min_lr: Minimum learning rate (default: 1e-6)
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
    by_pid = sort_samples_within_patient(samples)   # defaultdict(list), {"patient_id": [sample1, sample2, ...]}
    # print(f"\nBy pid: {by_pid['10001401']}")
    # build_pairs has issues... We should use all past visit records of the patient to predict the next diagnosis, not the previous visit
    # Never mind, the cond_hist field contains all previous visits
    pairs = build_pairs(by_pid)  # [(sample, sample["conditions"])]
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
    (diag_stoi, diag_itos), (proc_stoi,_), (drug_stoi,_), (y_stoi, y_itos) = build_vocab_from_pairs(train_pairs)
    vocab_size = len(diag_stoi)
    vocabs = (diag_stoi, proc_stoi, drug_stoi, y_stoi)

    # 4) Dataset creation
    class DiagnosisDataset(Dataset):
        """Custom Dataset for diagnosis prediction"""
        def __init__(self, pairs, diag_stoi, y_stoi, max_seq_len):
            """
            Args:
                pairs: List of (sample, y_codes_list) tuples
                diag_stoi: Dictionary mapping diagnosis codes to integers
                y_stoi: Dictionary mapping label codes to integers
                max_seq_len: Maximum sequence length for padding
            """
            self.pairs = pairs
            self.diag_stoi = diag_stoi
            self.y_stoi = y_stoi
            self.max_seq_len = max_seq_len
        
        def __len__(self):
            return len(self.pairs)
        
        def __getitem__(self, idx):
            sample, y_codes_list = self.pairs[idx]
            
            # Extract cond_hist from sample (list of lists)
            cond_hist = sample.get("cond_hist", [])
            
            # Flatten cond_hist into a single list of codes
            # Skip the last empty visit (to prevent leakage)
            flattened_codes = []
            if len(cond_hist) > 1:
                # Skip last empty visit
                for visit_codes in cond_hist[:-1]:
                    flattened_codes.extend(visit_codes)
            elif len(cond_hist) == 1:
                # Only one visit (should be empty, but handle it anyway)
                flattened_codes.extend(cond_hist[0])
            
            # Convert codes to integers using diag_stoi
            condition_tokens = []
            for code in flattened_codes:
                if code in self.diag_stoi:
                    condition_tokens.append(self.diag_stoi[code])
            
            # Pad to max_seq_len with 0
            if len(condition_tokens) > self.max_seq_len:
                condition_tokens = condition_tokens[:self.max_seq_len]
            else:
                condition_tokens = condition_tokens + [0] * (self.max_seq_len - len(condition_tokens))
            
            # Create multi-hot vector from y_codes_list using y_stoi
            label = torch.zeros(len(self.y_stoi), dtype=torch.float32)
            for code in y_codes_list:
                if code in self.y_stoi:
                    label[self.y_stoi[code]] = 1.0
            
            return torch.tensor(condition_tokens, dtype=torch.long), label
    
    train_dataset = DiagnosisDataset(train_pairs, diag_stoi, y_stoi, max_seq_length)
    val_dataset = DiagnosisDataset(val_pairs, diag_stoi, y_stoi, max_seq_length)
    test_dataset = DiagnosisDataset(test_pairs, diag_stoi, y_stoi, max_seq_length)

    # 5) Create DataLoaders for batch training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 6) Model and loss (multi-label)
    device = torch.device('cuda')
    out_dim = len(y_stoi)  # Number of label classes
    save_data = load_pkl_file(embedding_file) if use_hyperbolic_embeddings else None
    # Pass diag_itos to model for converting token IDs back to code strings when using hyperbolic embeddings
    model = create_model(model_type, vocab_size=vocab_size, hidden=hidden, out_dim=out_dim, 
                        save_data=save_data, diag_itos=diag_itos if use_hyperbolic_embeddings else None, **model_kwargs)
    model = model.to(device)  # Move model to device
    
    # Calculate positive weights from training dataset
    # Collect all labels from training dataset
    all_labels = []
    for _, label in train_dataset:
        all_labels.append(label)
    Ytr = torch.stack(all_labels)
    pw = bce_pos_weight(Ytr).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='max', factor=lr_scheduler_factor, 
            patience=lr_scheduler_patience, min_lr=lr_scheduler_min_lr, verbose=True
        )

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
            batch_X = batch_X.to(device)
            
            # print(batch_X.shape, batch_Y.shape)
            # print(batch_X[0])
            # print(batch_Y[0])
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
            
            # Update learning rate scheduler
            scheduler.step(current_metric)
            current_lr = opt.param_groups[0]['lr']
            print(f"Epoch {ep:02d} | avg_loss={avg_loss:.4f} | lr={current_lr:.2e} | "
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

    # 8) Test set evaluation (Visit-level P@k, Code-level Acc@k)
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
