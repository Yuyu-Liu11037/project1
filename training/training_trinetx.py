"""
TriNetX Training module
Contains main model training functions for TriNetX dataset
"""
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import defaultdict
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score

from model.models import create_model
from util.data_processing_trinetx import (
    sort_samples_within_patient, 
    build_pairs, 
    build_vocab_from_pairs,
    prepare_XY, 
    split_by_patient,
    build_dialysis_pairs,
    build_dialysis_vocab_from_pairs,
    prepare_dialysis_XY,
    build_dialysis_vocab_from_samples,
    prepare_dialysis_XY_from_samples
)
from util.hyperbolic_conditions import ConditionsHyperbolicEmbedder
from metrics.metrics import bce_pos_weight, evaluate


def k_fold_cross_validation(samples, k_folds=5, seeds=[42, 123, 456], **train_kwargs):
    """
    Perform k-fold cross validation with multiple random seeds
    
    Args:
        samples: Sample data
        k_folds: Number of folds for cross validation
        seeds: List of random seeds to use
        **train_kwargs: Additional training parameters
    
    Returns:
        Dictionary containing aggregated results across all folds and seeds
    """
    print(f"\nStarting {k_folds}-fold cross validation with {len(seeds)} random seeds...")
    print(f"Seeds: {seeds}")
    
    all_results = []
    fold_results = []
    
    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"SEED {seed_idx + 1}/{len(seeds)}: {seed}")
        print(f"{'='*60}")
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Process samples for this seed
        samples_by_pid = sort_samples_within_patient(samples)
        
        # Build training pairs
        task = train_kwargs.get('task', 'next')
        pairs = build_pairs(samples_by_pid, task=task)
        
        if len(pairs) == 0:
            print(f"Warning: No training pairs found for seed {seed}")
            continue
        
        print(f"Built {len(pairs)} training pairs")
        
        # Build vocabulary
        vocabs = build_vocab_from_pairs(pairs)
        diag_stoi, proc_stoi, drug_stoi, y_stoi = vocabs
        
        print(f"Vocabulary sizes - Diagnoses: {len(diag_stoi)}, Procedures: {len(proc_stoi)}, Drugs: {len(drug_stoi)}, Labels: {len(y_stoi)}")
        
        # Prepare data
        use_current_step = train_kwargs.get('use_current_step', False)
        X, Y = prepare_XY(pairs, vocabs, use_current_step=use_current_step)
        
        print(f"Data shape - X: {X.shape}, Y: {Y.shape}")
        
        # Split by patient to avoid data leakage
        train_pairs, val_pairs, test_pairs = split_by_patient(pairs, seed=seed)
        
        if len(train_pairs) == 0 or len(val_pairs) == 0 or len(test_pairs) == 0:
            print(f"Warning: Insufficient data for train/val/test split for seed {seed}")
            continue
        
        print(f"Data split - Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")
        
        # Prepare train/val/test data
        X_train, Y_train = prepare_XY(train_pairs, vocabs, use_current_step=use_current_step)
        X_val, Y_val = prepare_XY(val_pairs, vocabs, use_current_step=use_current_step)
        X_test, Y_test = prepare_XY(test_pairs, vocabs, use_current_step=use_current_step)
        
        # Perform k-fold cross validation
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        
        # Combine train and val for k-fold
        X_combined = torch.cat([X_train, X_val], dim=0)
        Y_combined = torch.cat([Y_train, Y_val], dim=0)
        
        fold_results_seed = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_combined)):
            print(f"\n--- FOLD {fold + 1}/{k_folds} ---")
            
            # Split data for this fold
            X_fold_train = X_combined[train_idx]
            Y_fold_train = Y_combined[train_idx]
            X_fold_val = X_combined[val_idx]
            Y_fold_val = Y_combined[val_idx]
            
            # Train model for this fold
            fold_result = train_single_fold(
                X_fold_train, Y_fold_train, X_fold_val, Y_fold_val, X_test, Y_test,
                vocabs, fold=fold+1, **train_kwargs
            )
            
            fold_results_seed.append(fold_result)
            fold_results.append(fold_result)
            
            print(f"Fold {fold + 1} Results:")
            for metric, value in fold_result.items():
                if isinstance(value, dict):
                    print(f"  {metric}: {value['mean']:.4f} ± {value['std']:.4f}")
                else:
                    print(f"  {metric}: {value:.4f}")
        
        # Aggregate results for this seed
        seed_results = aggregate_fold_results(fold_results_seed)
        all_results.append(seed_results)
        
        print(f"\nSeed {seed} Results:")
        for metric, stats in seed_results.items():
            print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    # Aggregate results across all seeds
    final_results = aggregate_seed_results(all_results)
    
    print(f"\n{'='*60}")
    print("FINAL CROSS VALIDATION RESULTS")
    print(f"{'='*60}")
    for metric, stats in final_results.items():
        print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    return final_results, fold_results


def train_single_fold(X_train, Y_train, X_val, Y_val, X_test, Y_test, vocabs, fold=1, **train_kwargs):
    """
    Train a single fold of cross validation
    
    Args:
        X_train, Y_train: Training data
        X_val, Y_val: Validation data
        X_test, Y_test: Test data
        vocabs: Vocabulary dictionaries
        fold: Fold number
        **train_kwargs: Training parameters
    
    Returns:
        Dictionary containing test results
    """
    # Extract training parameters
    model_type = train_kwargs.get('model_type', 'mlp')
    hidden = train_kwargs.get('hidden', 512)
    lr = train_kwargs.get('lr', 1e-4)
    wd = train_kwargs.get('wd', 1e-5)
    epochs = train_kwargs.get('epochs', 100)
    batch_size = train_kwargs.get('batch_size', 256)
    early_stopping = train_kwargs.get('early_stopping', True)
    patience = train_kwargs.get('patience', 10)
    min_delta = train_kwargs.get('min_delta', 0.001)
    monitor_metric = train_kwargs.get('monitor_metric', 'Acc@10')
    use_gpu = train_kwargs.get('use_gpu', True)
    force_cpu = train_kwargs.get('force_cpu', False)
    
    # Device setup
    if force_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    
    print(f"Using device: {device}")
    
    # Move data to device
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_val = X_val.to(device)
    Y_val = Y_val.to(device)
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)
    
    # Create model
    model_kwargs = {k: v for k, v in train_kwargs.items() 
                   if k in ['p', 'num_heads', 'num_layers']}
    
    model = create_model(
        input_dim=X_train.shape[1],
        output_dim=Y_train.shape[1],
        hidden_dim=hidden,
        model_type=model_type,
        **model_kwargs
    ).to(device)
    
    # Loss and optimizer
    pos_weight = bce_pos_weight(Y_train)
    print(f"Pos weight: {pos_weight}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    # Data loaders
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val, Y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    test_dataset = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Training loop
    best_val_score = 0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            if conditions_embedder is not None and model_type == 'transformer':
                batch_mask = torch.ones(batch_X.size(0), batch_X.size(1)).to(device)
            else:
                batch_mask = None
            outputs = model(batch_X, batch_mask) if batch_mask is not None else model(batch_X)
            loss = criterion(outputs.squeeze(), batch_Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                if conditions_embedder is not None and model_type == 'transformer':
                    batch_mask = torch.ones(batch_X.size(0), batch_X.size(1)).to(device)
                else:
                    batch_mask = None
                outputs = model(batch_X, batch_mask) if batch_mask is not None else model(batch_X)
                loss = criterion(outputs.squeeze(), batch_Y)
                val_loss += loss.item()
                
                val_preds.append(torch.sigmoid(outputs.squeeze()))
                val_targets.append(batch_Y)
        
        val_loss /= len(val_loader)
        val_preds = torch.cat(val_preds, dim=0)
        val_targets = torch.cat(val_targets, dim=0)
        
        # Evaluate validation
        val_metrics = evaluate(val_preds, val_targets)
        val_score = val_metrics.get(monitor_metric, 0)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val {monitor_metric}: {val_score:.4f}")
        
        # Early stopping
        if early_stopping:
            if val_score > best_val_score + min_delta:
                best_val_score = val_score
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation on test set
    model.eval()
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            if conditions_embedder is not None and model_type == 'transformer':
                batch_mask = torch.ones(batch_X.size(0), batch_X.size(1)).to(device)
            else:
                batch_mask = None
            outputs = model(batch_X, batch_mask) if batch_mask is not None else model(batch_X)
            test_preds.append(torch.sigmoid(outputs.squeeze()))
            test_targets.append(batch_Y)
    
    test_preds = torch.cat(test_preds, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    
    test_metrics = evaluate(test_preds, test_targets)
    
    return test_metrics


def train_diagnosis_model_on_samples(samples, model_type='mlp', task='next', use_current_step=False, 
                          hidden=512, lr=1e-4, wd=1e-5, epochs=100, seed=42,
                          train_percentage=1.0, batch_size=256, early_stopping=True,
                          patience=10, min_delta=0.001, monitor_metric='Acc@10',
                          use_gpu=True, force_cpu=False, **model_kwargs):
    """
    Train model on samples for diagnosis prediction
    
    Args:
        samples: List of sample dictionaries
        model_type: Type of model ('mlp' or 'transformer')
        task: Prediction task ('current' or 'next')
        use_current_step: Whether to use current step information
        hidden: Hidden layer dimension
        lr: Learning rate
        wd: Weight decay
        epochs: Number of training epochs
        seed: Random seed
        train_percentage: Percentage of training data to use
        batch_size: Batch size
        early_stopping: Whether to use early stopping
        patience: Patience for early stopping
        min_delta: Minimum change for early stopping
        monitor_metric: Metric to monitor for early stopping
        use_gpu: Whether to use GPU
        force_cpu: Whether to force CPU usage
        **model_kwargs: Additional model parameters
    
    Returns:
        Tuple of (model, vocabs, y_itos, test_metrics)
    """
    print(f"\nTraining {model_type.upper()} model for diagnosis prediction...")
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Process samples
    samples_by_pid = sort_samples_within_patient(samples)
    
    # Build training pairs
    pairs = build_pairs(samples_by_pid, task=task)
    print(f"Built {len(pairs)} training pairs")
    
    if len(pairs) == 0:
        raise ValueError("No training pairs found")
    
    # Build vocabulary
    vocabs = build_vocab_from_pairs(pairs)
    diag_stoi, proc_stoi, drug_stoi, y_stoi = vocabs
    
    print(f"Vocabulary sizes - Diagnoses: {len(diag_stoi)}, Procedures: {len(proc_stoi)}, Drugs: {len(drug_stoi)}, Labels: {len(y_stoi)}")
    
    # Prepare data
    X, Y = prepare_XY(pairs, vocabs, use_current_step=use_current_step)
    print(f"Data shape - X: {X.shape}, Y: {Y.shape}")
    
    # Split by patient to avoid data leakage
    train_pairs, val_pairs, test_pairs = split_by_patient(pairs, seed=seed)
    
    print(f"Data split - Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")
    
    # Subsample training data if requested
    if train_percentage < 1.0:
        n_train = int(len(train_pairs) * train_percentage)
        train_pairs = train_pairs[:n_train]
        print(f"Subsampled training data to {len(train_pairs)} pairs ({train_percentage:.1%})")
    
    # Prepare train/val/test data
    X_train, Y_train = prepare_XY(train_pairs, vocabs, use_current_step=use_current_step)
    X_val, Y_val = prepare_XY(val_pairs, vocabs, use_current_step=use_current_step)
    X_test, Y_test = prepare_XY(test_pairs, vocabs, use_current_step=use_current_step)
    
    # Device setup
    if force_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    
    print(f"Using device: {device}")
    
    # Move data to device
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_val = X_val.to(device)
    Y_val = Y_val.to(device)
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)
    
    # Create model
    model = create_model(
        input_dim=X_train.shape[1],
        output_dim=Y_train.shape[1],
        hidden_dim=hidden,
        model_type=model_type,
        **model_kwargs
    ).to(device)
    
    # Loss and optimizer
    pos_weight = bce_pos_weight(Y_train)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    # Data loaders
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val, Y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    test_dataset = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Training loop
    best_val_score = 0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            if conditions_embedder is not None and model_type == 'transformer':
                batch_mask = torch.ones(batch_X.size(0), batch_X.size(1)).to(device)
            else:
                batch_mask = None
            outputs = model(batch_X, batch_mask) if batch_mask is not None else model(batch_X)
            loss = criterion(outputs.squeeze(), batch_Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                if conditions_embedder is not None and model_type == 'transformer':
                    batch_mask = torch.ones(batch_X.size(0), batch_X.size(1)).to(device)
                else:
                    batch_mask = None
                outputs = model(batch_X, batch_mask) if batch_mask is not None else model(batch_X)
                loss = criterion(outputs.squeeze(), batch_Y)
                val_loss += loss.item()
                
                val_preds.append(torch.sigmoid(outputs.squeeze()))
                val_targets.append(batch_Y)
        
        val_loss /= len(val_loader)
        val_preds = torch.cat(val_preds, dim=0)
        val_targets = torch.cat(val_targets, dim=0)
        
        # Evaluate validation
        val_metrics = evaluate(val_preds, val_targets)
        val_score = val_metrics.get(monitor_metric, 0)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val {monitor_metric}: {val_score:.4f}")
        
        # Early stopping
        if early_stopping:
            if val_score > best_val_score + min_delta:
                best_val_score = val_score
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation on test set
    model.eval()
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            if conditions_embedder is not None and model_type == 'transformer':
                batch_mask = torch.ones(batch_X.size(0), batch_X.size(1)).to(device)
            else:
                batch_mask = None
            outputs = model(batch_X, batch_mask) if batch_mask is not None else model(batch_X)
            test_preds.append(torch.sigmoid(outputs.squeeze()))
            test_targets.append(batch_Y)
    
    test_preds = torch.cat(test_preds, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    
    test_metrics = evaluate(test_preds, test_targets)
    
    # Create y_itos for compatibility
    y_itos = [code for code, _ in sorted(y_stoi.items(), key=lambda x: x[1])]
    
    return model, vocabs, y_itos, test_metrics


def train_dialysis_model_on_samples(samples, model_type='mlp', hidden=512, lr=1e-4, wd=1e-5, 
                                   epochs=100, seed=42, train_percentage=1.0, batch_size=256,
                                   early_stopping=True, patience=10, min_delta=0.001,
                                   monitor_metric='auprc', use_gpu=True, force_cpu=False, 
                                   use_optimal_threshold=True, conditions_embedder=None, 
                                   max_seq_length=None, **model_kwargs):
    """
    Train model on samples for dialysis prediction
    
    Args:
        samples: List of sample dictionaries
        model_type: Type of model ('mlp' or 'transformer')
        hidden: Hidden layer dimension
        lr: Learning rate
        wd: Weight decay
        epochs: Number of training epochs
        seed: Random seed
        train_percentage: Percentage of training data to use
        batch_size: Batch size
        early_stopping: Whether to use early stopping
        patience: Patience for early stopping
        min_delta: Minimum change for early stopping
        monitor_metric: Metric to monitor for early stopping ('auprc', 'auc', 'accuracy')
        use_gpu: Whether to use GPU
        force_cpu: Whether to force CPU usage
        use_optimal_threshold: Whether to use F1-optimal threshold for final evaluation
        conditions_embedder: Pre-trained ConditionsHyperbolicEmbedder instance (optional)
        **model_kwargs: Additional model parameters
    
    Returns:
        Tuple of (model, vocabs, test_metrics)
    """
    print(f"\nTraining {model_type.upper()} model for dialysis prediction...")
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Check if we have samples
    if len(samples) == 0:
        raise ValueError("No dialysis prediction samples found")
    
    print(f"Processing {len(samples)} dialysis prediction samples")
    
    # Build vocabulary directly from samples (more efficient)
    (med_stoi, med_itos), (cond_stoi, cond_itos), (proc_stoi, proc_itos) = build_dialysis_vocab_from_samples(samples)
    vocabs = (med_stoi, cond_stoi, proc_stoi)
    
    print(f"Vocabulary sizes - Medications: {len(med_stoi)}, Conditions: {len(cond_stoi)}, Procedures: {len(proc_stoi)}")
    
    # Use provided hyperbolic embedder or None
    if conditions_embedder is not None:
        print(f"Using pre-trained hyperbolic embeddings (dim: {conditions_embedder.get_embedding_dim()})")
    else:
        print("Using one-hot encoding for conditions")
    
    # Prepare data directly from samples (more efficient)
    X, Y = prepare_dialysis_XY_from_samples(samples, vocabs, conditions_embedder, max_seq_length)
    print(f"Max sequence length: {max_seq_length}")
    
    # Handle variable-length sequences for transformer models
    if max_seq_length is None and isinstance(X, list):
        # For transformer models, we need to determine a suitable max_seq_length
        if model_type == 'transformer':
            from util.data_processing_trinetx import suggest_max_seq_length
            suggested_length = suggest_max_seq_length(samples, percentile=95)
            print(f"Auto-determining max_seq_length for transformer: {suggested_length}")
            
            # Re-prepare data with padding
            X, Y = prepare_dialysis_XY_from_samples(samples, vocabs, conditions_embedder, suggested_length)
        else:
            # For non-transformer models, convert list to stacked tensor
            # This will pad sequences to the maximum length in the batch
            max_len = max(x.size(0) for x in X) if X else 0
            if max_len > 0:
                padded_X = []
                for x in X:
                    if x.size(0) < max_len:
                        padding = torch.zeros(max_len - x.size(0), x.size(1))
                        x_padded = torch.cat([x, padding], dim=0)
                    else:
                        x_padded = x
                    padded_X.append(x_padded)
                X = torch.stack(padded_X)
            else:
                # Handle empty case
                X = torch.zeros(len(X), 0, 20)  # Assuming embedding_dim=20
    
    # Split data
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=seed)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=seed)
    
    print(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Subsample training data if requested
    if train_percentage < 1.0:
        n_train = int(len(X_train) * train_percentage)
        X_train = X_train[:n_train]
        Y_train = Y_train[:n_train]
        print(f"Subsampled training data to {len(X_train)} samples ({train_percentage:.1%})")
    
    # Device setup
    if force_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    
    print(f"Using device: {device}")
    
    # Move data to device
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_val = X_val.to(device)
    Y_val = Y_val.to(device)
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)
    
    # Create main model (binary classification)
    if conditions_embedder is not None and model_type == 'transformer':
        # Use SequentialTransformerModel for hyperbolic embeddings
        model = create_model(
            model_type='sequential_transformer',
            in_dim=X_train.shape[2],  # embedding_dim
            hidden=hidden,
            out_dim=1,  # Binary classification
            **model_kwargs
        ).to(device)
    else:
        # Use regular model for other cases
        model = create_model(
            model_type=model_type,
            in_dim=X_train.shape[1],
            hidden=hidden,
            out_dim=1,  # Binary classification
            **model_kwargs
        ).to(device)
    
    # Create attention masks for SequentialTransformerModel
    attention_masks = None
    if conditions_embedder is not None and model_type == 'transformer':
        # Create attention masks (1 for real tokens, 0 for padding)
        attention_masks = torch.ones(X_train.size(0), X_train.size(1)).to(device)
        # Note: For now, we assume all tokens are real (no padding)
        # In practice, you might want to create proper masks based on actual sequence lengths
        print(f"Created attention masks: {attention_masks.shape}")
    
    # Loss and optimizer
    pos_weight = bce_pos_weight(Y_train)
    print(f"Pos weight: {pos_weight}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    # Data loaders
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val, Y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    test_dataset = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initial evaluation before training
    print("\n=== Initial Model Evaluation (Before Training) ===")
    model.eval()
    initial_val_preds = []
    initial_val_targets = []
    
    with torch.no_grad():
        for batch_X, batch_Y in val_loader:
            outputs = model(batch_X)
            initial_val_preds.append(torch.sigmoid(outputs.squeeze()))
            initial_val_targets.append(batch_Y)
    
    initial_val_preds = torch.cat(initial_val_preds, dim=0)
    initial_val_targets = torch.cat(initial_val_targets, dim=0)
    
    # Calculate initial metrics
    initial_val_preds_binary = (initial_val_preds > 0.5).float()
    initial_val_accuracy = (initial_val_preds_binary == initial_val_targets).float().mean().item()
    
    initial_val_auprc = average_precision_score(initial_val_targets.cpu().numpy(), initial_val_preds.cpu().numpy())
    initial_val_auc = roc_auc_score(initial_val_targets.cpu().numpy(), initial_val_preds.cpu().numpy())
    
    print(f"Initial Val Accuracy: {initial_val_accuracy:.4f}, Initial Val AUPRC: {initial_val_auprc:.4f}, Initial Val AUC: {initial_val_auc:.4f}")
    
    # Training loop
    best_val_score = 0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            if conditions_embedder is not None and model_type == 'transformer':
                batch_mask = torch.ones(batch_X.size(0), batch_X.size(1)).to(device)
            else:
                batch_mask = None
            # print(f"Batch X shape: {batch_X.shape}")
            outputs = model(batch_X, batch_mask) if batch_mask is not None else model(batch_X)
            # print(f"Outputs shape: {outputs.shape}")
            loss = criterion(outputs.squeeze(), batch_Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                if conditions_embedder is not None and model_type == 'transformer':
                    batch_mask = torch.ones(batch_X.size(0), batch_X.size(1)).to(device)
                else:
                    batch_mask = None
                outputs = model(batch_X, batch_mask) if batch_mask is not None else model(batch_X)
                loss = criterion(outputs.squeeze(), batch_Y)
                val_loss += loss.item()
                
                val_preds.append(torch.sigmoid(outputs.squeeze()))
                val_targets.append(batch_Y)
        
        val_loss /= len(val_loader)
        val_preds = torch.cat(val_preds, dim=0)
        val_targets = torch.cat(val_targets, dim=0)
        
        # Calculate metrics for dialysis prediction
        val_preds_binary = (val_preds > 0.5).float()
        val_accuracy = (val_preds_binary == val_targets).float().mean().item()
        
        # Calculate AUPRC
        val_auprc = average_precision_score(val_targets.cpu().numpy(), val_preds.cpu().numpy())
        
        # Calculate AUC
        val_auc = roc_auc_score(val_targets.cpu().numpy(), val_preds.cpu().numpy())
        
        # Select metric based on monitor_metric parameter
        if monitor_metric == 'auprc':
            val_score = val_auprc
        elif monitor_metric == 'auc':
            val_score = val_auc
        elif monitor_metric == 'accuracy':
            val_score = val_accuracy
        else:
            val_score = val_accuracy  # default to accuracy
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val AUPRC: {val_auprc:.4f}, Val AUC: {val_auc:.4f}")
        
        # Early stopping
        if early_stopping:
            if val_score > best_val_score + min_delta:
                best_val_score = val_score
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation on test set
    model.eval()
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            if conditions_embedder is not None and model_type == 'transformer':
                batch_mask = torch.ones(batch_X.size(0), batch_X.size(1)).to(device)
            else:
                batch_mask = None
            outputs = model(batch_X, batch_mask) if batch_mask is not None else model(batch_X)
            test_preds.append(torch.sigmoid(outputs.squeeze()))
            test_targets.append(batch_Y)
    
    test_preds = torch.cat(test_preds, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    
    # Convert to numpy for analysis
    test_preds_np = test_preds.cpu().numpy()
    test_targets_np = test_targets.cpu().numpy()
    
    # Debug: Print prediction probability distribution
    print(f"\n=== Prediction Probability Distribution Debug ===")
    print(f"Prediction probabilities range: {test_preds_np.min():.4f} - {test_preds_np.max():.4f}")
    print(f"Prediction probabilities mean: {test_preds_np.mean():.4f}")
    print(f"Prediction probabilities std: {test_preds_np.std():.4f}")
    print(f"Number of predictions > 0.5: {(test_preds_np > 0.5).sum()}")
    print(f"Number of predictions > 0.3: {(test_preds_np > 0.3).sum()}")
    print(f"Number of predictions > 0.1: {(test_preds_np > 0.1).sum()}")
    
    # Debug: Print label distribution
    print(f"True positive rate: {test_targets_np.sum() / len(test_targets_np):.4f}")
    print(f"Number of positive samples: {test_targets_np.sum()}")
    print(f"Number of negative samples: {len(test_targets_np) - test_targets_np.sum()}")
    
    # Calculate additional metrics
    
    # Use fixed threshold of 0.5
    optimal_threshold = 0.5
    
    print(f"\n=== Threshold Analysis ===")
    print(f"Using fixed threshold (0.5): F1 = {f1_score(test_targets_np, test_preds_np > 0.5):.4f}")
    
    # Calculate AUPRC (Area Under Precision-Recall Curve)
    auprc = average_precision_score(test_targets_np, test_preds_np)
    
    # Calculate metrics with threshold 0.5
    test_preds_binary = (test_preds > 0.5).float()
    
    # Calculate confusion matrix components
    tp = ((test_preds_np > 0.5) & (test_targets_np == 1)).sum()
    fp = ((test_preds_np > 0.5) & (test_targets_np == 0)).sum()
    fn = ((test_preds_np <= 0.5) & (test_targets_np == 1)).sum()
    tn = ((test_preds_np <= 0.5) & (test_targets_np == 0)).sum()
    
    print(f"\nConfusion Matrix (threshold 0.5):")
    print(f"  TP: {tp}, FP: {fp}")
    print(f"  FN: {fn}, TN: {tn}")
    
    # Calculate test accuracy
    test_accuracy = (test_preds_binary == test_targets).float().mean().item()
    
    test_metrics = {
        'accuracy': test_accuracy,
        'precision': precision_score(test_targets_np, test_preds_binary.cpu().numpy()),
        'recall': recall_score(test_targets_np, test_preds_binary.cpu().numpy()),
        'f1': f1_score(test_targets_np, test_preds_binary.cpu().numpy()),
        'auc': roc_auc_score(test_targets_np, test_preds_np),
        'auprc': auprc,
        'threshold': 0.5
    }
    
    return model, vocabs, test_metrics


def aggregate_fold_results(fold_results):
    """Aggregate results across folds for a single seed"""
    if not fold_results:
        return {}
    
    # Get all metric names
    all_metrics = set()
    for result in fold_results:
        all_metrics.update(result.keys())
    
    aggregated = {}
    for metric in all_metrics:
        values = [result.get(metric, 0) for result in fold_results]
        aggregated[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }
    
    return aggregated


def aggregate_seed_results(seed_results):
    """Aggregate results across seeds"""
    if not seed_results:
        return {}
    
    # Get all metric names
    all_metrics = set()
    for result in seed_results:
        all_metrics.update(result.keys())
    
    final_aggregated = {}
    for metric in all_metrics:
        # Collect all values across seeds and folds
        all_values = []
        for seed_result in seed_results:
            if metric in seed_result:
                all_values.extend(seed_result[metric]['values'])
        
        if all_values:
            final_aggregated[metric] = {
                'mean': np.mean(all_values),
                'std': np.std(all_values),
                'count': len(all_values)
            }
    
    return final_aggregated
