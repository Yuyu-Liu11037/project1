"""
Diagnosis prediction main program
Using TriNetX dataset for diagnosis prediction task
"""
import argparse
import torch
import numpy as np
import warnings

from util.data_processing_trinetx import (
    load_trinetx_data, 
    process_trinetx_patients, 
    process_trinetx_dialysis_patients,
    sort_samples_within_patient,
    build_pairs,
    build_vocab_from_pairs,
    prepare_XY,
    split_by_patient,
    build_dialysis_pairs,
    build_dialysis_vocab_from_pairs,
    prepare_dialysis_XY
)
from training.training_trinetx import train_diagnosis_model_on_samples, k_fold_cross_validation, train_dialysis_model_on_samples
from train_hyperbolic_embeddings import load_embeddings

warnings.filterwarnings('ignore')


def save_cross_validation_results(final_results, fold_results, args):
    """
    Save cross validation results to files
    
    Args:
        final_results: Aggregated results across all seeds
        fold_results: Individual fold results
        args: Command line arguments
    """
    import json
    import pandas as pd
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save final aggregated results
    results_summary = {
        'model': args.model,
        'task': args.task,
        'k_folds': args.k_folds,
        'seeds': args.seed_range,
        'timestamp': timestamp,
        'final_results': final_results
    }
    
    with open(f'cv_results_{args.model}_{timestamp}.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save detailed fold results as CSV
    fold_data = []
    for i, fold_result in enumerate(fold_results):
        row = {'fold': i + 1}
        row.update(fold_result)
        fold_data.append(row)
    
    df = pd.DataFrame(fold_data)
    df.to_csv(f'cv_fold_results_{args.model}_{timestamp}.csv', index=False)
    
    print(f"\nResults saved to:")
    print(f"  - cv_results_{args.model}_{timestamp}.json")
    print(f"  - cv_fold_results_{args.model}_{timestamp}.csv")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Diagnosis prediction model training')
    
    # Model selection
    parser.add_argument('--model', type=str, default='transformer', 
                       choices=['mlp', 'transformer'],
                       help='Model type: mlp or transformer (default: mlp)')
    
    # Task selection
    parser.add_argument('--task_type', type=str, default='dialysis',
                       choices=['diagnosis', 'dialysis'],
                       help='Task type: diagnosis prediction or dialysis prediction (default: diagnosis)')
    
    # Training parameters
    parser.add_argument('--task', type=str, default='next',
                       choices=['current', 'next'],
                       help='Prediction task: current or next (default: next)')
    parser.add_argument('--use_current_step', action='store_true',
                       help='Whether to use current step information (default: False)')
    parser.add_argument('--hidden', type=int, default=512,
                       help='Hidden layer dimension (default: 512)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--wd', type=float, default=1e-5,
                       help='Weight decay (default: 1e-5)')
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--train_percentage', type=float, default=0.05,
                       help='Percentage of training data to use for few-shot training (0.01-1.0, default: 1.0)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for training (default: 256)')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                       help='Use GPU for training if available (default: True)')
    parser.add_argument('--force_cpu', action='store_true',
                       help='Force CPU usage even if GPU is available (default: False)')
    
    # Cross validation parameters
    parser.add_argument('--k_folds', type=int, default=5,
                       help='Number of folds for k-fold cross validation (default: 5)')
    parser.add_argument('--num_seeds', type=int, default=3,
                       help='Number of different random seeds to use (default: 3)')
    parser.add_argument('--seed_range', type=str, default='42,123,456',
                       help='Comma-separated list of random seeds to use (default: 42,123,456)')
    parser.add_argument('--use_cross_validation', action='store_true',
                       help='Enable k-fold cross validation with multiple seeds (default: False)')
    
    # Early stopping parameters
    parser.add_argument('--early_stopping', action='store_true', default=True,
                       help='Enable early stopping (default: True)')
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
    parser.add_argument('--max_seq_length', type=int, default=None,
                       help='Maximum sequence length for padding (default: None, auto-determined)')
    
    # Data path
    parser.add_argument('--data_path', type=str, 
                       default="/data/yuyu/data/trinetx_data",
                       help='TriNetX data path')
    
    # Hyperbolic embeddings
    parser.add_argument('--use_hyperbolic_embeddings', action='store_true',
                       help='Use pre-trained hyperbolic embeddings for conditions (default: False)')
    parser.add_argument('--embedding_file', type=str, default='hyperbolic_embeddings.pkl',
                       help='Path to pre-trained hyperbolic embeddings file (default: hyperbolic_embeddings.pkl)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Validate train_percentage argument
    if not 0.01 <= args.train_percentage <= 1.0:
        raise ValueError("train_percentage must be between 0.01 and 1.0")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Using model: {args.model}")
    print(f"Task type: {args.task_type}")
    if args.task_type == 'diagnosis':
        print(f"Prediction task: {args.task}")
    print(f"Hidden layer dimension: {args.hidden}")
    print(f"Learning rate: {args.lr}")
    print(f"Training epochs: {args.epochs}")
    print(f"Training data percentage: {args.train_percentage:.1%}")
    print(f"Batch size: {args.batch_size}")
    print(f"Early stopping: {args.early_stopping}")
    if args.early_stopping:
        # Use appropriate monitor metric based on task type
        actual_monitor = 'auprc' if args.task_type == 'dialysis' else args.monitor_metric
        print(f"  Patience: {args.patience}, Min delta: {args.min_delta}, Monitor: {actual_monitor}")
    
    if args.model == 'transformer':
        print(f"Transformer parameters - attention heads: {args.num_heads}, layers: {args.num_layers}")
    
    # Cross validation settings
    if args.use_cross_validation:
        print(f"Cross validation: {args.k_folds}-fold with {args.num_seeds} seeds")
        print(f"Seeds: {args.seed_range}")
    else:
        print(f"Single training run with seed: {args.seed}")
    
    print(f"\nLoading TriNetX dataset for {args.task_type} prediction...")
    
    # Load TriNetX data
    trinetx_data = load_trinetx_data(args.data_path)
    
    # Process data based on task type
    if args.task_type == 'diagnosis':
        samples = process_trinetx_patients(trinetx_data)
    elif args.task_type == 'dialysis':
        samples = process_trinetx_dialysis_patients(trinetx_data)
        # for i in range(10):
        #     print(samples[i])
        # exit()
    else:
        raise ValueError(f"Unknown task_type: {args.task_type}")
    
    print(f"Loaded {len(samples)} samples for {args.task_type} prediction")

    # Load hyperbolic embeddings if requested
    conditions_embedder = None
    if args.task_type == 'dialysis' and args.use_hyperbolic_embeddings:
        try:
            conditions_embedder = load_embeddings(args.embedding_file)
        except FileNotFoundError:
            print(f"Warning: Embedding file {args.embedding_file} not found. Using one-hot encoding instead.")
            conditions_embedder = None
        except Exception as e:
            print(f"Warning: Error loading embeddings from {args.embedding_file}: {e}. Using one-hot encoding instead.")
            conditions_embedder = None

    # Prepare model parameters
    model_kwargs = {
        'p': args.dropout,
    }
    
    if args.model == 'transformer':
        model_kwargs.update({
            'num_heads': args.num_heads,
            'num_layers': args.num_layers,
        })
    
    if args.use_cross_validation:
        # Parse seed range
        seeds = [int(s.strip()) for s in args.seed_range.split(',')]
        
        print(f"\nStarting {args.k_folds}-fold cross validation with {len(seeds)} seeds...")
        final_results, fold_results = k_fold_cross_validation(
            samples,
            k_folds=args.k_folds,
            seeds=seeds,
            model_type=args.model,
            task=args.task,
            use_current_step=args.use_current_step,
            hidden=args.hidden,
            lr=args.lr,
            wd=args.wd,
            epochs=args.epochs,
            train_percentage=args.train_percentage,
            batch_size=args.batch_size,
            early_stopping=args.early_stopping,
            patience=args.patience,
            min_delta=args.min_delta,
            monitor_metric=args.monitor_metric,
            use_gpu=args.use_gpu,
            force_cpu=args.force_cpu,
            **model_kwargs
        )
        
        print(f"\n[DONE] Cross validation results for {args.model.upper()} model:")
        for metric, stats in final_results.items():
            print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # Save detailed results
        save_cross_validation_results(final_results, fold_results, args)
        
    else:
        print(f"\nStarting single training run for {args.model} model...")
        
        if args.task_type == 'diagnosis':
            model, vocabs, y_itos, test_metrics = train_diagnosis_model_on_samples(
                samples,
                model_type=args.model,
                task=args.task,
                use_current_step=args.use_current_step,
                hidden=args.hidden,
                lr=args.lr,
                wd=args.wd,
                epochs=args.epochs,
                seed=args.seed,
                train_percentage=args.train_percentage,
                batch_size=args.batch_size,
                early_stopping=args.early_stopping,
                patience=args.patience,
                min_delta=args.min_delta,
                monitor_metric=args.monitor_metric,
                use_gpu=args.use_gpu,
                force_cpu=args.force_cpu,
                **model_kwargs
            )
        elif args.task_type == 'dialysis':
            model, vocabs, test_metrics = train_dialysis_model_on_samples(
                samples,
                model_type=args.model,
                hidden=args.hidden,
                lr=args.lr,
                wd=args.wd,
                epochs=args.epochs,
                seed=args.seed,
                train_percentage=args.train_percentage,
                batch_size=args.batch_size,
                early_stopping=args.early_stopping,
                patience=args.patience,
                min_delta=args.min_delta,
                monitor_metric=actual_monitor,
                use_gpu=args.use_gpu,
                force_cpu=args.force_cpu,
                conditions_embedder=conditions_embedder,
                max_seq_length=args.max_seq_length,
                **model_kwargs
            )
        else:
            raise ValueError(f"Unknown task_type: {args.task_type}")
        
        print(f"\n[DONE] {args.model.upper()} model test results:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")