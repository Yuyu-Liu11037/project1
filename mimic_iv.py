"""
Diagnosis prediction main program
Using MIMIC-IV dataset for diagnosis prediction task
"""
import argparse
import torch
import numpy as np
import warnings
from pyhealth.datasets import MIMIC4Dataset

from util.data_processing import diag_prediction_mimic4_fn, dialysis_prediction_mimic4_fn, preprocess_conditions_codes, GEMMapper
from training.training import train_model_on_samples, train_dialysis_model_on_samples

warnings.filterwarnings('ignore')


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Diagnosis prediction model training')
    
    # Model selection
    parser.add_argument('--model', type=str, default='transformer', 
                       choices=['mlp', 'transformer'],
                       help='Model type: mlp or transformer (default: mlp)')
    
    # Task selection
    parser.add_argument('--task_type', type=str, default='diagnosis',
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
    
    # Hyperbolic embeddings parameters
    parser.add_argument('--hyperbolic_embeddings_file', type=str, default="hyperbolic_embeddings.pkl",
                       help='Path to hyperbolic embeddings file (optional)')
    parser.add_argument('--max_seq_length', type=int, default=None,
                       help='Maximum sequence length for transformer (optional)')
    parser.add_argument('--use_riemannian_embeddings', action='store_true',
                       help='Use Riemannian optimization for hyperbolic embeddings (default: False)')
    parser.add_argument('--riemannian_optimizer', type=str, default='riemannian_sgd',
                       choices=['riemannian_sgd', 'riemannian_adam'],
                       help='Type of Riemannian optimizer for embeddings (default: riemannian_sgd)')
    
    # Transformer specific parameters
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of Transformer attention heads (default: 8)')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of Transformer layers (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate (default: 0.3)')
    
    # Data path
    parser.add_argument('--data_path', type=str, 
                       default="/data/yuyu/data/MIMIC_IV/hosp",
                       help='MIMIC-IV data path')
    
    # GEM mapping parameters
    parser.add_argument('--gem_file', type=str, default=None,
                       help='Path to GEM mapping file for ICD-9-CM to ICD-10-CM conversion (optional)')
    parser.add_argument('--disable_gem_mapping', action='store_true',
                       help='Disable ICD-9-CM to ICD-10-CM mapping (default: False)')
    
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
        actual_monitor = 'accuracy' if args.task_type == 'dialysis' else args.monitor_metric
        print(f"  Patience: {args.patience}, Min delta: {args.min_delta}, Monitor: {actual_monitor}")
    
    if args.model == 'transformer':
        print(f"Transformer parameters - attention heads: {args.num_heads}, layers: {args.num_layers}")
    
    print(f"Single training run with seed: {args.seed}")
    
    print(f"\nLoading MIMIC-IV dataset for {args.task_type} prediction...")
    mimic4_base = MIMIC4Dataset(
        root=args.data_path,
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
    )

    # Choose the appropriate task function based on task_type
    if args.task_type == 'diagnosis':
        mimic4_prediction = mimic4_base.set_task(diag_prediction_mimic4_fn)
    elif args.task_type == 'dialysis':
        mimic4_prediction = mimic4_base.set_task(dialysis_prediction_mimic4_fn)
    else:
        raise ValueError(f"Unknown task_type: {args.task_type}")

    # Prepare model parameters
    model_kwargs = {
        'p': args.dropout,
    }
    
    if args.model == 'transformer':
        model_kwargs.update({
            'num_heads': args.num_heads,
            'num_layers': args.num_layers,
        })
    
    print(f"\nPreprocessing conditions codes...")
    
    # Initialize GEM mapper if not disabled
    gem_mapper = None
    if not args.disable_gem_mapping:
        gem_mapper = GEMMapper(gem_file_path=args.gem_file)
        stats = gem_mapper.get_mapping_stats()
        print(f"GEM mapping statistics:")
        print(f"  Total ICD-9-CM codes: {stats['total_icd9_codes']}")
        print(f"  Single mappings: {stats['single_mappings']}")
        print(f"  Multiple mappings: {stats['multiple_mappings']}")
        print(f"  Average ICD-10-CM codes per ICD-9-CM: {stats['avg_icd10_per_icd9']:.2f}")
    else:
        print("GEM mapping disabled.")
    
    mimic4_prediction.samples = preprocess_conditions_codes(mimic4_prediction.samples, gem_mapper=gem_mapper)
    print(f"Preprocessing completed. Processed {len(mimic4_prediction.samples)} samples.")
    
    print(f"\nStarting single training run for {args.model} model...")
    
    if args.task_type == 'diagnosis':
        model, vocabs, y_itos, test_metrics = train_model_on_samples(
            mimic4_prediction.samples,
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
            hyperbolic_embeddings_file=args.hyperbolic_embeddings_file,
            max_seq_length=args.max_seq_length,
            use_riemannian_embeddings=args.use_riemannian_embeddings,
            riemannian_optimizer=args.riemannian_optimizer,
            **model_kwargs
        )
    elif args.task_type == 'dialysis':
        model, vocabs, test_metrics = train_dialysis_model_on_samples(
            mimic4_prediction.samples,
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
            monitor_metric='accuracy',  # Use accuracy for dialysis prediction
            use_gpu=args.use_gpu,
            force_cpu=args.force_cpu,
            **model_kwargs
        )
    else:
        raise ValueError(f"Unknown task_type: {args.task_type}")
    
    print(f"\n[DONE] {args.model.upper()} model test results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")