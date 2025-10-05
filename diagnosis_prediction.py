"""
Diagnosis prediction main program
Using MIMIC-IV dataset for diagnosis prediction task
"""
import argparse
import torch
import numpy as np
import warnings
from pyhealth.datasets import MIMIC4Dataset

from util.data_processing import diag_prediction_mimic4_fn
from training.training import train_model_on_samples

warnings.filterwarnings('ignore')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Diagnosis prediction model training')
    
    # Model selection
    parser.add_argument('--model', type=str, default='transformer', 
                       choices=['mlp', 'transformer'],
                       help='Model type: mlp or transformer (default: mlp)')
    
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
    
    # Early stopping parameters
    parser.add_argument('--early_stopping', action='store_true', default=True,
                       help='Enable early stopping (default: True)')
    parser.add_argument('--patience', type=int, default=50,
                       help='Number of epochs to wait before stopping (default: 10)')
    parser.add_argument('--min_delta', type=float, default=0.001,
                       help='Minimum change to qualify as improvement (default: 0.001)')
    parser.add_argument('--monitor_metric', type=str, default='Acc@10',
                       choices=['P@10', 'Acc@10', 'P@20', 'Acc@20'],
                       help='Metric to monitor for early stopping (default: Acc@10)')
    
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
    print(f"Prediction task: {args.task}")
    print(f"Hidden layer dimension: {args.hidden}")
    print(f"Learning rate: {args.lr}")
    print(f"Training epochs: {args.epochs}")
    print(f"Training data percentage: {args.train_percentage:.1%}")
    print(f"Batch size: {args.batch_size}")
    print(f"Early stopping: {args.early_stopping}")
    if args.early_stopping:
        print(f"  Patience: {args.patience}, Min delta: {args.min_delta}, Monitor: {args.monitor_metric}")
    
    if args.model == 'transformer':
        print(f"Transformer parameters - attention heads: {args.num_heads}, layers: {args.num_layers}")
    
    print("\nLoading MIMIC-IV dataset...")
    mimic4_base = MIMIC4Dataset(
        root=args.data_path,
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
    )

    mimic4_prediction = mimic4_base.set_task(diag_prediction_mimic4_fn)
    print(f"Total samples: {len(mimic4_prediction.samples)}")

    # Prepare model parameters
    model_kwargs = {
        'p': args.dropout,
    }
    
    if args.model == 'transformer':
        model_kwargs.update({
            'num_heads': args.num_heads,
            'num_layers': args.num_layers,
        })
    
    print(f"\nStarting training {args.model} model...")
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
        **model_kwargs
    )
    
    print(f"\n[DONE] {args.model.upper()} model test results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")