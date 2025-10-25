"""
Hyperbolic embedding training script for ICD-10-CM codes
Trains hyperbolic embeddings using the complete ICD-10-CM code set
Supports both standard and Riemannian optimization
"""
import argparse
import torch
import pickle
import json
from typing import List, Dict, Optional
from util.hyperbolic_conditions import ConditionsHyperbolicEmbedder
from util.hyperbolic_riemannian import RiemannianConditionsHyperbolicEmbedder
from util.icd10_manager import get_cached_icd10_codes


def train_and_save_embeddings(embedding_dim: int = 20, 
                             output_file: str = "hyperbolic_embeddings.pkl",
                             steps: int = 500, batch_size: int = 256, lr: float = 1e-4,
                             lambda_hierarchy: float = 1.0, 
                             icd10_year: str = "2024",
                             use_riemannian: bool = False,
                             optimizer_type: str = "riemannian_sgd"):
    """
    Train hyperbolic embeddings for ICD-10-CM codes and save to file
    
    Args:
        embedding_dim: Dimension of hyperbolic embeddings
        output_file: Path to save the trained embeddings
        steps: Number of training steps
        batch_size: Batch size for training
        lr: Learning rate for training
        lambda_hierarchy: Weight for hierarchy constraint loss
        icd10_year: Year of ICD-10-CM codes to use
        use_riemannian: Whether to use Riemannian optimization
        optimizer_type: Type of Riemannian optimizer ("riemannian_sgd" or "riemannian_adam")
    """
    print(f"Training hyperbolic embeddings for ICD-10-CM codes...")
    if use_riemannian:
        print(f"Using Riemannian optimization with {optimizer_type}")
    else:
        print("Using standard Adam optimization")
    print("Using origin initialization (near origin)")
    
    # Get comprehensive ICD-10-CM code set
    print(f"Loading complete ICD-10-CM code set for year {icd10_year}...")
    all_conditions = get_cached_icd10_codes(icd10_year)
    print(f"Loaded {len(all_conditions)} ICD-10-CM codes")
    print(f"Sample codes: {sorted(all_conditions)[:10]}...")
    
    # Create appropriate embedder based on optimization type
    if use_riemannian:
        conditions_embedder = RiemannianConditionsHyperbolicEmbedder(
            all_conditions, 
            embedding_dim=embedding_dim
        )
    else:
        conditions_embedder = ConditionsHyperbolicEmbedder(
            all_conditions, 
            embedding_dim=embedding_dim
        )
    
    if steps > 0:
        print(f"Using origin initialization before training with dim={embedding_dim}")
        # Initialize with origin method first
        init_file = output_file.replace('.pkl', '_init.pkl')
        conditions_embedder.initialize_origin_embeddings(init_file)
        
        print(f"Starting training from origin initialization with steps={steps}, batch_size={batch_size}, lr={lr}, lambda_hierarchy={lambda_hierarchy}")
        # Continue training from initialization
        if use_riemannian:
            conditions_embedder.train_embeddings(
                steps=steps,
                batch_size=batch_size,
                lr=lr,
                lambda_hierarchy=lambda_hierarchy,
                optimizer_type=optimizer_type
            )
        else:
            conditions_embedder.train_embeddings(
                steps=steps,
                batch_size=batch_size,
                lr=lr,
                lambda_hierarchy=lambda_hierarchy
            )
        conditions_embedder.trained = True
    elif steps == 0:
        print(f"Using origin initialization only (no training) with dim={embedding_dim}")
        # Use origin initialization only
        conditions_embedder.initialize_origin_embeddings(output_file)
        conditions_embedder.trained = True
    
    print(f"Trained hyperbolic embeddings for {len(all_conditions)} conditions codes")
    with open(output_file, 'wb') as f:
        pickle.dump(conditions_embedder, f)
    
    print(f"Saved embeddings to: {output_file}")
    
    return conditions_embedder


def load_embeddings(embedding_file: str) -> ConditionsHyperbolicEmbedder:
    """
    Load pre-trained hyperbolic embeddings from file
    
    Args:
        embedding_file: Path to the saved embeddings file
        
    Returns:
        ConditionsHyperbolicEmbedder instance with loaded embeddings
    """
    with open(embedding_file, 'rb') as f:
        conditions_embedder = pickle.load(f)
    
    print(f"Loaded hyperbolic embeddings from: {embedding_file}")
    print(f"Embedding dimension: {conditions_embedder.get_embedding_dim()}")
    print(f"Number of conditions: {len(conditions_embedder.conditions_codes)}")
    
    return conditions_embedder


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train hyperbolic embeddings for ICD-10-CM codes')
    
    # Embedding parameters
    parser.add_argument('--embedding_dim', type=int, default=32,
                       help='Dimension of hyperbolic embeddings (default: 32)')
    parser.add_argument('--output_file', type=str, default='hyperbolic_embeddings.pkl',
                       help='Output file to save embeddings (default: hyperbolic_embeddings.pkl)')
    
    # ICD-10 parameters
    parser.add_argument('--icd10_year', type=str, default='2024',
                       choices=['2022', '2023', '2024'],
                       help='Year of ICD-10-CM codes to use (default: 2024)')
    
    # Training parameters
    parser.add_argument('--steps', type=int, default=5000,
                       help='Number of training steps (default: 5000). Set to 0 for initialization only.')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for training (default: 256)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate for training (default: 1e-3)')
    parser.add_argument('--lambda_hierarchy', type=float, default=100,
                       help='Weight for hierarchy constraint loss (default: 100)')
    
    # Optimization parameters
    parser.add_argument('--use_riemannian', action='store_true',
                       help='Use Riemannian optimization instead of standard Adam (default: False)')
    parser.add_argument('--riemannian_optimizer', type=str, default='riemannian_sgd',
                       choices=['riemannian_sgd', 'riemannian_adam'],
                       help='Type of Riemannian optimizer (default: riemannian_sgd)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print(f"Output file: {args.output_file}")
    print(f"Embedding dimension: {args.embedding_dim}")
    print(f"Training steps: {args.steps}")
    print(f"ICD-10 year: {args.icd10_year}")
    print(f"Use Riemannian optimization: {args.use_riemannian}")
    if args.use_riemannian:
        print(f"Riemannian optimizer: {args.riemannian_optimizer}")
    
    # Train and save embeddings
    train_and_save_embeddings(
        embedding_dim=args.embedding_dim,
        output_file=args.output_file,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_hierarchy=args.lambda_hierarchy,
        icd10_year=args.icd10_year,
        use_riemannian=args.use_riemannian,
        optimizer_type=args.riemannian_optimizer
    )
    print("Hyperbolic embedding initialization(training) completed!")
