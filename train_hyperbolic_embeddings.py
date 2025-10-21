"""
Hyperbolic embedding training script for conditions codes
Trains hyperbolic embeddings and saves them to a file for later use
"""
import argparse
import torch
import pickle
import json
from typing import List, Dict
from util.data_processing_trinetx import load_trinetx_data, process_trinetx_dialysis_patients
from util.hyperbolic_conditions import ConditionsHyperbolicEmbedder


def initialize_and_save_embeddings(samples: List[Dict], embedding_dim: int = 20, 
                                 output_file: str = "hyperbolic_embeddings.pkl",
                                 use_hierarchical_init: bool = True):
    """
    Initialize hyperbolic embeddings for conditions codes using hierarchical rules and save to file
    
    Args:
        samples: List of sample dictionaries containing conditions
        embedding_dim: Dimension of hyperbolic embeddings
        output_file: Path to save the initialized embeddings
        use_hierarchical_init: Whether to use hierarchical initialization (True) or random initialization (False)
    """
    print(f"Initializing hyperbolic embeddings for conditions codes...")
    
    # Collect all unique condition codes from samples
    all_conditions = set()
    for sample in samples:
        all_conditions.update(sample["conditions"])
    all_conditions = list(all_conditions)
    
    print(f"Found {len(all_conditions)} unique condition codes")
    
    # Create hyperbolic embedder
    conditions_embedder = ConditionsHyperbolicEmbedder(
        all_conditions, 
        embedding_dim=embedding_dim
    )
    
    if use_hierarchical_init:
        print(f"Using hierarchical initialization with dim={embedding_dim}")
        # Use hierarchical initialization
        embeddings = conditions_embedder.initialize_hierarchical_embeddings(output_file)
        conditions_embedder.code2embedding = embeddings
        conditions_embedder.trained = True
    else:
        print(f"Using random initialization with dim={embedding_dim}")
        # Use random initialization (fallback)
        conditions_embedder.train_embeddings(steps=0)  # No training steps
    
    print(f"Initialized hyperbolic embeddings for {len(all_conditions)} conditions codes")
    
    # Save the embedder to file (if not already saved by hierarchical init)
    if not use_hierarchical_init:
        with open(output_file, 'wb') as f:
            pickle.dump(conditions_embedder, f)
    
    # Save metadata as JSON for easy inspection
    metadata = {
        'embedding_dim': embedding_dim,
        'num_conditions': len(all_conditions),
        'initialization_method': 'hierarchical' if use_hierarchical_init else 'random',
        'conditions_codes': sorted(all_conditions)
    }
    
    metadata_file = output_file.replace('.pkl', '_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved embeddings to: {output_file}")
    print(f"Saved metadata to: {metadata_file}")
    
    return conditions_embedder


def train_and_save_embeddings(samples: List[Dict], embedding_dim: int = 20, 
                             output_file: str = "hyperbolic_embeddings.pkl",
                             steps: int = 500, batch_size: int = 256, lr: float = 1e-4,
                             lambda_hierarchy: float = 1.0, use_hierarchical_init: bool = True):
    """
    Train hyperbolic embeddings for conditions codes and save to file
    
    Args:
        samples: List of sample dictionaries containing conditions
        embedding_dim: Dimension of hyperbolic embeddings
        output_file: Path to save the trained embeddings
        steps: Number of training steps
        batch_size: Batch size for training
        lr: Learning rate for training
        lambda_hierarchy: Weight for hierarchy constraint loss
        use_hierarchical_init: Whether to use hierarchical initialization before training
    """
    print(f"Training hyperbolic embeddings for conditions codes...")
    
    # Collect all unique condition codes from samples
    all_conditions = set()
    for sample in samples:
        all_conditions.update(sample["conditions"])
    all_conditions = list(all_conditions)
    
    print(f"Found {len(all_conditions)} unique condition codes")
    
    # Create hyperbolic embedder
    conditions_embedder = ConditionsHyperbolicEmbedder(
        all_conditions, 
        embedding_dim=embedding_dim
    )
    
    if use_hierarchical_init and steps > 0:
        print(f"Using hierarchical initialization before training with dim={embedding_dim}")
        # Initialize with hierarchical rules first
        embeddings = conditions_embedder.initialize_hierarchical_embeddings(output_file.replace('.pkl', '_init.pkl'))
        conditions_embedder.code2embedding = embeddings
        conditions_embedder.trained = True
        
        print(f"Starting training from hierarchical initialization with steps={steps}, batch_size={batch_size}, lr={lr}, lambda_hierarchy={lambda_hierarchy}")
        # Continue training from hierarchical initialization
        conditions_embedder.train_embeddings(
            steps=steps,
            batch_size=batch_size,
            lr=lr,
            lambda_hierarchy=lambda_hierarchy
        )
    elif use_hierarchical_init and steps == 0:
        print(f"Using hierarchical initialization only (no training) with dim={embedding_dim}")
        # Use hierarchical initialization only
        embeddings = conditions_embedder.initialize_hierarchical_embeddings(output_file)
        conditions_embedder.code2embedding = embeddings
        conditions_embedder.trained = True
    else:
        print(f"Training from random initialization with dim={embedding_dim}, steps={steps}, batch_size={batch_size}, lr={lr}, lambda_hierarchy={lambda_hierarchy}")
        # Train from random initialization
        conditions_embedder.train_embeddings(
            steps=steps,
            batch_size=batch_size,
            lr=lr,
            lambda_hierarchy=lambda_hierarchy
        )
        
        # Save the trained embedder to file
        with open(output_file, 'wb') as f:
            pickle.dump(conditions_embedder, f)
    
    print(f"Trained hyperbolic embeddings for {len(all_conditions)} conditions codes")
    
    # Save metadata as JSON for easy inspection
    metadata = {
        'embedding_dim': embedding_dim,
        'num_conditions': len(all_conditions),
        'training_steps': steps,
        'batch_size': batch_size,
        'learning_rate': lr,
        'lambda_hierarchy': lambda_hierarchy,
        'use_hierarchical_init': use_hierarchical_init,
        'conditions_codes': sorted(all_conditions)
    }
    
    metadata_file = output_file.replace('.pkl', '_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved embeddings to: {output_file}")
    print(f"Saved metadata to: {metadata_file}")
    
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
    parser = argparse.ArgumentParser(description='Train or initialize hyperbolic embeddings for conditions codes')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, 
                       default="/data/yuyu/data/trinetx_data",
                       help='TriNetX data path')
    
    # Embedding parameters
    parser.add_argument('--embedding_dim', type=int, default=20,
                       help='Dimension of hyperbolic embeddings (default: 20)')
    parser.add_argument('--output_file', type=str, default='hyperbolic_embeddings.pkl',
                       help='Output file to save embeddings (default: hyperbolic_embeddings.pkl)')
    
    # Initialization parameters
    parser.add_argument('--use_hierarchical_init', action='store_true', default=True,
                       help='Use hierarchical initialization (default: True)')
    parser.add_argument('--no_hierarchical_init', action='store_false', dest='use_hierarchical_init',
                       help='Disable hierarchical initialization and use random initialization')
    parser.add_argument('--init_only', action='store_true', default=False,
                       help='Only initialize embeddings without training (steps=0)')
    
    # Training parameters
    parser.add_argument('--steps', type=int, default=0,
                       help='Number of training steps (default: 0). Set to 0 for initialization only.')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for training (default: 256)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate for training (default: 1e-3)')
    parser.add_argument('--lambda_hierarchy', type=float, default=100,
                       help='Weight for hierarchy constraint loss (default: 100)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Handle init_only flag
    if args.init_only:
        args.steps = 0
    
    print(f"Loading TriNetX dataset for dialysis prediction...")
    
    # Load TriNetX data
    trinetx_data = load_trinetx_data(args.data_path)
    
    # Process dialysis patients to get samples
    samples = process_trinetx_dialysis_patients(trinetx_data)
    
    print(f"Loaded {len(samples)} dialysis samples")
    
    # Choose between initialization only or training
    if args.steps == 0:
        print("Running initialization only (no training)...")
        initialize_and_save_embeddings(
            samples=samples,
            embedding_dim=args.embedding_dim,
            output_file=args.output_file,
            use_hierarchical_init=args.use_hierarchical_init
        )
        print("Hyperbolic embedding initialization completed!")
    else:
        print("Running training with optional hierarchical initialization...")
        train_and_save_embeddings(
            samples=samples,
            embedding_dim=args.embedding_dim,
            output_file=args.output_file,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            lambda_hierarchy=args.lambda_hierarchy,
            use_hierarchical_init=args.use_hierarchical_init
        )
        print("Hyperbolic embedding training completed!")
