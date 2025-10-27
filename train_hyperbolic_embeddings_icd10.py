"""
Hyperbolic embedding training script for conditions codes
Trains hyperbolic embeddings and saves them to a file for later use
"""
import argparse
import torch
import pickle
import json
from typing import List
from util.hyperbolic_conditions import ConditionsHyperbolicEmbedder


def load_icd10_codes(icd10_file_path: str) -> List[str]:
    """
    Load all ICD-10 condition codes from the MIMIC-IV file
    
    Args:
        icd10_file_path: Path to the ICD-10 codes file
        
    Returns:
        List of ICD-10 condition codes
    """
    all_conditions = []
    with open(icd10_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Extract the code (first part before tab/space)
                code = line.split()[0]
                all_conditions.append(code)
    
    return all_conditions


def train_and_save_embeddings(icd10_file_path: str, embedding_dim: int = 20, 
                             output_file: str = "hyperbolic_embeddings.pkl",
                             steps: int = 500, batch_size: int = 256, lr: float = 1e-4,
                             lambda_hierarchy: float = 1.0, origin_init: bool = False):
    print(f"Training hyperbolic embeddings for conditions codes...")
    
    # Load all ICD-10 condition codes from file
    all_conditions = load_icd10_codes(icd10_file_path)
    
    print(f"Loaded {len(all_conditions)} ICD-10 condition codes from file")
    
    # Create and train hyperbolic embedder
    conditions_embedder = ConditionsHyperbolicEmbedder(
        all_conditions, 
        embedding_dim=embedding_dim
    )
    
    init_type = "origin" if origin_init else "hierarchy-aware"
    print(f"Training embeddings with dim={embedding_dim}, steps={steps}, batch_size={batch_size}, lr={lr}, lambda_hierarchy={lambda_hierarchy}, init={init_type}")
    conditions_embedder.train_embeddings(
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        lambda_hierarchy=lambda_hierarchy,
        origin_init=origin_init
    )
    
    print(f"Trained hyperbolic embeddings for {len(all_conditions)} conditions codes")
    
    # Save the trained embedder to file
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
    parser = argparse.ArgumentParser(description='Train hyperbolic embeddings for conditions codes')
    
    # Data parameters
    parser.add_argument('--icd10_file', type=str, 
                       default="/data/yuyu/data/MIMIC_IV/icd10cm-codes-April-2024.txt",
                       help='Path to ICD-10 codes file')
    
    # Embedding parameters
    parser.add_argument('--embedding_dim', type=int, default=20,
                       help='Dimension of hyperbolic embeddings (default: 20)')
    parser.add_argument('--output_file', type=str, default='hyperbolic_embeddings.pkl',
                       help='Output file to save embeddings (default: hyperbolic_embeddings.pkl)')
    
    # Training parameters
    parser.add_argument('--steps', type=int, default=15000,
                       help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for training (default: 256)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate for training (default: 1e-3)')
    parser.add_argument('--lambda_hierarchy', type=float, default=100,
                       help='Weight for hierarchy constraint loss')
    parser.add_argument('--origin_init', action='store_true', default=True,
                       help='Initialize all codes near origin instead of hierarchy-aware init')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print(f"Loading ICD-10 condition codes from: {args.icd10_file}")
    
    # Train and save embeddings using ICD-10 codes
    train_and_save_embeddings(
        icd10_file_path=args.icd10_file,
        embedding_dim=args.embedding_dim,
        output_file=args.output_file,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_hierarchy=args.lambda_hierarchy,
        origin_init=args.origin_init
    )
    
    print("Hyperbolic embedding training completed!")
