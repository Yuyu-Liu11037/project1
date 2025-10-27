"""
Hyperbolic embedding training script for conditions codes
Trains hyperbolic embeddings and saves them to a file for later use
"""
import argparse
import torch
import pickle
import json
from typing import List, Dict
from util.hyperbolic_conditions import ConditionsHyperbolicEmbedder, get_icd_to_ccs_mapping


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
                             lambda_hierarchy: float = 1.0, lambda_cone: float = 1.0, origin_init: bool = False):
    print(f"Training hyperbolic embeddings for conditions codes...")
    
    # Load all ICD-10 condition codes from file
    all_conditions = load_icd10_codes(icd10_file_path)
    
    print(f"Loaded {len(all_conditions)} ICD-10 condition codes from file")
    
    # Get CCS code mappings for all ICD codes
    print("Mapping ICD codes to CCS codes...")
    icd_to_ccs = get_icd_to_ccs_mapping(all_conditions)
    
    # Assign UNKNOWN_CCS to codes without mapping
    mapped_count = len(icd_to_ccs)
    unmapped_count = len(all_conditions) - mapped_count
    for code in all_conditions:
        if code not in icd_to_ccs:
            icd_to_ccs[code] = "UNKNOWN_CCS"
    
    unique_ccs_codes = set(icd_to_ccs.values())
    print(f"CCS mapping statistics:")
    print(f"  - ICD codes with CCS mapping: {mapped_count}")
    print(f"  - ICD codes without CCS mapping: {unmapped_count}")
    print(f"  - Unique CCS codes: {len(unique_ccs_codes)}")
    
    # Create and train hyperbolic embedder
    conditions_embedder = ConditionsHyperbolicEmbedder(
        all_conditions, 
        embedding_dim=embedding_dim,
        icd_to_ccs=icd_to_ccs
    )
    
    init_type = "origin" if origin_init else "hierarchy-aware"
    print(f"Training embeddings with dim={embedding_dim}, steps={steps}, batch_size={batch_size}, lr={lr}, lambda_hierarchy={lambda_hierarchy}, lambda_cone={lambda_cone}, init={init_type}")
    conditions_embedder.train_embeddings(
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        lambda_hierarchy=lambda_hierarchy,
        lambda_cone=lambda_cone,
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
    
    # Display CCS code statistics if available
    if conditions_embedder.icd_to_ccs is not None:
        unique_ccs_codes = set(conditions_embedder.icd_to_ccs.values())
        # Count codes that are actually CCS codes (not ICD codes in the embedding dict)
        if conditions_embedder.code2embedding is not None:
            ccs_count = sum(1 for code in conditions_embedder.code2embedding.keys() 
                           if code not in conditions_embedder.conditions_codes)
            print(f"Number of unique CCS codes in embeddings: {ccs_count}")
    
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
    parser.add_argument('--lambda_cone', type=float, default=1.0,
                       help='Weight for cone cohesion loss (default: 1.0)')
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
        lambda_cone=args.lambda_cone,
        origin_init=args.origin_init
    )
    
    print("Hyperbolic embedding training completed!")
