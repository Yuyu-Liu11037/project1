"""
Riemannian SGD implementation for hyperbolic embeddings
Uses geoopt library for proper Riemannian optimization on Poincaré ball
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
from geoopt.manifolds import PoincareBall
from geoopt.optim import RiemannianSGD, RiemannianAdam
from typing import Dict, List, Tuple
import random
import numpy as np
from tqdm import tqdm

from .hyperbolic_conditions import (
    build_parent_child_pairs_from_codes,
    ParentChildDataset,
    reconstruction_loss,
    hierarchy_constraint_loss,
    EPS,
    MAX_NORM
)


class RiemannianPoincareEmbedding(nn.Module):
    """
    Hyperbolic embedding model using geoopt's Poincaré ball manifold
    """
    def __init__(self, num_nodes: int, dim: int = 16, init_scale=1e-4):
        super().__init__()
        
        # Create Poincaré ball manifold
        self.manifold = PoincareBall(c=1.0)  # c=1 corresponds to unit ball
        
        # Create embedding parameter on the manifold
        self.emb = geoopt.ManifoldParameter(
            torch.randn(num_nodes, dim) * init_scale,
            manifold=self.manifold
        )
        
        # Initialize all embeddings randomly with small scale (near origin)
        with torch.no_grad():
            self.emb.data = torch.randn(num_nodes, dim) * init_scale
            # Project to manifold
            self.emb.data = self.manifold.projx(self.emb.data)
    
    def forward(self, idx):
        """Forward pass - embeddings are already on the manifold"""
        return self.emb[idx]


def train_conditions_hyperbolic_embedding_riemannian(
    conditions_codes: List[str],
    dim: int = 20,
    neg_k: int = 10,
    steps: int = 100,
    batch_size: int = 256,
    lr: float = 1e-4,
    lambda_hierarchy: float = 0.5,
    optimizer_type: str = "riemannian_sgd",  # "riemannian_sgd" or "riemannian_adam"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Train hyperbolic embeddings using Riemannian optimization
    
    Args:
        conditions_codes: List of condition codes without decimal points
        dim: Embedding dimension
        neg_k: Number of negative samples per positive pair
        steps: Number of training steps
        batch_size: Batch size for training
        lr: Learning rate
        lambda_hierarchy: Weight for hierarchy constraint loss
        optimizer_type: Type of Riemannian optimizer ("riemannian_sgd" or "riemannian_adam")
        device: Device to use for training
    
    Returns:
        Dictionary mapping condition codes to their hyperbolic embeddings
    """
    print(f"Training hyperbolic embeddings with Riemannian {optimizer_type.upper()}...")
    
    # Build vocabulary - include original codes and all their parents/children
    original_codes = sorted(list(set(conditions_codes)))
    
    # Get parent-child pairs to extract all related codes
    parent_child_pairs = build_parent_child_pairs_from_codes(original_codes)
    
    # Collect all unique codes including parents and children
    all_codes = set(original_codes)  # Start with original codes
    for parent, child in parent_child_pairs:
        all_codes.add(parent)
        all_codes.add(child)
    
    codes = sorted(list(all_codes))
    code2id = {c: i for i, c in enumerate(codes)}
    num_nodes = len(code2id)
    
    print(f"Original codes: {len(original_codes)}")
    print(f"Total codes (including parents/children): {len(codes)}")
    print("???")
    print(f"Parent-child pairs: {len(parent_child_pairs)}")
    
    if len(parent_child_pairs) == 0:
        print("Warning: No parent-child pairs found. Returning zero embeddings.")
        return {code: torch.zeros(dim) for code in original_codes}
    
    # Create dataset and loader using parent-child pairs
    ds = ParentChildDataset(parent_child_pairs, code2id, num_nodes, neg_k=neg_k)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)
    
    # Create Riemannian model
    print(f"Creating Riemannian model with num_nodes={num_nodes} and dim={dim}")
    model = RiemannianPoincareEmbedding(num_nodes, dim=dim).to(device)
    
    # Create Riemannian optimizer
    if optimizer_type == "riemannian_sgd":
        optimizer = RiemannianSGD(model.parameters(), lr=lr)
    elif optimizer_type == "riemannian_adam":
        optimizer = RiemannianAdam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    print(f"Using {optimizer_type} optimizer with learning rate {lr}")
    
    # Training loop
    print("Starting training loop")
    model.train()
    it = iter(loader)
    
    # Create progress bar
    pbar = tqdm(range(1, steps + 1), desc="Riemannian Training", unit="step")
    
    for step in pbar:
        try:
            # Parent-child dataset returns (parent, child, negs, weight)
            p_idx, c_idx, neg_idx, weights = next(it)
            p_idx = p_idx.to(device)
            c_idx = c_idx.to(device)
            neg_idx = neg_idx.to(device)
            weights = weights.to(device)
            
            # Compute reconstruction loss (parent-child similarity)
            recon_loss = reconstruction_loss(model, p_idx, c_idx, neg_idx, weights)
            
            # Compute hierarchy constraint loss (parent closer to origin than child)
            hierarchy_loss = hierarchy_constraint_loss(model, p_idx, c_idx, lambda_hierarchy)
            
            # Check for NaN values
            if torch.isnan(recon_loss) or torch.isnan(hierarchy_loss):
                print(f"Warning: NaN detected at step {step}. Stopping training.")
                break
            
            # Total loss
            total_loss = recon_loss + hierarchy_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            
            # Riemannian optimizer handles the manifold constraints automatically
            optimizer.step()
            
            # Update progress bar with loss information
            pbar.set_postfix({
                'recon_loss': f'{recon_loss.item():.4f}',
                'hierarchy_loss': f'{hierarchy_loss.item():.4f}',
                'total_loss': f'{total_loss.item():.4f}'
            })
                
        except StopIteration:
            # Restart iterator if it runs out
            it = iter(loader)
            continue
    
    pbar.close()
    
    # Return code → embedding dict (already on manifold)
    with torch.no_grad():
        emb = model.emb.data.cpu()
    
    id2code = {i: c for c, i in code2id.items()}
    return {id2code[i]: emb[i] for i in range(num_nodes)}


class RiemannianConditionsHyperbolicEmbedder:
    """
    Wrapper class for using Riemannian hyperbolic embeddings
    """
    
    def __init__(self, conditions_codes: List[str], embedding_dim: int = 16):
        self.conditions_codes = conditions_codes
        self.embedding_dim = embedding_dim
        self.code2embedding = None
        self.trained = False
        self.manifold = PoincareBall(c=1.0)
    
    def train_embeddings(self, optimizer_type: str = "riemannian_sgd", **kwargs):
        """Train hyperbolic embeddings using Riemannian optimization"""
        self.code2embedding = train_conditions_hyperbolic_embedding_riemannian(
            self.conditions_codes,
            dim=self.embedding_dim,
            optimizer_type=optimizer_type,
            **kwargs
        )
        self.trained = True
    
    def get_embedding_vector(self, conditions_list: List[str]) -> torch.Tensor:
        """
        Get hyperbolic embedding vector for a list of conditions
        
        Args:
            conditions_list: List of condition codes
            
        Returns:
            Fixed-size embedding vector by averaging all condition embeddings
        """
        if not self.trained:
            raise ValueError("Embeddings not trained yet. Call train_embeddings() first.")
        
        embeddings = []
        for cond in conditions_list:
            if cond in self.code2embedding:
                embeddings.append(self.code2embedding[cond])
            else:
                # Use zero embedding for unknown codes
                embeddings.append(torch.zeros(self.embedding_dim))
        
        if len(embeddings) == 0:
            # Return zero vector if no conditions
            return torch.zeros(self.embedding_dim)
        
        # Average all embeddings to get a fixed-size representation
        return torch.stack(embeddings).mean(dim=0)
    
    def get_embedding_sequences(self, conditions_list: List[str]) -> torch.Tensor:
        """
        Get sequence of hyperbolic embeddings for a list of conditions
        
        Args:
            conditions_list: List of condition codes
            
        Returns:
            Tensor of shape (len(conditions_list), embedding_dim)
        """
        if not self.trained:
            raise ValueError("Embeddings not trained yet. Call train_embeddings() first.")
        
        embeddings = []
        for cond in conditions_list:
            if cond in self.code2embedding:
                embeddings.append(self.code2embedding[cond])
            else:
                # Use zero embedding for unknown codes
                embeddings.append(torch.zeros(self.embedding_dim))
        
        if len(embeddings) == 0:
            # Return empty tensor if no conditions
            return torch.zeros(0, self.embedding_dim)
        
        return torch.stack(embeddings)
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension"""
        return self.embedding_dim
    
    def initialize_origin_embeddings(self, output_file: str = "riemannian_hyperbolic_embeddings.pkl"):
        """
        Initialize Riemannian hyperbolic embeddings near the origin (uniform small scale)
        """
        print(f"Initializing Riemannian hyperbolic embeddings near origin...")
        
        # Build vocabulary
        original_codes = sorted(list(set(self.conditions_codes)))
        parent_child_pairs = build_parent_child_pairs_from_codes(original_codes)
        
        # Collect all unique codes
        all_codes = set(original_codes)
        for parent, child in parent_child_pairs:
            all_codes.add(parent)
            all_codes.add(child)
        
        codes = sorted(list(all_codes))
        code2id = {c: i for i, c in enumerate(codes)}
        
        print(f"Original codes: {len(original_codes)}")
        print(f"Total codes (including parents/children): {len(codes)}")
        
        # Create model with origin initialization (small scale near origin)
        model = RiemannianPoincareEmbedding(
            len(codes), 
            dim=self.embedding_dim, 
            init_scale=1e-4  # Very small scale to keep near origin
        )
        
        # Extract embeddings
        with torch.no_grad():
            emb = model.emb.data.cpu()
        
        id2code = {i: c for c, i in code2id.items()}
        self.code2embedding = {id2code[i]: emb[i] for i in range(len(codes))}
        self.trained = True
        
        # Save to file
        import pickle
        with open(output_file, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"Saved origin-initialized Riemannian embeddings to: {output_file}")
        return self.code2embedding
