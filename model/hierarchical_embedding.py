"""
Hierarchical embedding module for ICD-10-CM codes
Implements decomposable embeddings using ancestor paths with two modes:
- weighted_sum: Möbius addition in hyperbolic space
- concatenation: Logmap to Euclidean, concatenate, then linear projection
"""
import torch
import torch.nn as nn
import geoopt
from collections import defaultdict


def mobius_add(u, v, c=1.0):
    """
    Möbius addition in Poincaré ball (hyperbolic space).
    
    Args:
        u: First point in hyperbolic space (..., d)
        v: Second point in hyperbolic space (..., d)
        c: Curvature parameter (default: 1.0)
    
    Returns:
        Möbius sum u ⊕ v in hyperbolic space (..., d)
    """
    ball = geoopt.PoincareBall(c=c)
    # Möbius addition formula: u ⊕ v = ((1 + 2c <u,v> + c||v||^2)u + (1 + c||u||^2)v) / (1 + 2c <u,v> + c^2||u||^2||v||^2)
    uv = torch.sum(u * v, dim=-1, keepdim=True)  # <u,v>
    u_norm_sq = torch.sum(u * u, dim=-1, keepdim=True)  # ||u||^2
    v_norm_sq = torch.sum(v * v, dim=-1, keepdim=True)  # ||v||^2
    
    denominator = 1 + 2 * c * uv + c * c * u_norm_sq * v_norm_sq
    numerator = (1 + 2 * c * uv + c * v_norm_sq) * u + (1 + c * u_norm_sq) * v
    
    result = numerator / (denominator + 1e-15)  # Add small epsilon for numerical stability
    return ball.projx(result)  # Project back to ball if needed


def mobius_weighted_sum(embeddings, weights, c=1.0):
    """
    Weighted sum in hyperbolic space using Möbius addition.
    
    Args:
        embeddings: List or tensor of embeddings (L, ..., d) where L is number of levels
        weights: Weight tensor (L, ...) or (..., L) 
        c: Curvature parameter
    
    Returns:
        Weighted sum in hyperbolic space (..., d)
    """
    if isinstance(embeddings, list):
        embeddings = torch.stack(embeddings, dim=0)  # (L, ..., d)
    
    # Ensure embeddings are in (L, ..., d) format
    if embeddings.dim() == 2:
        embeddings = embeddings.unsqueeze(1)  # (L, 1, d)
    
    # Normalize weights (should sum to 1)
    if weights.dim() == 1:
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # (L, 1, 1)
    elif weights.dim() == 2:
        weights = weights.unsqueeze(-1)  # (L, ..., 1)
    
    # Start with first embedding scaled by its weight
    # In hyperbolic space, scaling is done via exponential/logarithmic map
    ball = geoopt.PoincareBall(c=c)
    
    # For weighted sum, we can use Möbius scalar-vector multiplication
    # Simplified: iterative Möbius addition of weighted embeddings
    result = embeddings[0] * weights[0]
    result = ball.projx(result)
    
    for i in range(1, embeddings.shape[0]):
        weighted_emb = embeddings[i] * weights[i]
        weighted_emb = ball.projx(weighted_emb)
        result = mobius_add(result, weighted_emb, c=c)
    
    return result


class HierarchicalHyperbolicEmbedding(nn.Module):
    """
    Hierarchical embedding for ICD-10-CM codes using ancestor paths.
    Each code's embedding is a weighted combination of its ancestor embeddings.
    
    Two modes:
    1. weighted_sum: Use Möbius addition in hyperbolic space
    2. concatenation: Logmap to Euclidean, concatenate, linear projection
    """
    
    def __init__(self, vocab_size, embedding_dim, ancestors_dict, itos, 
                 c=1.0, padding_idx=0, init_scale=1e-3,
                 mode="weighted_sum", max_depth=None):
        """
        Args:
            vocab_size: Size of vocabulary (including padding)
            embedding_dim: Embedding dimension
            ancestors_dict: Dictionary mapping code -> list of ancestor codes [root, ..., code]
            itos: Dictionary mapping index -> code string
            c: Curvature for Poincaré ball
            padding_idx: Padding index (usually 0)
            init_scale: Initialization scale for embeddings
            mode: "weighted_sum" or "concatenation"
            max_depth: Maximum depth of hierarchy (auto-computed if None)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.ancestors_dict = ancestors_dict
        self.itos = itos
        self.c = c
        self.padding_idx = padding_idx
        self.mode = mode
        self.ball = geoopt.PoincareBall(c=c)
        
        # Build level vocabularies
        level_vocabs = self._build_level_vocabs(ancestors_dict)
        self.max_depth = max_depth if max_depth is not None else max(level_vocabs.keys()) + 1
        self.level_vocabs = level_vocabs
        
        # Create per-level embedding tables
        self.level_embeddings = nn.ModuleDict()
        for level_idx in range(self.max_depth):
            num_nodes = len(level_vocabs.get(level_idx, []))
            if num_nodes > 0:
                # Add 1 for padding
                emb = HyperbolicEmbedding(
                    num_nodes + 1, embedding_dim, c=c, 
                    padding_idx=0, init_scale=init_scale
                )
                self.level_embeddings[str(level_idx)] = emb
        
        # Build code -> level code index mappings
        self.code_to_level_indices = self._build_code_mappings(level_vocabs)
        
        # For concatenation mode: linear projection after concatenation
        if mode == "concatenation":
            self.projection = nn.Linear(embedding_dim * self.max_depth, embedding_dim)
        
        # Learnable weights for combining levels (used in both modes)
        # Equal weights by default, but can be learned
        self.level_weights = nn.Parameter(torch.ones(self.max_depth) / self.max_depth)
    
    def _build_level_vocabs(self, ancestors_dict):
        """Build vocabularies for each level."""
        level_vocabs = defaultdict(set)
        
        for code, ancestors in ancestors_dict.items():
            for level_idx, ancestor_code in enumerate(ancestors):
                level_vocabs[level_idx].add(ancestor_code)
        
        # Convert to sorted lists
        level_vocabs = {level: sorted(list(codes)) for level, codes in level_vocabs.items()}
        
        # Create stoi for each level
        level_stoi = {}
        for level_idx, codes in level_vocabs.items():
            level_stoi[level_idx] = {code: idx + 1 for idx, code in enumerate(codes)}  # +1 for padding
        
        self.level_stoi = level_stoi
        return level_vocabs
    
    def _build_code_mappings(self, level_vocabs):
        """Build mapping from code index to level code indices."""
        code_to_level_indices = {}
        
        for code_idx in range(self.vocab_size):
            if code_idx == self.padding_idx:
                code_to_level_indices[code_idx] = None
                continue
            
            code_str = self.itos.get(code_idx)  # Direct lookup: 0=padding, 1+=codes
            if code_str is None or code_str == "<pad>":
                code_to_level_indices[code_idx] = None
                continue
            
            ancestors = self.ancestors_dict.get(code_str, [])
            level_indices = []
            
            for level_idx, ancestor_code in enumerate(ancestors):
                level_stoi = self.level_stoi.get(level_idx, {})
                ancestor_idx = level_stoi.get(ancestor_code, 0)  # 0 = padding if not found
                level_indices.append(ancestor_idx)
            
            code_to_level_indices[code_idx] = level_indices
        
        return code_to_level_indices
    
    def forward(self, idx):
        """
        Args:
            idx: Code indices (B, L) where 0=padding
        
        Returns:
            Embeddings in Euclidean space (B, L, embedding_dim)
            (projects from hyperbolic via logmap0)
        """
        device = idx.device
        B, L = idx.shape
        
        # Flatten for processing
        idx_flat = idx.flatten()  # (B*L,)
        
        # Get level indices for all codes
        all_level_embs = []  # Will be list of (B*L, max_depth, d)
        
        for pos in range(B * L):
            code_idx = idx_flat[pos].item()
            
            if code_idx == self.padding_idx:
                # Padding: use zero embeddings for all levels
                level_embs = torch.zeros(self.max_depth, self.embedding_dim, device=device)
                all_level_embs.append(level_embs)
                continue
            
            level_indices = self.code_to_level_indices.get(code_idx, None)
            if level_indices is None:
                level_embs = torch.zeros(self.max_depth, self.embedding_dim, device=device)
                all_level_embs.append(level_embs)
                continue
            
            # Get embeddings from each level
            level_emb_list = []
            for level_idx, level_code_idx in enumerate(level_indices):
                if str(level_idx) in self.level_embeddings and level_code_idx > 0:
                    level_emb_module = self.level_embeddings[str(level_idx)]
                    level_emb_h = level_emb_module.weight[level_code_idx]
                    level_emb_list.append(level_emb_h)
                else:
                    level_emb_h = torch.zeros(self.embedding_dim, device=device)
                    level_emb_list.append(level_emb_h)
            
            # Pad to max_depth if needed
            while len(level_emb_list) < self.max_depth:
                level_emb_list.append(torch.zeros(self.embedding_dim, device=device))
            
            level_embs = torch.stack(level_emb_list, dim=0)  # (max_depth, d)
            all_level_embs.append(level_embs)
        
        # Stack all: (B*L, max_depth, d)
        all_level_embs = torch.stack(all_level_embs, dim=0)
        
        # Process based on mode
        if self.mode == "weighted_sum":
            # Möbius weighted sum
            weights = torch.softmax(self.level_weights, dim=0)  # (max_depth,)
            weights = weights.view(1, self.max_depth, 1)  # (1, max_depth, 1)
            
            # Apply weights: simple scalar multiplication (approximation)
            weighted_embs = all_level_embs * weights  # (B*L, max_depth, d)
            weighted_embs = self.ball.projx(weighted_embs.view(-1, self.embedding_dim))
            weighted_embs = weighted_embs.view(B * L, self.max_depth, self.embedding_dim)
            
            # Möbius sum across levels (iterative)
            combined = weighted_embs[:, 0, :]  # (B*L, d)
            for i in range(1, self.max_depth):
                combined = mobius_add(combined, weighted_embs[:, i, :], c=self.c)
            
            # Project to Euclidean
            result_flat = self.ball.logmap0(combined)  # (B*L, d)
        
        elif self.mode == "concatenation":
            # Project each level to Euclidean, then concatenate
            level_embs_euclidean = self.ball.logmap0(
                all_level_embs.view(-1, self.embedding_dim)
            )  # (B*L * max_depth, d)
            level_embs_euclidean = level_embs_euclidean.view(B * L, self.max_depth * self.embedding_dim)
            
            # Linear projection
            result_flat = self.projection(level_embs_euclidean)  # (B*L, d)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Reshape back: (B, L, d)
        result = result_flat.view(B, L, self.embedding_dim)
        
        # Handle padding mask: set padding positions to zero
        padding_mask = (idx == self.padding_idx).unsqueeze(-1)  # (B, L, 1)
        result = result.masked_fill(padding_mask, 0.)
        
        return result
    
    @torch.no_grad()
    def reproject_(self):
        """Reproject all level embeddings to Poincaré ball."""
        for level_emb in self.level_embeddings.values():
            level_emb.reproject_()

    def get_hyperbolic_embeddings(self, idx):
        """
        Return combined embeddings in hyperbolic space before logmap.
        Shape matches forward: (B, L, embedding_dim) but values live on the ball.
        For concatenation mode, we still construct a single hyperbolic representative
        by computing a Möbius weighted sum across levels using level_weights.
        """
        device = idx.device
        B, L = idx.shape
        idx_flat = idx.flatten()

        # Collect per-position per-level hyperbolic embeddings
        all_level_embs = []  # (B*L, max_depth, d)
        for pos in range(B * L):
            code_idx = idx_flat[pos].item()
            if code_idx == self.padding_idx:
                all_level_embs.append(torch.zeros(self.max_depth, self.embedding_dim, device=device))
                continue
            level_indices = self.code_to_level_indices.get(code_idx, None)
            if level_indices is None:
                all_level_embs.append(torch.zeros(self.max_depth, self.embedding_dim, device=device))
                continue
            level_emb_list = []
            for level_idx, level_code_idx in enumerate(level_indices):
                if str(level_idx) in self.level_embeddings and level_code_idx > 0:
                    level_emb_module = self.level_embeddings[str(level_idx)]
                    level_emb_h = level_emb_module.weight[level_code_idx]
                    level_emb_list.append(level_emb_h)
                else:
                    level_emb_list.append(torch.zeros(self.embedding_dim, device=device))
            while len(level_emb_list) < self.max_depth:
                level_emb_list.append(torch.zeros(self.embedding_dim, device=device))
            all_level_embs.append(torch.stack(level_emb_list, dim=0))

        all_level_embs = torch.stack(all_level_embs, dim=0)  # (B*L, max_depth, d)

        # Möbius weighted sum across levels to form a single hyperbolic point per position
        weights = torch.softmax(self.level_weights, dim=0).view(1, self.max_depth, 1)
        weighted_embs = all_level_embs * weights  # (B*L, max_depth, d)
        weighted_embs = self.ball.projx(weighted_embs.view(-1, self.embedding_dim))
        weighted_embs = weighted_embs.view(B * L, self.max_depth, self.embedding_dim)

        combined = weighted_embs[:, 0, :]
        for i in range(1, self.max_depth):
            combined = mobius_add(combined, weighted_embs[:, i, :], c=self.c)

        # Reshape and zero-out padding positions
        combined = combined.view(B, L, self.embedding_dim)
        padding_mask = (idx == self.padding_idx).unsqueeze(-1)
        combined = combined.masked_fill(padding_mask, 0.)
        return self.ball.projx(combined)


class HyperbolicEmbedding(nn.Module):
    """Helper class: basic hyperbolic embedding (same as in models.py)"""
    def __init__(self, num_embeddings, embedding_dim, c=1.0, padding_idx=0, init_scale=1e-3):
        super().__init__()
        self.ball = geoopt.PoincareBall(c=c)
        self.padding_idx = padding_idx
        w = torch.randn(num_embeddings, embedding_dim) * init_scale
        w = self.ball.projx(w)
        self.weight = geoopt.ManifoldParameter(w, manifold=self.ball)
        if padding_idx is not None:
            with torch.no_grad():
                self.weight.data[padding_idx].fill_(0.)
                self.weight.data = self.ball.projx(self.weight.data)
    
    @torch.no_grad()
    def reproject_(self):
        self.weight.data = self.ball.projx(self.weight.data)

