"""
Hyperbolic embedding module for conditions codes
Integrates with the existing dialysis prediction pipeline
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm

# ---------------------------
# Poincaré ball utilities
# ---------------------------
EPS = 1e-7  # Smaller epsilon for better numerical stability
MAX_NORM = 1 - 1e-4  # Keep further from boundary for stability

def mobius_proj(x, max_norm=MAX_NORM):
    """Project to inside the unit ball"""
    norm = x.norm(dim=-1, keepdim=True).clamp_min(EPS)
    factor = torch.where(norm >= max_norm, max_norm / norm, torch.ones_like(norm))
    return x * factor

def poincare_dist(u, v, eps=EPS):
    """Poincaré distance between two points with improved numerical stability"""
    uu = torch.clamp(1 - (u*u).sum(dim=-1), min=eps)
    vv = torch.clamp(1 - (v*v).sum(dim=-1), min=eps)
    uv = (u - v).pow(2).sum(dim=-1)
    x = 1 + 2 * uv / (uu * vv)
    # Clamp x to prevent acosh from producing NaN
    x = torch.clamp(x, min=1+eps, max=1e6)  # Upper bound to prevent overflow
    return torch.acosh(x)

def extract_icd_parent_child_pair(code: str) -> List[Tuple[str, str]]:
    """
    Extract parent-child pairs for an ICD-10-CM code.
    
    For codes with length > 3, the first three characters are considered the parent.
    For codes with length >= 6, additional parent-child pairs are extracted:
    (code[:n-1], code[:n]) for n >= 6 and n <= len(code)
    
    Examples:
    - 'E11.65' -> [('E11', 'E11.6'), ('E11.6', 'E11.65')]
    - 'N17.9' -> [('N17', 'N17.9')] 
    - 'E11' -> [] (no parent)
    - 'E11.654' -> [('E11', 'E11.6'), ('E11.6', 'E11.65'), ('E11.65', 'E11.654')]
    
    Args:
        code: ICD code string (e.g., 'E11.65', 'N17.9')
        
    Returns:
        List of (parent, child) tuples
    """
    if not code or len(code) <= 3:
        return []
    
    pairs = []
    
    # For codes with length >= 6, extract step-by-step parent-child pairs starting from n=4
    if len(code) >= 6:
        for n in range(4, len(code) + 1):
            parent = code[:n-1]
            child = code[:n]
            pairs.append((parent, child))
    elif len(code) > 3:
        # For codes with length 4-5, first 3 characters are the parent
        parent = code[:3]
        child = code
        pairs.append((parent, child))
    
    return pairs

def build_parent_child_pairs_from_codes(codes: List[str]) -> List[Tuple[str, str]]:
    """
    Build parent-child pairs for all codes.
    
    Args:
        codes: List of ICD codes
        
    Returns:
        List of (parent, child) tuples representing hierarchy relationships
    """
    pairs = []
    
    for code in codes:
        code_pairs = extract_icd_parent_child_pair(code)
        pairs.extend(code_pairs)
    
    # Remove duplicates while preserving order
    unique_pairs = []
    seen = set()
    for pair in pairs:
        if pair not in seen:
            unique_pairs.append(pair)
            seen.add(pair)
    
    return unique_pairs

# ---------------------------
# Hyperbolic embedding model
# ---------------------------
class PoincareEmbedding(nn.Module):
    def __init__(self, num_nodes: int, dim: int = 16, init_scale=1e-3, 
                 hierarchy_init: bool = False, code2id: Dict[str, int] = None):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, dim)
        
        if hierarchy_init and code2id is not None:
            # Initialize with hierarchy-aware strategy
            self._hierarchy_aware_init(code2id, init_scale)
        else:
            # Initialize all embeddings randomly
            nn.init.uniform_(self.emb.weight, a=-init_scale, b=init_scale)
        
        with torch.no_grad():
            self.emb.weight.copy_(mobius_proj(self.emb.weight))
    
    def _hierarchy_aware_init(self, code2id: Dict[str, int], init_scale: float):
        """Initialize embeddings with hierarchy awareness: parents closer to origin"""
        id2code = {i: c for c, i in code2id.items()}
        
        with torch.no_grad():
            for node_id in range(len(code2id)):
                code = id2code[node_id]
                
                # Determine hierarchy level based on code length
                if len(code) <= 3:
                    # Top-level codes (e.g., "E11") - closest to origin
                    scale = init_scale * 0.1
                elif len(code) <= 5:
                    # Mid-level codes (e.g., "E11.6") - medium distance
                    scale = init_scale * 0.5
                else:
                    # Leaf-level codes (e.g., "E11.65") - furthest from origin
                    scale = init_scale * 1.0
                
                # Initialize with smaller scale for higher hierarchy levels
                nn.init.uniform_(self.emb.weight[node_id], a=-scale, b=scale)
    
    def forward(self, idx):
        return mobius_proj(self.emb(idx))

# ---------------------------
# Training utilities
# ---------------------------
class HierPairs(torch.utils.data.IterableDataset):
    """Stream positive (parent,child) pairs with on-the-fly negatives."""
    def __init__(self, id_edges: List[Tuple[int,int]], num_nodes: int, neg_k=10):
        super().__init__()
        self.pos = id_edges
        self.num_nodes = num_nodes
        self.neg_k = neg_k
        # adjacency for quick "avoid trivial negatives" if desired
        self.adj = {i:set() for i in range(num_nodes)}
        for p,c in id_edges:
            self.adj[p].add(c)
            self.adj[c].add(p)
    
    def __iter__(self):
        while True:
            p, c = random.choice(self.pos)
            negs = []
            while len(negs) < self.neg_k:
                j = random.randrange(self.num_nodes)
                if j != p and j != c and (j not in self.adj[p]):
                    negs.append(j)
            yield p, c, torch.tensor(negs, dtype=torch.long)

class ParentChildDataset(torch.utils.data.IterableDataset):
    """Dataset that samples parent-child pairs for hyperbolic embedding training."""
    def __init__(self, parent_child_pairs: List[Tuple[str, str]], code2id: Dict[str, int], 
                 num_nodes: int, neg_k=10):
        super().__init__()
        self.parent_child_pairs = parent_child_pairs
        self.code2id = code2id
        self.num_nodes = num_nodes
        self.neg_k = neg_k
        
        # Build adjacency for negative sampling
        self.adj = {i: set() for i in range(num_nodes)}
        for parent, child in parent_child_pairs:
            if parent in code2id and child in code2id:
                parent_id = code2id[parent]
                child_id = code2id[child]
                self.adj[parent_id].add(child_id)
                self.adj[child_id].add(parent_id)
    
    def __iter__(self):
        while True:
            # Sample a random parent-child pair
            parent, child = random.choice(self.parent_child_pairs)
            
            if parent not in self.code2id or child not in self.code2id:
                continue
                
            parent_id = self.code2id[parent]
            child_id = self.code2id[child]
            
            # Generate negative samples
            negs = []
            while len(negs) < self.neg_k:
                j = random.randrange(self.num_nodes)
                if j != parent_id and j != child_id and (j not in self.adj[parent_id]):
                    negs.append(j)
            
            # All parent-child pairs have equal weight (1.0)
            weight = 1.0
            
            yield parent_id, child_id, torch.tensor(negs, dtype=torch.long), weight

def reconstruction_loss(model: PoincareEmbedding, p_idx, c_idx, negs_idx, weights=None):
    """Loss function for hyperbolic embedding training with optional weights"""
    p = model(p_idx)         # [B, d]
    c = model(c_idx)         # [B, d]
    negs = model(negs_idx)   # [B, K, d]
    # distances
    d_pos = poincare_dist(p, c)                  # [B]
    d_neg = poincare_dist(p.unsqueeze(1), negs)  # [B, K]
    # logits: higher = better ⇒ use -distance
    logits = torch.cat([(-d_pos).unsqueeze(1), -d_neg], dim=1)  # [B, 1+K]
    targets = torch.zeros(p.size(0), dtype=torch.long, device=logits.device)  # positive at index 0
    
    loss = F.cross_entropy(logits, targets, reduction='none')  # [B]
    
    # Apply weights if provided
    if weights is not None:
        loss = loss * weights
    
    return loss.mean()

def hierarchy_constraint_loss(model: PoincareEmbedding, p_idx, c_idx, lambda_hierarchy=1.0):
    """
    Hierarchy constraint loss to ensure parent codes are closer to origin than child codes.
    Uses a more numerically stable approach.
    
    Args:
        model: PoincareEmbedding model
        p_idx: Parent indices [B]
        c_idx: Child indices [B] 
        lambda_hierarchy: Weight for hierarchy constraint
    
    Returns:
        Hierarchy constraint loss
    """
    p = model(p_idx)  # [B, d]
    c = model(c_idx)  # [B, d]
    
    # Use squared norms directly for numerical stability
    # Higher hierarchy (closer to origin) should have smaller squared norm
    p_norm_sq = (p * p).sum(dim=-1)  # [B]
    c_norm_sq = (c * c).sum(dim=-1)  # [B]
    
    # Constraint: parent should have smaller squared norm than child
    # Loss = max(0, child_norm_sq - parent_norm_sq)
    hierarchy_loss = F.relu(c_norm_sq - p_norm_sq)
    
    return lambda_hierarchy * hierarchy_loss.mean()

# ---------------------------
# Main training function
# ---------------------------
def train_conditions_hyperbolic_embedding(
    conditions_codes: List[str],
    dim: int = 20,
    neg_k: int = 10,
    steps: int = 100,
    batch_size: int = 256,
    lr: float = 1e-4,  # More conservative learning rate
    lambda_hierarchy: float = 0.5,  # Reduced hierarchy constraint weight
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Train hyperbolic embeddings for conditions codes using parent-child relationships
    
    Args:
        conditions_codes: List of condition codes (e.g., ICD-10 codes)
        dim: Embedding dimension
        neg_k: Number of negative samples per positive pair
        steps: Number of training steps
        batch_size: Batch size for training
        lr: Learning rate
        lambda_hierarchy: Weight for hierarchy constraint loss
        device: Device to use for training
    
    Returns:
        Dictionary mapping condition codes to their hyperbolic embeddings
    """
    import random
    
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
    print(f"Parent-child pairs: {len(parent_child_pairs)}")
    
    if len(parent_child_pairs) == 0:
        print("Warning: No parent-child relationships found. Using random initialization.")
        model = PoincareEmbedding(num_nodes, dim=dim, hierarchy_init=False).to(device)
        with torch.no_grad():
            emb = mobius_proj(model.emb.weight.data).cpu()
        id2code = {i: c for c, i in code2id.items()}
        return {id2code[i]: emb[i] for i in range(num_nodes)}
    
    # Create dataset and loader using parent-child pairs
    ds = ParentChildDataset(parent_child_pairs, code2id, num_nodes, neg_k=neg_k)
    
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)
    
    # Create model with hierarchy-aware initialization
    model = PoincareEmbedding(num_nodes, dim=dim, hierarchy_init=True, code2id=code2id).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8)  # Better numerical stability
    
    # Training loop
    model.train()
    it = iter(loader)
    
    for step in range(1, steps + 1):
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
            
            opt.zero_grad()
            total_loss.backward()
            
            # Gradient clipping for numerical stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            opt.step()
            
            # Re-project after optimizer step
            with torch.no_grad():
                model.emb.weight.copy_(mobius_proj(model.emb.weight))
            
            if step % 10 == 0:
                print(f"Hyperbolic embedding step {step:6d}  recon_loss = {recon_loss.item():.4f}  hierarchy_loss = {hierarchy_loss.item():.4f}  total_loss = {total_loss.item():.4f}")
                
        except StopIteration:
            # Restart iterator if it runs out
            it = iter(loader)
            continue
    
    # Return code → embedding dict (torch tensor)
    with torch.no_grad():
        emb = mobius_proj(model.emb.weight.data).cpu()
    
    id2code = {i: c for c, i in code2id.items()}
    return {id2code[i]: emb[i] for i in range(num_nodes)}

# ---------------------------
# Integration utilities
# ---------------------------
class ConditionsHyperbolicEmbedder:
    """Wrapper class for using hyperbolic embeddings in the dialysis prediction pipeline"""
    
    def __init__(self, conditions_codes: List[str], embedding_dim: int = 16):
        self.conditions_codes = conditions_codes
        self.embedding_dim = embedding_dim
        self.code2embedding = None
        self.trained = False
    
    def train_embeddings(self, lambda_hierarchy: float = 1.0, **kwargs):
        """Train hyperbolic embeddings for conditions codes with hierarchy constraints"""
        self.code2embedding = train_conditions_hyperbolic_embedding(
            self.conditions_codes,
            dim=self.embedding_dim,
            lambda_hierarchy=lambda_hierarchy,
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
        Get hyperbolic embedding sequences for a list of conditions (for transformer)
        
        Args:
            conditions_list: List of condition codes
            
        Returns:
            Embedding tensor of shape [n, embedding_dim] where n is the number of conditions
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
            # Return empty tensor with correct shape
            return torch.zeros(0, self.embedding_dim)
        
        # Return sequence of embeddings without averaging
        return torch.stack(embeddings)
    
    def get_embedding_dim(self) -> int:
        """Get the total embedding dimension for a single condition"""
        return self.embedding_dim
    
    def initialize_hierarchical_embeddings(self, output_file: str = "hyperbolic_embeddings.pkl") -> Dict[str, torch.Tensor]:
        """
        Initialize embeddings in Poincare ball according to hierarchical rules:
        1. All codes on the same level (uniquely decided by code length) should be equally far from origin
        2. Codes with shorter length (lower level) should be closer to origin
        3. For every code on level n, the nearest code on level n-1 should be its parent code
        4. The parent code for the lowest level code (length=3) is origin of Poincare ball
        
        Args:
            output_file: Path to save the initialized embeddings
            
        Returns:
            Dictionary mapping condition codes to their initialized hyperbolic embeddings
        """
        import pickle
        import numpy as np
        from collections import defaultdict
        
        print(f"Initializing hierarchical embeddings for {len(self.conditions_codes)} condition codes...")
        
        # Group codes by hierarchy level (code length)
        codes_by_level = defaultdict(list)
        for code in self.conditions_codes:
            level = len(code)
            codes_by_level[level].append(code)
        
        # Sort levels (shorter codes = higher hierarchy = closer to origin)
        sorted_levels = sorted(codes_by_level.keys())
        print(f"Found hierarchy levels: {sorted_levels}")
        
        # Initialize embeddings dictionary
        code2embedding = {}
        
        # Define radius for each level (closer to origin for higher hierarchy)
        # Level 3 (highest hierarchy) gets smallest radius, increasing for lower levels
        level_radii = {}
        max_level = max(sorted_levels) if sorted_levels else 3
        min_level = min(sorted_levels) if sorted_levels else 3
        
        for level in sorted_levels:
            # Normalize level to [0, 1] range, then scale to [0.1, 0.9] for Poincare ball
            normalized_level = (level - min_level) / (max_level - min_level) if max_level > min_level else 0
            radius = 0.1 + 0.8 * normalized_level  # Range from 0.1 to 0.9
            level_radii[level] = radius
        
        print(f"Level radii: {level_radii}")
        
        # Initialize embeddings for each level
        for level in sorted_levels:
            codes_at_level = codes_by_level[level]
            radius = level_radii[level]
            
            print(f"Initializing level {level} with {len(codes_at_level)} codes at radius {radius:.3f}")
            
            # Generate points on sphere at given radius
            embeddings_at_level = self._generate_points_on_sphere(
                len(codes_at_level), self.embedding_dim, radius
            )
            
            # Assign embeddings to codes
            for i, code in enumerate(codes_at_level):
                code2embedding[code] = embeddings_at_level[i]
        
        # Apply parent-child proximity constraints
        print("Applying parent-child proximity constraints...")
        code2embedding = self._apply_parent_child_constraints(code2embedding, codes_by_level)
        
        # Save embeddings to file
        print(f"Saving initialized embeddings to {output_file}")
        with open(output_file, 'wb') as f:
            pickle.dump(code2embedding, f)
        
        print(f"Successfully initialized and saved hierarchical embeddings!")
        return code2embedding
    
    def _generate_points_on_sphere(self, n_points: int, dim: int, radius: float) -> torch.Tensor:
        """
        Generate n_points uniformly distributed on a sphere of given radius in dim dimensions
        
        Args:
            n_points: Number of points to generate
            dim: Dimension of the embedding space
            radius: Radius of the sphere (distance from origin)
            
        Returns:
            Tensor of shape [n_points, dim] with points on the sphere
        """
        if n_points == 0:
            return torch.empty(0, dim)
        
        # Generate random points in unit ball
        points = torch.randn(n_points, dim)
        
        # Normalize to unit sphere
        norms = torch.norm(points, dim=1, keepdim=True)
        points = points / norms
        
        # Scale to desired radius
        points = points * radius
        
        return points
    
    def _apply_parent_child_constraints(self, code2embedding: Dict[str, torch.Tensor], 
                                      codes_by_level: Dict[int, List[str]]) -> Dict[str, torch.Tensor]:
        """
        Apply parent-child proximity constraints to ensure children are near their parents
        
        Args:
            code2embedding: Dictionary mapping codes to embeddings
            codes_by_level: Dictionary mapping level to list of codes at that level
            
        Returns:
            Updated dictionary with parent-child constraints applied
        """
        sorted_levels = sorted(codes_by_level.keys())
        
        # For each level (except the highest), adjust children to be near their parents
        for i in range(1, len(sorted_levels)):
            current_level = sorted_levels[i]
            parent_level = sorted_levels[i-1]
            
            codes_at_level = codes_by_level[current_level]
            parent_codes = codes_by_level[parent_level]
            
            for child_code in codes_at_level:
                # Find the parent code (first parent_level characters)
                parent_code = child_code[:parent_level]
                
                if parent_code in code2embedding:
                    # Get parent embedding
                    parent_embedding = code2embedding[parent_code]
                    
                    # Generate new child embedding near parent
                    child_embedding = self._generate_child_near_parent(
                        parent_embedding, self.embedding_dim, 
                        current_level, sorted_levels
                    )
                    
                    code2embedding[child_code] = child_embedding
        
        return code2embedding
    
    def _generate_child_near_parent(self, parent_embedding: torch.Tensor, dim: int, 
                                  child_level: int, all_levels: List[int]) -> torch.Tensor:
        """
        Generate a child embedding near its parent
        
        Args:
            parent_embedding: Parent's embedding vector
            dim: Embedding dimension
            child_level: Level of the child code
            all_levels: List of all levels for radius calculation
            
        Returns:
            Child embedding vector near the parent
        """
        # Calculate radius for child level
        min_level = min(all_levels)
        max_level = max(all_levels)
        normalized_level = (child_level - min_level) / (max_level - min_level) if max_level > min_level else 0
        child_radius = 0.1 + 0.8 * normalized_level
        
        # Generate random direction
        direction = torch.randn(dim)
        direction = direction / torch.norm(direction)
        
        # Scale to child radius
        child_embedding = direction * child_radius
        
        # Add small perturbation towards parent (but maintain radius constraint)
        parent_direction = parent_embedding / torch.norm(parent_embedding)
        perturbation_strength = 0.1  # How much to move towards parent
        
        # Blend child direction with parent direction
        blended_direction = (1 - perturbation_strength) * direction + perturbation_strength * parent_direction
        blended_direction = blended_direction / torch.norm(blended_direction)
        
        # Final child embedding
        child_embedding = blended_direction * child_radius
        
        return child_embedding