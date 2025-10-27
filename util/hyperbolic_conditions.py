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

def get_icd_to_ccs_mapping(icd_codes: List[str]) -> Dict[str, str]:
    """Map ICD-10-CM codes to their primary CCS codes using pyhealth"""
    from pyhealth.medcode import CrossMap
    mapping = CrossMap("ICD10CM", "CCSCM")
    icd_to_ccs = {}
    for icd_code in icd_codes:
        ccs_result = mapping.map(icd_code)
        if ccs_result:
            icd_to_ccs[icd_code] = ccs_result[0]  # Use primary CCS code
    return icd_to_ccs

def extract_icd_parent_child_pair(code: str, ccs_code: str = None) -> List[Tuple[str, str]]:
    if not code or len(code) <= 3:
        return []
    
    pairs = []
    
    # Add CCS → root ICD pair if CCS code is provided
    if ccs_code is not None and len(code) >= 3:
        root_icd = code[:3]
        pairs.append((ccs_code, root_icd))
    
    # Add ICD hierarchy pairs
    if len(code) >= 4:
        for n in range(4, len(code) + 1):
            parent = code[:n-1]
            child = code[:n]
            pairs.append((parent, child))
    
    return pairs

def build_parent_child_pairs_from_codes(codes: List[str], icd_to_ccs: Dict[str, str] = None) -> List[Tuple[str, str]]:
    """
    Build parent-child pairs for all codes.
    
    Args:
        codes: List of ICD codes
        icd_to_ccs: Optional mapping from ICD codes to CCS codes
        
    Returns:
        List of (parent, child) tuples representing hierarchy relationships
    """
    pairs = []
    
    for code in codes:
        # Get CCS code for this ICD code if available
        ccs_code = icd_to_ccs.get(code, None) if icd_to_ccs else None
        code_pairs = extract_icd_parent_child_pair(code, ccs_code)
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
                 hierarchy_init: bool = False, origin_init: bool = False, code2id: Dict[str, int] = None):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, dim)
        
        if hierarchy_init and code2id is not None:
            # Initialize with hierarchy-aware strategy
            self._hierarchy_aware_init(code2id, init_scale)
        elif origin_init:
            # Initialize all codes near origin
            self._origin_init(init_scale)
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
                
                # Check if this is a CCS code
                # CCS codes are either:
                # 1. Numeric strings (e.g., "1", "2", "123")
                # 2. "UNKNOWN_CCS" special code
                # ICD codes always contain at least one letter followed by digits
                # and may contain dots (e.g., "E11", "E11.6", "I25.110")
                is_ccs_code = (code == "UNKNOWN_CCS" or 
                               (len(code) > 0 and code[0].isdigit() and not any(c in code for c in "./-")))
                
                # Determine hierarchy level based on code type and length
                if is_ccs_code:
                    # CCS codes are at the root level - closest to origin
                    scale = init_scale * 0.05
                elif len(code) <= 3:
                    # Top-level ICD codes (e.g., "E11") - close to origin
                    scale = init_scale * 0.1
                elif len(code) <= 5:
                    # Mid-level ICD codes (e.g., "E11.6") - medium distance
                    scale = init_scale * 0.5
                else:
                    # Leaf-level ICD codes (e.g., "E11.65") - furthest from origin
                    scale = init_scale * 1.0
                
                # Initialize with smaller scale for higher hierarchy levels
                nn.init.uniform_(self.emb.weight[node_id], a=-scale, b=scale)
    
    def _origin_init(self, init_scale: float):
        """Initialize all embeddings near the origin with small random values"""
        with torch.no_grad():
            # Use a smaller scale to keep all embeddings close to origin
            origin_scale = init_scale * 0.1  # Much smaller than default
            nn.init.uniform_(self.emb.weight, a=-origin_scale, b=origin_scale)
    
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

def cone_cohesion_loss(model: PoincareEmbedding, ccs_idx, children_indices, lambda_cone=1.0):
    """
    Cone cohesion loss to ensure children of the same CCS code are in the same cone.
    
    In hyperbolic space, a "cone" is the set of points that share the same direction from origin.
    This loss encourages children to have similar normalized directions (same cone).
    
    Args:
        model: PoincareEmbedding model
        ccs_idx: CCS code index [B]
        children_indices: List of child indices for each CCS [B, variable]
        lambda_cone: Weight for cone cohesion constraint
    
    Returns:
        Cone cohesion loss
    """
    ccs_emb = model(ccs_idx)  # [B, d]
    
    losses = []
    for i in range(ccs_idx.size(0)):
        ccs = ccs_emb[i]  # [d]
        children = model(children_indices[i])  # [num_children, d]
        
        # Compute normalized direction vectors (unit vectors)
        ccs_norm = torch.norm(ccs, dim=-1, keepdim=True).clamp_min(EPS)
        ccs_direction = ccs / ccs_norm  # [d]
        
        children_norms = torch.norm(children, dim=-1, keepdim=True).clamp_min(EPS)
        children_directions = children / children_norms  # [num_children, d]
        
        # Cosine similarity between CCS direction and each child direction
        # Higher similarity means same cone
        cos_sim = (ccs_direction.unsqueeze(0) * children_directions).sum(dim=-1)  # [num_children]
        
        # Loss = negative cosine similarity (we want high similarity)
        # This encourages children to be in the same cone as the CCS
        loss_per_ccs = (1 - cos_sim).mean()  # Average over children
        losses.append(loss_per_ccs)
    
    if len(losses) == 0:
        return torch.tensor(0.0, device=ccs_idx.device)
    
    return lambda_cone * torch.stack(losses).mean()

# ---------------------------
# Main training function
# ---------------------------
def train_conditions_hyperbolic_embedding(
    conditions_codes: List[str],
    dim: int = 20,
    neg_k: int = 10,
    steps: int = 100,
    batch_size: int = 256,
    lr: float = 1e-4,  # More conservative penalty rate
    lambda_hierarchy: float = 0.5,  # Reduced hierarchy constraint weight
    lambda_cone: float = 1.0,  # Weight for cone cohesion loss
    origin_init: bool = False,  # Initialize all codes near origin
    icd_to_ccs: Dict[str, str] = None,  # Mapping from ICD codes to CCS codes
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
        lambda_cone: Weight for cone cohesion loss
        origin_init: If True, initialize all codes near origin instead of hierarchy-aware init
        icd_to_ccs: Mapping from ICD codes to CCS codes
        device: Device to use for training
    
    Returns:
        Dictionary mapping condition codes to their hyperbolic embeddings
    """
   # Build vocabulary - include original codes and all their parents/children
    original_codes = sorted(list(set(conditions_codes)))
    
    # Get parent-child pairs to extract all related codes (includes CCS → ICD pairs)
    parent_child_pairs = build_parent_child_pairs_from_codes(original_codes, icd_to_ccs=icd_to_ccs)
    
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
    
    # Build CCS to children mapping for cone cohesion loss
    ccs_to_children = defaultdict(set)
    if icd_to_ccs is not None:
        for icd_code, ccs_code in icd_to_ccs.items():
            if icd_code in code2id and ccs_code in code2id:
                ccs_to_children[ccs_code].add(icd_code)
        
        # Convert to lists and filter out CCS with too few children
        ccs_to_children = {ccs: list(children) for ccs, children in ccs_to_children.items() 
                           if len(children) >= 2}  # Need at least 2 children for cone loss
        print(f"CCS codes with children for cone loss: {len(ccs_to_children)}")
    
    if len(parent_child_pairs) == 0:
        print("Warning: No parent-child relationships found. Using random initialization.")
        model = PoincareEmbedding(num_nodes, dim=dim, hierarchy_init=False, origin_init=origin_init).to(device)
        with torch.no_grad():
            emb = mobius_proj(model.emb.weight.data).cpu()
        id2code = {i: c for c, i in code2id.items()}
        return {id2code[i]: emb[i] for i in range(num_nodes)}
    
    # Create dataset and loader using parent-child pairs
    ds = ParentChildDataset(parent_child_pairs, code2id, num_nodes, neg_k=neg_k)
    
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)
    
    # Create model with specified initialization
    if origin_init:
        print("Using origin initialization: all codes initialized near origin")
        model = PoincareEmbedding(num_nodes, dim=dim, origin_init=True).to(device)
    else:
        print("Using hierarchy-aware initialization")
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
            
            # Compute cone cohesion loss (if CCS mapping available)
            cone_loss = torch.tensor(0.0, device=device)
            if icd_to_ccs is not None and len(ccs_to_children) > 0 and lambda_cone > 0:
                # Sample a few CCS codes for cone loss computation
                sampled_ccs = random.sample(list(ccs_to_children.keys()), 
                                           min(batch_size // 4, len(ccs_to_children)))
                
                for ccs_code in sampled_ccs:
                    children_codes = ccs_to_children[ccs_code]
                    ccs_idx = torch.tensor([code2id[ccs_code]], device=device)
                    children_idx = torch.tensor([code2id[c] for c in children_codes], device=device)
                    
                    # Compute cone loss for this CCS (don't apply lambda_cone here, it's applied in the function)
                    cone_loss += cone_cohesion_loss(model, ccs_idx, [children_idx], lambda_cone=1.0)
                
                # Average over sampled CCS codes and apply lambda_cone
                if len(sampled_ccs) > 0:
                    cone_loss = (cone_loss / len(sampled_ccs)) * lambda_cone
            
            # Check for NaN values
            if torch.isnan(recon_loss) or torch.isnan(hierarchy_loss) or torch.isnan(cone_loss):
                print(f"Warning: NaN detected at step {step}. Stopping training.")
                break
            
            # Total loss
            total_loss = recon_loss + hierarchy_loss + cone_loss
            
            opt.zero_grad()
            total_loss.backward()
            
            # Gradient clipping for numerical stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            opt.step()
            
            # Re-project after optimizer step
            with torch.no_grad():
                model.emb.weight.copy_(mobius_proj(model.emb.weight))
            
            if step % 10 == 0:
                print(f"Step {step:6d}  recon_loss = {recon_loss.item():.4f}  hierarchy_loss = {hierarchy_loss.item():.4f}  cone_loss = {cone_loss.item():.4f}  total_loss = {total_loss.item():.4f}")
                
        except StopIteration:
            # Restart iterator if it runs out
            it = iter(loader)
            continue
    
    # Return code → embedding dict (torch tensor)
    with torch.no_grad():
        emb = mobius_proj(model.emb.weight.data).cpu()
    
    id2code = {i: c for c, i in code2id.items()}
    
    # CRITICAL FIX: Use .clone() to avoid shared storage
    # Without clone(), emb[i] creates a view that references the entire emb tensor
    # This causes pickle to serialize the entire tensor for each dictionary entry (massive duplication!)
    # Fixed: 772GB -> ~8MB by ensuring each embedding is independent
    return {id2code[i]: emb[i].clone() for i in range(num_nodes)}

# ---------------------------
# Integration utilities
# ---------------------------
class ConditionsHyperbolicEmbedder:
    def __init__(self, conditions_codes: List[str], embedding_dim: int = 16, icd_to_ccs: Dict[str, str] = None):
        self.conditions_codes = conditions_codes
        self.embedding_dim = embedding_dim
        self.icd_to_ccs = icd_to_ccs
        self.code2embedding = None
        self.trained = False
    
    def train_embeddings(self, lambda_hierarchy: float = 1.0, lambda_cone: float = 1.0, origin_init: bool = False, **kwargs):
        """Train hyperbolic embeddings for conditions codes with hierarchy constraints"""
        self.code2embedding = train_conditions_hyperbolic_embedding(
            self.conditions_codes,
            dim=self.embedding_dim,
            lambda_hierarchy=lambda_hierarchy,
            lambda_cone=lambda_cone,
            origin_init=origin_init,
            icd_to_ccs=self.icd_to_ccs,
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
