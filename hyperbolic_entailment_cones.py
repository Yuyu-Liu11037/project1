# -*- coding: utf-8 -*-
import math
import random
import argparse
import pickle
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Utilities for Poincaré ball
# =========================

class PoincareOps:
    @staticmethod
    def lambda_x(x: torch.Tensor) -> torch.Tensor:
        # λ_x = 2 / (1 - ||x||^2)
        x2 = (x * x).sum(dim=-1, keepdim=True)
        return 2.0 / (1.0 - x2).clamp(min=1e-15)

    @staticmethod
    def proj_to_ball(x: torch.Tensor, max_norm: float = 1 - 1e-5, min_norm: float = 0.0) -> torch.Tensor:
        # Project to annulus: ||x|| in [min_norm, max_norm]
        norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-15)
        x = x * (max_norm / norm).where(norm > max_norm, torch.ones_like(norm))
        if min_norm > 0:
            x = x * (min_norm / norm).where(norm < min_norm, torch.ones_like(norm))
        return x

    @staticmethod
    def exp_map(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map on the Poincaré ball (closed form, Eq. (7) in paper).
        Added numerical stability: clipping lam*v_norm and zero-division protection.
        x, v: (..., d)
        """
        # λ_x
        lam = PoincareOps.lambda_x(x)  # (..., 1)

        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-15)
        # inner <x, v/||v||>
        xv = (x * (v / v_norm)).sum(dim=-1, keepdim=True)

        # Numerical stability: clip lam * v_norm to prevent overflow in cosh/sinh
        lam_v_norm = (lam * v_norm).clamp(max=50.0)  # cosh(50) is still safe, prevents inf

        # Use clipped value for hyperbolic functions
        cosh_val = torch.cosh(lam_v_norm)
        sinh_val = torch.sinh(lam_v_norm)

        a = lam * (cosh_val + xv * sinh_val)
        b = 1.0 + (lam - 1.0) * cosh_val + lam * xv * sinh_val
        
        # Zero-division protection
        b = b.clamp(min=1e-15)
        
        term1 = (a / b) * x
        term2 = (sinh_val / (v_norm * b)) * v
        y = term1 + term2
        
        # Final safety check: if NaN or Inf occurred, return original x (no update)
        if torch.isnan(y).any() or torch.isinf(y).any():
            return x
        
        return y

    @staticmethod
    def angle_Xi(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Ξ(u,v) = arccos( numerator / denominator )  —— Eq. (28)
        where
        num = <u,v>(1+||u||^2) - ||u||^2(1+||v||^2)
        den = ||u|| * ||u - v|| * sqrt(1 + ||u||^2||v||^2 - 2<u,v>)
        """
        # norms and dot
        uu = (u * u).sum(dim=-1)
        vv = (v * v).sum(dim=-1)
        uv = (u * v).sum(dim=-1)

        num = uv * (1 + uu) - uu * (1 + vv)
        u_minus_v = (u - v)
        u_minus_v_norm = u_minus_v.norm(dim=-1).clamp(min=1e-15)
        u_norm = u.norm(dim=-1).clamp(min=1e-15)

        inside = (1 + uu * vv - 2 * uv).clamp(min=1e-15)
        den = u_norm * u_minus_v_norm * torch.sqrt(inside)

        cos_val = (num / den).clamp(min=-1.0 + 1e-7, max=1.0 - 1e-7)
        return torch.acos(cos_val)  # radians

    @staticmethod
    def psi(u: torch.Tensor, K: float, eps: float) -> torch.Tensor:
        """
        ψ(u) = arcsin( K * (1 - ||u||^2) / ||u|| ) —— Eq. (26)
        We clamp the argument within [-1,1], and only define for ||u|| >= eps.
        """
        u_norm = u.norm(dim=-1).clamp(min=eps)  # respect domain D \ B(0, eps)
        arg = K * (1.0 - u_norm * u_norm) / u_norm
        arg = arg.clamp(min=-1.0 + 1e-7, max=1.0 - 1e-7)
        return torch.asin(arg)  # radians

    @staticmethod
    def poincare_distance(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-15) -> torch.Tensor:
        """
        Compute Poincaré distance between two points on the Poincaré ball.
        d(u,v) = arccosh(1 + 2 * ||u - v||² / ((1 - ||u||²) * (1 - ||v||²)))
        
        Args:
            u: First point tensor (..., d)
            v: Second point tensor (..., d)
            eps: Small epsilon for numerical stability
            
        Returns:
            Poincaré distance (...,)
        """
        # Compute ||u - v||²
        uv_sq = ((u - v) * (u - v)).sum(dim=-1)
        
        # Compute 1 - ||u||² and 1 - ||v||² with clamping for numerical stability
        uu = torch.clamp(1.0 - (u * u).sum(dim=-1), min=eps)
        vv = torch.clamp(1.0 - (v * v).sum(dim=-1), min=eps)
        
        # Compute the argument for arccosh: 1 + 2 * ||u-v||² / ((1-||u||²)(1-||v||²))
        x = 1.0 + 2.0 * uv_sq / (uu * vv)
        
        # Clamp x to prevent arccosh from producing NaN or Inf
        # arccosh is defined for x >= 1, and we clamp upper bound to prevent overflow
        x = torch.clamp(x, min=1.0 + eps, max=1e6)
        
        return torch.acosh(x)

    @staticmethod
    def log_map(x: torch.Tensor, y: torch.Tensor, c: float = 1.0, eps: float = 1e-15) -> torch.Tensor:
        """
        Logarithmic map: maps point y in Poincaré ball to tangent space at x.
        log_x(y) = (2/λ_x) * artanh(||-x ⊕_c y||) * (-x ⊕_c y) / ||-x ⊕_c y||
        
        Args:
            x: Base point in Poincaré ball (..., d)
            y: Point to map to tangent space (..., d)
            c: Curvature parameter (default: 1.0)
            eps: Small epsilon for numerical stability
            
        Returns:
            Vector in tangent space at x (..., d)
        """
        # Compute -x ⊕_c y (Möbius addition of -x and y)
        # Direct computation to avoid circular dependency
        neg_x = -x
        neg_x2 = (neg_x * neg_x).sum(dim=-1, keepdim=True)
        y2 = (y * y).sum(dim=-1, keepdim=True)
        neg_xy = (neg_x * y).sum(dim=-1, keepdim=True)
        
        denominator = 1.0 + 2.0 * c * neg_xy + c * c * neg_x2 * y2
        denominator = denominator.clamp(min=1e-15)
        
        numerator_x = (1.0 + 2.0 * c * neg_xy + c * y2) * neg_x
        numerator_y = (1.0 - c * neg_x2) * y
        mobius_sum = (numerator_x + numerator_y) / denominator
        mobius_sum = PoincareOps.proj_to_ball(mobius_sum, max_norm=1.0 - 1e-5)
        
        # Compute norm of Möbius sum
        mobius_norm = mobius_sum.norm(dim=-1, keepdim=True).clamp(min=eps)
        
        # Compute artanh of norm
        artanh_arg = (mobius_norm).clamp(min=eps, max=1.0 - eps)
        artanh_val = torch.atanh(artanh_arg)
        
        # Get λ_x
        lam = PoincareOps.lambda_x(x)
        
        # Compute log map
        log_map_result = (2.0 / lam) * artanh_val * (mobius_sum / mobius_norm)
        
        # Safety check
        if torch.isnan(log_map_result).any() or torch.isinf(log_map_result).any():
            return torch.zeros_like(y)
        
        return log_map_result

    @staticmethod
    def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """
        Möbius addition in Poincaré ball: x ⊕_c y
        Formula: ((1 + 2c<x,y> + c||y||²)x + (1 - c||x||²)y) / (1 + 2c<x,y> + c²||x||²||y||²)
        
        Args:
            x: First point (..., d)
            y: Second point (..., d)
            c: Curvature parameter (default: 1.0)
            
        Returns:
            Möbius sum x ⊕_c y (..., d)
        """
        x2 = (x * x).sum(dim=-1, keepdim=True)
        y2 = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        
        denominator = 1.0 + 2.0 * c * xy + c * c * x2 * y2
        denominator = denominator.clamp(min=1e-15)
        
        numerator_x = (1.0 + 2.0 * c * xy + c * y2) * x
        numerator_y = (1.0 - c * x2) * y
        
        result = (numerator_x + numerator_y) / denominator
        
        # Project back to ball for numerical stability
        result = PoincareOps.proj_to_ball(result, max_norm=1.0 - 1e-5)
        
        # Safety check
        if torch.isnan(result).any() or torch.isinf(result).any():
            return x
        
        return result

    @staticmethod
    def mobius_matvec(M: torch.Tensor, x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """
        Möbius matrix-vector multiplication: M ⊗_c x
        Formula: (1/√c) * tanh(||M·artanh(√c·||x||)·x/||x||||) * M·artanh(√c·||x||)·x/||x|| / ||M·artanh(√c·||x||)·x/||x||||
        
        Simplified version for c=1: tanh(||M·v||) * M·v / ||M·v|| where v = artanh(||x||) * x / ||x||
        
        Args:
            M: Matrix (..., out_dim, in_dim) or (out_dim, in_dim)
            x: Vector in Poincaré ball (..., in_dim)
            c: Curvature parameter (default: 1.0)
            
        Returns:
            Result of Möbius matrix-vector multiplication (..., out_dim)
        """
        # Handle batched case
        x_norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-15, max=1.0 - 1e-5)
        
        # Compute artanh(||x||) * x / ||x|| (unit direction scaled by artanh of norm)
        artanh_norm = torch.atanh(x_norm.clamp(min=1e-15, max=1.0 - 1e-5))
        v = artanh_norm * (x / x_norm)
        
        # Apply matrix multiplication M·v
        # Handle different shapes of M
        if len(M.shape) == 2:
            # M is (out_dim, in_dim), x is (..., in_dim)
            Mv = torch.matmul(v, M.t())  # (..., out_dim)
        else:
            # M is (..., out_dim, in_dim)
            Mv = torch.matmul(v.unsqueeze(-2), M.transpose(-2, -1)).squeeze(-2)  # (..., out_dim)
        
        # Compute norm of Mv
        Mv_norm = Mv.norm(dim=-1, keepdim=True).clamp(min=1e-15)
        
        # Apply tanh and normalize
        result = (torch.tanh(Mv_norm) / Mv_norm) * Mv
        
        # Project back to ball
        result = PoincareOps.proj_to_ball(result, max_norm=1.0 - 1e-5)
        
        # Safety check
        if torch.isnan(result).any() or torch.isinf(result).any():
            return x
        
        return result

    @staticmethod
    def einstein_midpoint(x: torch.Tensor, weights: torch.Tensor = None, mask: torch.Tensor = None, c: float = 1.0, eps: float = 1e-15) -> torch.Tensor:
        """
        Einstein midpoint (weighted average in hyperbolic space).
        Uses iterative computation based on Möbius addition.
        
        Args:
            x: Points in Poincaré ball (batch_size, seq_len, dim) or (seq_len, dim)
            weights: Optional weights (batch_size, seq_len) or (seq_len,)
            mask: Optional mask indicating valid points (batch_size, seq_len) or (seq_len,)
            c: Curvature parameter (default: 1.0)
            eps: Small epsilon for numerical stability
            
        Returns:
            Einstein midpoint (batch_size, dim) or (dim,)
        """
        # Handle different input shapes
        if len(x.shape) == 2:
            # (seq_len, dim) -> add batch dimension
            x = x.unsqueeze(0)
            needs_squeeze = True
        else:
            needs_squeeze = False
        
        batch_size, seq_len, dim = x.shape
        
        # Initialize weights
        if weights is None:
            weights = torch.ones(batch_size, seq_len, device=x.device, dtype=x.dtype)
        
        # Apply mask if provided
        if mask is not None:
            if len(mask.shape) == 1:
                mask = mask.unsqueeze(0)
            weights = weights * mask.float()
        
        # Normalize weights
        weight_sum = weights.sum(dim=-1, keepdim=True).clamp(min=eps)
        weights = weights / weight_sum
        
        # Initialize midpoint as weighted average (in tangent space)
        # First, map all points to tangent space at origin, then average
        # For simplicity, use iterative Möbius addition
        # Start with first weighted point
        valid_indices = (weights > eps).any(dim=0)
        if not valid_indices.any():
            # All weights are zero, return zero vector
            midpoint = torch.zeros(batch_size, dim, device=x.device, dtype=x.dtype)
            if needs_squeeze:
                return midpoint.squeeze(0)
            return midpoint
        
        # Use weighted combination: iterate with Möbius addition
        # Simplified: use weighted average in tangent space
        # Map points to tangent space at origin, average, then map back
        midpoints = []
        for b in range(batch_size):
            # Get valid points for this batch
            batch_weights = weights[b]  # (seq_len,)
            batch_x = x[b]  # (seq_len, dim)
            
            # Find non-zero weights
            valid = batch_weights > eps
            if not valid.any():
                midpoints.append(torch.zeros(dim, device=x.device, dtype=x.dtype))
                continue
            
            valid_x = batch_x[valid]  # (num_valid, dim)
            valid_weights = batch_weights[valid]  # (num_valid,)
            valid_weights = valid_weights / valid_weights.sum()
            
            # Map to tangent space at origin, then weighted average
            # For each point: log_0(x) = artanh(||x||) * x / ||x||
            log_points = []
            for i in range(len(valid_x)):
                point = valid_x[i]
                point_norm = point.norm().clamp(min=eps, max=1.0 - 1e-5)
                if point_norm < eps:
                    log_point = torch.zeros_like(point)
                else:
                    artanh_norm = torch.atanh(point_norm.clamp(min=1e-15, max=1.0 - 1e-5))
                    log_point = artanh_norm * (point / point_norm)
                log_points.append(valid_weights[i] * log_point)
            
            # Sum in tangent space
            log_midpoint = sum(log_points)
            
            # Map back to Poincaré ball: exp_0(v) = tanh(||v||) * v / ||v||
            log_norm = log_midpoint.norm().clamp(min=eps)
            midpoint = torch.tanh(log_norm) * (log_midpoint / log_norm)
            
            # Project to ball
            midpoint = PoincareOps.proj_to_ball(midpoint.unsqueeze(0), max_norm=1.0 - 1e-5).squeeze(0)
            midpoints.append(midpoint)
        
        result = torch.stack(midpoints)  # (batch_size, dim)
        
        if needs_squeeze:
            result = result.squeeze(0)
        
        # Safety check
        if torch.isnan(result).any() or torch.isinf(result).any():
            return torch.zeros_like(result)
        
        return result


# =========================
# Hyperbolic Entailment Cones model
# =========================

class HyperbolicEntailmentCones(nn.Module):
    """
    ICD-10-CM code embeddings on the Poincaré ball with entailment cones.
    Each directed edge (parent -> child) means: parent entails child (child ⊂ parent).
    """
    def __init__(
        self,
        num_codes: int,
        dim: int = 10,
        eps: float = 0.1,
        K_scale: float = 0.9,  # choose K = K_scale * eps/(1-eps^2) to satisfy Eq. (25)
        init_radius: float = 0.1,
        seed: int = 42
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.num_codes = num_codes
        self.dim = dim
        self.eps = eps
        self.max_norm = 1.0 - 1e-5

        # K must satisfy: K <= eps/(1-eps^2) (Eq. 25)
        self.K_max = eps / (1.0 - eps * eps)
        self.K = float(K_scale) * self.K_max

        # embeddings in D^n
        self.emb = nn.Parameter(torch.empty(num_codes, dim))
        nn.init.normal_(self.emb, mean=0.0, std=1e-2)
        with torch.no_grad():
            # push away from origin and keep within ball
            self.emb.data = self.emb.data + init_radius * F.normalize(self.emb.data, dim=-1)
            self.emb.data = PoincareOps.proj_to_ball(self.emb.data, max_norm=self.max_norm, min_norm=self.eps)

    def forward(self, heads: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        u = self.emb[heads]  # parents (entailers)
        v = self.emb[tails]  # children (entailed)
        Xi = PoincareOps.angle_Xi(u, v)                    # Eq. (28)
        psi_u = PoincareOps.psi(u, K=self.K, eps=self.eps)  # Eq. (26)
        energy = torch.relu(Xi - psi_u)                     # Eq. (33)
        return energy

    def riemannian_step(self, lr: float):
        """
        One Riemannian SGD step:
        - scale Euclidean grad by (1/λ_x)^2 to get Riemannian grad (Eq. 36),
        - move by exponential map (Eq. 35 with Eq. 7),
        - project back into annulus D^n \ B(0, eps).
        """
        with torch.no_grad():
            x = self.emb.data
            grad = self.emb.grad

            lam = PoincareOps.lambda_x(x)  # (...,1)
            # ∇^R = (1/λ_x)^2 ∇
            rgrad = (1.0 / (lam * lam)) * grad
            
            # update via exp map
            self.emb.data = PoincareOps.exp_map(x, -lr * rgrad)

            # numerical safety
            self.emb.data = PoincareOps.proj_to_ball(self.emb.data,
                                                 max_norm=self.max_norm,
                                                 min_norm=self.eps)

        self.emb.grad.zero_()


# =========================
# Training helper
# =========================

def make_id_map(codes: List[str]) -> Dict[str, int]:
    return {c: i for i, c in enumerate(codes)}

def corrupt_tail(num_codes: int, head: int, tail: int) -> int:
    # sample a random tail different from the true one
    t = tail
    while t == tail:
        t = random.randrange(num_codes)
    return t

def corrupt_head(num_codes: int, head: int, tail: int) -> int:
    h = head
    while h == head:
        h = random.randrange(num_codes)
    return h

def build_batch(
    edges: List[Tuple[int, int]],
    num_codes: int,
    batch_size: int,
    neg_ratio: int = 5
):
    """
    edges: list of (parent_id, child_id) positive edges
    returns tensors of heads_pos, tails_pos, heads_neg, tails_neg
    """
    if len(edges) == 0:
        # keep original empty behavior
        return (torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long))

    positives = set(edges)  # NEW: for filtering negatives

    batch = random.sample(edges, k=min(batch_size, len(edges)))
    hp = torch.tensor([h for h, _ in batch], dtype=torch.long)
    tp = torch.tensor([t for _, t in batch], dtype=torch.long)

    def sample_negative(h, t):
        # NEW: resample until (hn, tn) is not a positive edge
        while True:
            if random.random() < 0.5:
                # corrupt tail
                tn = random.randrange(num_codes)
                hn = h
            else:
                hn = random.randrange(num_codes)
                tn = t
            if (hn, tn) not in positives:
                return hn, tn

    hn_list, tn_list = [], []
    for h, t in batch:
        for _ in range(neg_ratio):
            hn, tn = sample_negative(h, t)
            hn_list.append(hn)
            tn_list.append(tn)

    hn = torch.tensor(hn_list, dtype=torch.long)
    tn = torch.tensor(tn_list, dtype=torch.long)
    return hp, tp, hn, tn


def train_hyperbolic_cones(
    codes: List[str],
    parent_child_edges: List[Tuple[str, str]],
    dim: int = 10,
    lr: float = 1e-3,
    epochs: int = 200,
    batch_size: int = 256,
    neg_ratio: int = 10,
    margin: float = 1e-2,
    eps: float = 0.1,
    K_scale: float = 0.9,
    seed: int = 42,
    device: str = "cpu"
):
    """
    parent_child_edges: list of (parent_code, child_code), meaning parent entails child.
    Returns: trained model and code->id mapping
    """
    random.seed(seed)
    torch.manual_seed(seed)

    id_map = make_id_map(codes)
    edges_id = [(id_map[p], id_map[c]) for (p, c) in parent_child_edges if p in id_map and c in id_map]

    model = HyperbolicEntailmentCones(
        num_codes=len(codes),
        dim=dim,
        eps=eps,
        K_scale=K_scale,
        seed=seed
    ).to(device)

    def compute_depths(num_nodes: int, edges: List[Tuple[int, int]]) -> List[int]:
        """
        Compute DAG depths from roots (nodes with no incoming edges).
        If cycles exist (shouldn't), this is a best-effort breadth layering.
        """
        indeg = [0] * num_nodes
        children = [[] for _ in range(num_nodes)]
        for u, v in edges:
            children[u].append(v)
            indeg[v] += 1
        roots = [i for i in range(num_nodes) if indeg[i] == 0]
        # if no explicit root, pick all as roots (avoids empty)
        if not roots:
            roots = list(range(num_nodes))

        depth = [-1] * num_nodes
        from collections import deque
        q = deque()
        for r in roots:
            depth[r] = 0
            q.append(r)
        while q:
            u = q.popleft()
            for v in children[u]:
                if depth[v] < 0 or depth[v] > depth[u] + 1:
                    depth[v] = depth[u] + 1
                    q.append(v)
        # fill any isolated/unreached with 0
        for i in range(num_nodes):
            if depth[i] < 0:
                depth[i] = 0
        return depth

    def radial_initialize_embeddings(model, depths: List[int], eps_val: float, r_max: float = 0.9):
        with torch.no_grad():
            dmax = max(depths) if depths else 0
            # avoid divide-by-zero
            denom = float(dmax) if dmax > 0 else 1.0
            E = model.emb.data
            n, d = E.shape
            # random directions
            dirs = torch.randn_like(E)
            dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-15)
            # radii: r = eps + (r_max - eps) * depth/denom
            depths_t = torch.tensor(depths, dtype=E.dtype, device=E.device).view(-1, 1)
            radii = eps_val + (r_max - eps_val) * (depths_t / denom)
            model.emb.data = dirs * radii
            # safe projection
            model.emb.data = PoincareOps.proj_to_ball(model.emb.data,
                                                       max_norm=model.max_norm,
                                                       min_norm=model.eps)

    if len(edges_id) > 0:
        depths = compute_depths(len(codes), edges_id)
        radial_initialize_embeddings(model, depths, eps_val=eps, r_max=0.9)

    # We'll use manual Riemannian update, so only need to keep autograd for 'emb'
    # No standard optimizer is strictly necessary here.
    model.train()

    if len(edges_id) == 0:
        print("Warning: No parent-child edges found. Training without hierarchy constraints.")
        # Return untrained model if no edges
        return model, id_map

    # ========== DEBUG: Save initial embedding state ==========
    initial_emb = None
    if True:  # DEBUG flag - set to False to disable
        with torch.no_grad():
            initial_emb = model.emb.data.clone()
    # ========== END DEBUG ==========

    for ep in range(1, epochs + 1):
        hp, tp, hn, tn = build_batch(edges_id, len(codes), batch_size, neg_ratio)
        
        # Skip if batch is empty
        if len(hp) == 0:
            if ep % 20 == 0 or ep == 1:
                print(f"[epoch {ep:4d}] No edges available, skipping...")
            continue
            
        hp, tp, hn, tn = hp.to(device), tp.to(device), hn.to(device), tn.to(device)

        pos_energy = model(hp, tp)                # E(u,v)
        neg_energy = model(hn, tn)                # E(u',v')

        # ========== DEBUG: Calculate Xi and psi_u to understand energy values ==========
        if True:  # DEBUG flag - set to False to disable
            with torch.no_grad():
                u = model.emb[hp]
                v = model.emb[tp]
                Xi = PoincareOps.angle_Xi(u, v)
                psi_u = PoincareOps.psi(u, K=model.K, eps=model.eps)
        # ========== END DEBUG ==========

        # Max-margin loss: sum_pos E + sum_neg relu(gamma - E)
        loss = pos_energy.mean() + torch.relu(margin - neg_energy).mean()
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"Warning: NaN loss detected at epoch {ep}. Stopping training.")
            break

        loss.backward()
        
        # ========== DEBUG: Check gradient stats before update ==========
        if True:  # DEBUG flag - set to False to disable
            if model.emb.grad is not None:
                grad_max = model.emb.grad.abs().max().item()
                grad_mean = model.emb.grad.abs().mean().item()
                grad_norm = model.emb.grad.norm().item()
            else:
                grad_max = grad_mean = grad_norm = 0.0
        # ========== END DEBUG ==========
        
        model.riemannian_step(lr)

        if ep % 20 == 0 or ep == 1:
            with torch.no_grad():
                # simple monitoring: fraction of satisfied positives (E ~ 0) on current batch
                sat = (pos_energy < 1e-1).float().mean().item()
            print(f"[epoch {ep:4d}] loss={loss.item():.6f}  pos_satisfied(batch)={sat*100:.2f}%")
        
        # Evaluate on all data every 1000 epochs
        if ep % 1000 == 0 and len(edges_id) > 0:
            with torch.no_grad():
                model.eval()
                # Convert all edges to tensors
                all_hp = torch.tensor([h for h, _ in edges_id], dtype=torch.long).to(device)
                all_tp = torch.tensor([t for _, t in edges_id], dtype=torch.long).to(device)
                
                # Compute energy for all positive pairs (in batches to avoid memory issues)
                all_pos_energies = []
                eval_batch_size = 1024  # Use larger batch for evaluation
                
                for i in range(0, len(all_hp), eval_batch_size):
                    end_idx = min(i + eval_batch_size, len(all_hp))
                    batch_hp = all_hp[i:end_idx]
                    batch_tp = all_tp[i:end_idx]
                    batch_energy = model(batch_hp, batch_tp)
                    all_pos_energies.append(batch_energy)
                
                # Concatenate all energies
                all_pos_energy = torch.cat(all_pos_energies, dim=0)
                
                # Calculate pos_satisfied on all data
                all_sat = (all_pos_energy < 1e-1).float().mean().item()
                all_mean_energy = all_pos_energy.mean().item()
                all_min_energy = all_pos_energy.min().item()
                all_max_energy = all_pos_energy.max().item()
                
                print(f"\n{'='*60}")
                print(f"FULL DATA EVALUATION at epoch {ep}")
                print(f"{'='*60}")
                print(f"Total positive pairs: {len(edges_id)}")
                print(f"  pos_energy (all): min={all_min_energy:.6f}, max={all_max_energy:.6f}, mean={all_mean_energy:.6f}")
                print(f"  pos_satisfied (all): {all_sat*100:.2f}%")
                print(f"{'='*60}\n")
                
                model.train()
        
        if ep % 20 == 0 or ep == 1:
            # ========== DEBUG: Print detailed debug information ==========
            if True:  # DEBUG flag - set to False to disable
                print(f"\n{'='*60}")
                print(f"DEBUG INFO at epoch {ep}")
                print(f"{'='*60}")
                print(f"Batch size: {len(hp)} positive pairs, {len(hn)} negative pairs")
                print(f"\nEnergy values:")
                print(f"  pos_energy: min={pos_energy.min().item():.6f}, max={pos_energy.max().item():.6f}, mean={pos_energy.mean().item():.6f}")
                print(f"  neg_energy: min={neg_energy.min().item():.6f}, max={neg_energy.max().item():.6f}, mean={neg_energy.mean().item():.6f}")
                print(f"\nXi and psi_u breakdown:")
                print(f"  Xi (angle): min={Xi.min().item():.6f}, max={Xi.max().item():.6f}, mean={Xi.mean().item():.6f}")
                print(f"  psi_u (cone angle): min={psi_u.min().item():.6f}, max={psi_u.max().item():.6f}, mean={psi_u.mean().item():.6f}")
                print(f"  Xi - psi_u: min={(Xi-psi_u).min().item():.6f}, max={(Xi-psi_u).max().item():.6f}, mean={(Xi-psi_u).mean().item():.6f}")
                print(f"  Energy = relu(Xi - psi_u), should match pos_energy")
                print(f"\nEmbedding stats:")
                emb_norms = model.emb.data.norm(dim=1)
                print(f"  Embedding norms: min={emb_norms.min().item():.4f}, max={emb_norms.max().item():.4f}, mean={emb_norms.mean().item():.4f}")
                print(f"\nModel parameters:")
                print(f"  K={model.K:.6f}, eps={model.eps:.6f}, lr={lr:.6f}")
                print(f"\nGradient stats (before update):")
                print(f"  grad_norm={grad_norm:.8f}, grad_max={grad_max:.8f}, grad_mean={grad_mean:.8f}")
                
                # Check if embeddings are changing (using hyperbolic distance)
                if initial_emb is not None:
                    # Compute hyperbolic distance for each embedding vector
                    hyperbolic_distances = PoincareOps.poincare_distance(
                        model.emb.data, 
                        initial_emb,
                        eps=1e-15
                    )
                    emb_change = hyperbolic_distances.mean().item()
                    print(f"\nEmbedding update:")
                    print(f"  Average hyperbolic distance from initial: {emb_change:.8f}")
                
                print(f"{'='*60}\n")
            # ========== END DEBUG ==========

    return model, id_map


# =========================
# Utilities for loading ICD-10 codes and building hierarchy
# =========================

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


def extract_icd_parent_child_pair(code: str) -> List[Tuple[str, str]]:
    """
    Extract parent-child pairs for an ICD-10 code based on hierarchy.
    Supports both formats:
    - With dots: "I11.0" -> [("I11", "I11.0")]
                 "I11.01" -> [("I11", "I11.0"), ("I11.0", "I11.01")]
    - Without dots: "I110" -> [("I11", "I110")]
                   "I1101" -> [("I11", "I110"), ("I110", "I1101")]
    
    ICD-10 hierarchy: 
    - 3 chars: "I11" (category)
    - With dots: 5 chars "I11.0" (subcategory), 6+ chars "I11.01" (more specific)
    - Without dots: 4 chars "I110" (subcategory), 5+ chars "I1101" (more specific)
    
    This function generates ALL possible parent-child relationships based on the code structure,
    even if the parent codes are not in the original code list.
    
    Args:
        code: ICD-10 code string (may or may not contain dots)
        
    Returns:
        List of (parent, child) tuples
    """
    if not code or len(code) < 3:
        return []
    
    pairs = []
    
    # Case 1: Code contains dots (e.g., "I11.0", "I11.01")
    if '.' in code:
        parts = code.split('.')
        prefix = parts[0]  # e.g., "I11"
        
        # Level 1: 3-char prefix (e.g., "I11" -> "I11.0")
        if len(prefix) >= 3 and len(code) > len(prefix):
            pairs.append((prefix, code))
        
        # Level 2+: For codes with suffix after dot (e.g., "I11.01")
        if len(parts) >= 2 and len(parts[1]) > 0:
            suffix = parts[1]
            # Generate parents by shortening the suffix
            for i in range(1, len(suffix)):
                parent_suffix = suffix[:i]
                parent_code = f"{prefix}.{parent_suffix}"
                if parent_code != code:
                    pairs.append((parent_code, code))
    
    # Case 2: Code without dots (e.g., "I110", "I1101")
    else:
        # Build hierarchy by incrementally increasing length
        # Only generate DIRECT parent-child relationships (adjacent levels)
        # e.g., "I1101" should generate:
        #   - ("I110", "I1101") - direct parent
        # But we also need to track all intermediate levels for the hierarchy
        # The direct parent is always the code with one less character
        if len(code) > 3:
            # Find the direct parent (one character shorter)
            # For ICD-10 codes, the parent is typically the code minus the last character
            # But we need to be careful: "A0100" -> parent is "A010" (not "A010" -> "A01" directly for "A0100")
            parent = code[:-1]  # Remove last character to get direct parent
            if len(parent) >= 3:  # Ensure parent is at least 3 characters (ICD-10 minimum)
                pairs.append((parent, code))
    
    return pairs


def build_parent_child_edges_from_codes(codes: List[str]) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    Build parent-child edges for all ICD-10 codes based on their hierarchy.
    Includes parent codes even if they're not in the original code set.
    
    Args:
        codes: List of ICD codes
        
    Returns:
        Tuple of (edges, all_codes):
        - edges: List of (parent, child) tuples representing hierarchy relationships
        - all_codes: List of all codes including original codes and generated parent codes
    """
    pairs = []
    code_set = set(codes)
    all_codes_set = set(codes)  # Track all codes including parents
    
    # First pass: extract all parent-child pairs and collect all parent codes
    # We need to iteratively build the hierarchy because intermediate parents might be missing
    for code in codes:
        code_pairs = extract_icd_parent_child_pair(code)
        for parent, child in code_pairs:
            # Add both parent and child to all_codes_set
            all_codes_set.add(parent)
            all_codes_set.add(child)
            # Only keep pairs where child is in the original code set
            # This ensures we build hierarchy even if parents are not explicitly in the list
            if child in code_set:
                pairs.append((parent, child))
    
    # Second pass: recursively build hierarchy for intermediate parents
    # This ensures we have complete chains: A01 -> A010 -> A0100
    # We need to process all_codes_set recursively to build the full hierarchy
    changed = True
    while changed:
        changed = False
        current_all_codes = set(all_codes_set)
        for code in current_all_codes:
            if len(code) > 3 and '.' not in code:
                # Check if this code itself has a parent
                parent = code[:-1]
                if len(parent) >= 3:
                    if parent not in all_codes_set:
                        # Add the intermediate parent
                        all_codes_set.add(parent)
                        changed = True
                    # Add the parent-child relationship (parent -> code)
                    # We add this edge if code is in all_codes_set (either original or generated)
                    pairs.append((parent, code))
    
    # Remove duplicates
    final_pairs = []
    seen = set()
    for parent, child in pairs:
        pair = (parent, child)
        if pair not in seen:
            final_pairs.append(pair)
            seen.add(pair)
    
    # Return all codes (original + generated parents) sorted for consistent ordering
    all_codes_list = sorted(list(all_codes_set))
    
    return final_pairs, all_codes_list


# =========================
# Training and saving functions
# =========================

def train_and_save_cones(
    icd10_file_path: str,
    output_file: str = "hyperbolic_cones_embeddings.pkl",
    dim: int = 10,
    lr: float = 2e-3,
    epochs: int = 200,
    batch_size: int = 256,
    neg_ratio: int = 10,
    margin: float = 1e-2,
    eps: float = 0.1,
    K_scale: float = 0.9,
    seed: int = 42,
    device: str = "cpu"
):
    """
    Train hyperbolic entailment cones model and save to file.
    
    Args:
        icd10_file_path: Path to ICD-10 codes file
        output_file: Path to save the trained model
        dim: Embedding dimension
        lr: Learning rate
        epochs: Number of training epochs
        batch_size: Batch size for training
        neg_ratio: Negative sampling ratio
        margin: Margin for max-margin loss
        eps: Epsilon parameter for entailment cones
        K_scale: K scale parameter
        seed: Random seed
        device: Device to use for training
    """
    print(f"Loading ICD-10 codes from: {icd10_file_path}")
    codes = load_icd10_codes(icd10_file_path)
    print(f"Loaded {len(codes)} ICD-10 codes")
    
    # Show sample codes for debugging
    if len(codes) > 0:
        print(f"Sample codes (first 10): {codes[:10]}")
    
    print("Building parent-child edges from code hierarchy...")
    parent_child_edges, all_codes = build_parent_child_edges_from_codes(codes)
    print(f"Built {len(parent_child_edges)} parent-child edges")
    print(f"Total codes (including generated parents): {len(all_codes)} (original: {len(codes)})")
    
    if len(parent_child_edges) > 0:
        sample_edges = random.sample(parent_child_edges, min(5, len(parent_child_edges)))
        print(f"Sample edges (random 5): {sample_edges}")
    
    if len(parent_child_edges) == 0:
        print("Warning: No parent-child relationships found. Training will be skipped.")
        print("This usually means codes don't follow standard ICD-10 hierarchy structure.")
        print("Please check the code format in the input file.")
        # Use original codes if no edges found
        all_codes = codes
    
    print(f"Training hyperbolic entailment cones model...")
    print(f"  dim={dim}, epochs={epochs}, batch_size={batch_size}, lr={lr}")
    print(f"  neg_ratio={neg_ratio}, margin={margin}, eps={eps}, K_scale={K_scale}")
    
    model, id_map = train_hyperbolic_cones(
        codes=all_codes,  # Use all_codes including generated parents
        parent_child_edges=parent_child_edges,
        dim=dim,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        neg_ratio=neg_ratio,
        margin=margin,
        eps=eps,
        K_scale=K_scale,
        seed=seed,
        device=device
    )
    
    print(f"Training completed!")
    
    # Save model and id_map
    save_data = {
        'model': model,
        'id_map': id_map,
        'codes': all_codes,  # Save all codes including generated parents
        'original_codes': codes,  # Save original codes for reference
        'dim': dim,
        'eps': eps,
        'K': model.K,
        'num_codes': len(all_codes)
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"Saved model and mappings to: {output_file}")
    print(f"Model parameters: dim={dim}, num_codes={len(codes)}, K={model.K:.6f}")
    
    return model, id_map


def load_cones_model(model_file: str):
    """
    Load trained hyperbolic entailment cones model from file.
    
    Args:
        model_file: Path to the saved model file
        
    Returns:
        Dictionary containing model, id_map, and other metadata
    """
    with open(model_file, 'rb') as f:
        save_data = pickle.load(f)
    
    print(f"Loaded hyperbolic cones model from: {model_file}")
    print(f"  Dimension: {save_data['dim']}")
    print(f"  Number of codes: {save_data['num_codes']}")
    print(f"  Epsilon: {save_data['eps']}")
    print(f"  K: {save_data['K']:.6f}")
    
    return save_data


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train hyperbolic entailment cones for ICD-10 codes')
    
    # Data parameters
    parser.add_argument('--icd10_file', type=str, 
                       default="/data/yuyu/data/MIMIC_IV/icd10cm-codes-April-2024.txt",
                       help='Path to ICD-10 codes file')
    parser.add_argument('--output_file', type=str, default='hyperbolic_cones_embeddings.pkl',
                       help='Output file to save model (default: hyperbolic_cones_embeddings.pkl)')
    
    # Model parameters
    parser.add_argument('--dim', type=int, default=100,
                       help='Dimension of hyperbolic embeddings')
    parser.add_argument('--eps', type=float, default=0.15,
                       help='Epsilon parameter for entailment cones')
    parser.add_argument('--K_scale', type=float, default=0.95,
                       help='K scale parameter')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100000,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for training (default: 256)')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate for training')
    parser.add_argument('--neg_ratio', type=int, default=10,
                       help='Negative sampling ratio (default: 10)')
    parser.add_argument('--margin', type=float, default=2.0,
                       help='Margin for max-margin loss')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    return parser.parse_args()


# =========================
# Main execution
# =========================

if __name__ == "__main__":
    args = parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        args.device = 'cpu'
    
    train_and_save_cones(
        icd10_file_path=args.icd10_file,
        output_file=args.output_file,
        dim=args.dim,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        neg_ratio=args.neg_ratio,
        margin=args.margin,
        eps=args.eps,
        K_scale=args.K_scale,
        seed=args.seed,
        device=args.device
    )
    
    print("Hyperbolic entailment cones training completed!")
