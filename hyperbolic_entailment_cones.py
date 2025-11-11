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
        self.K_scale = float(K_scale)  # Store K_scale for annealing
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

    def update_K_scale(self, new_K_scale: float):
        """
        Update K_scale and recalculate K.
        
        Args:
            new_K_scale: New K_scale value (should be in [0, 1])
        """
        self.K_scale = float(new_K_scale)
        self.K = self.K_scale * self.K_max


# =========================
# Training helper
# =========================

def get_learning_rate(
    epoch: int, 
    initial_lr: float, 
    T_max: float, 
    lr_min: float = 1e-6, 
    lr_warmup_epochs: int = 0,
    pos_satisfied: float = None,  # 新增参数
    adaptive_lr: bool = False,   # 是否启用自适应
    sat_threshold_high: float = 0.95,  # 高满足度阈值
    sat_threshold_low: float = 0.5,    # 低满足度阈值
    lr_scale_high: float = 0.5,        # 高满足度时的学习率缩放
    lr_scale_low: float = 1.2          # 低满足度时的学习率缩放（上限为 initial_lr）
) -> float:
    """
    Calculate learning rate with cosine annealing decay schedule.
    Optionally adapt based on pos_satisfied.
    
    Args:
        epoch: Current epoch (1-indexed)
        initial_lr: Initial learning rate
        T_max: Maximum number of epochs for cosine decay (period)
        lr_min: Minimum learning rate
        lr_warmup_epochs: Number of warmup epochs (linear warmup)
        pos_satisfied: Current pos_satisfied value (0-1)
        adaptive_lr: Whether to use adaptive learning rate based on pos_satisfied
        sat_threshold_high: High satisfaction threshold
        sat_threshold_low: Low satisfaction threshold
        lr_scale_high: Learning rate scale when satisfaction is high
        lr_scale_low: Learning rate scale when satisfaction is low
    
    Returns:
        Current learning rate
    """
    # Warmup phase
    if epoch <= lr_warmup_epochs and lr_warmup_epochs > 0:
        base_lr = initial_lr * (epoch / lr_warmup_epochs)
    else:
        # Adjust epoch for decay calculation (after warmup)
        effective_epoch = epoch - lr_warmup_epochs
        
        # Cosine annealing: lr = lr_min + (initial_lr - lr_min) * (1 + cos(π * epoch / T_max)) / 2
        if effective_epoch >= T_max:
            base_lr = lr_min
        else:
            base_lr = lr_min + (initial_lr - lr_min) * (1 + math.cos(math.pi * effective_epoch / T_max)) / 2
    
    # Adaptive adjustment based on pos_satisfied
    if adaptive_lr and pos_satisfied is not None:
        if pos_satisfied >= sat_threshold_high:
            # High satisfaction: reduce learning rate for fine-tuning
            base_lr = base_lr * lr_scale_high
        elif pos_satisfied < sat_threshold_low:
            # Low satisfaction: increase learning rate (but cap at initial_lr)
            base_lr = min(base_lr * lr_scale_low, initial_lr)
        # Medium satisfaction: keep base_lr
    
    return max(base_lr, lr_min)

def get_K_scale_annealed(
    epoch: int,
    K_scale_start: float = 0.99,
    K_scale_end: float = 0.9,
    T_max: float = 200,
    warmup_epochs: int = 0,
    annealing_type: str = "linear"
) -> float:
    """
    Calculate K_scale with annealing schedule.
    
    Args:
        epoch: Current epoch (1-indexed)
        K_scale_start: Initial K_scale value (default: 0.99)
        K_scale_end: Final K_scale value (default: 0.9)
        T_max: Maximum number of epochs for annealing (period)
        warmup_epochs: Number of warmup epochs (keep K_scale_start)
        annealing_type: Type of annealing - "linear" or "cosine"
    
    Returns:
        Current K_scale value
    """
    # Warmup phase: keep initial value
    if epoch <= warmup_epochs and warmup_epochs > 0:
        return K_scale_start
    
    # Adjust epoch for annealing calculation (after warmup)
    effective_epoch = epoch - warmup_epochs
    
    if effective_epoch >= T_max:
        # After annealing period, use final value
        return K_scale_end
    
    if annealing_type == "cosine":
        # Cosine annealing: smooth transition
        progress = effective_epoch / T_max
        K_scale = K_scale_end + (K_scale_start - K_scale_end) * (1 + math.cos(math.pi * progress)) / 2
    else:
        # Linear annealing: default
        progress = effective_epoch / T_max
        K_scale = K_scale_start + (K_scale_end - K_scale_start) * progress
    
    return K_scale

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
    return hp, tp, hn, tn  #positive heads, positive tails, negative heads, negative tails


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
    device: str = "cpu",
    T_max: float = 200,
    lr_min: float = 1e-6,
    lr_warmup_epochs: int = 0,
    # 新增参数：动态调整相关
    adaptive_lr: bool = True,              # 是否启用自适应学习率
    early_stopping: bool = True,          # 是否启用早停
    early_stop_patience: int = 50,        # 早停耐心值（epochs）
    early_stop_threshold: float = 0.98,   # 早停阈值（pos_satisfied）
    early_stop_min_epochs: int = 100,     # 最少训练轮数
    sat_history_window: int = 10,          # 用于计算平均满足度的窗口大小
    # K_scale 退火相关参数
    K_scale_annealing: bool = True,       # 是否启用 K_scale 退火
    K_scale_start: float = 0.99,          # K_scale 初始值
    K_scale_end: float = 0.9,             # K_scale 最终值
    K_scale_T_max: float = None,          # K_scale 退火周期（None 则使用 T_max）
    K_scale_warmup_epochs: int = 0,       # K_scale 预热轮数
    K_scale_annealing_type: str = "linear",  # 退火类型："linear" 或 "cosine"
    verbose: bool = True
):
    random.seed(seed)
    torch.manual_seed(seed)

    id_map = make_id_map(codes)   # {code : id}
    edges_id = [(id_map[p], id_map[c]) for (p, c) in parent_child_edges]

    # 如果启用 K_scale 退火，使用初始值；否则使用指定的 K_scale
    initial_K_scale = K_scale_start if K_scale_annealing else K_scale
    
    model = HyperbolicEntailmentCones(
        num_codes=len(codes),
        dim=dim,
        eps=eps,
        K_scale=initial_K_scale,
        seed=seed
    ).to(device)
    
    # 设置 K_scale 退火参数
    if K_scale_annealing:
        K_scale_T_max_val = K_scale_T_max if K_scale_T_max is not None else T_max
    else:
        K_scale_T_max_val = T_max

    depths = compute_depths(len(codes), edges_id)
    radial_initialize_embeddings(model, depths, eps_val=eps, r_max=0.9)
    model.train()

    # ========== DEBUG: Save initial embedding state ==========
    initial_emb = None
    if True:  # DEBUG flag - set to False to disable
        with torch.no_grad():
            initial_emb = model.emb.data.clone()
    # ========== END DEBUG ==========

    best_sat = 0.0
    best_epoch = 0
    sat_history = [] 
    patience_counter = 0
    best_model_state = None

    for ep in range(1, epochs + 1):
        hp, tp, hn, tn = build_batch(edges_id, len(codes), batch_size, neg_ratio)
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
        
        # 计算当前 batch 的 pos_satisfied 和 neg_unsatisfied
        with torch.no_grad():
            sat = (pos_energy < 1e-1).float().mean().item()
            sat_history.append(sat)
            # 保持历史窗口大小
            if len(sat_history) > sat_history_window:
                sat_history.pop(0)
            avg_sat = sum(sat_history) / len(sat_history) if sat_history else sat  # 平均满足度
            
            # neg_unsatisfied: 负样本能量 >= margin 的比例（被正确拒绝）
            neg_unsat = (neg_energy >= margin).float().mean().item()
        
        # 动态调整学习率
        if adaptive_lr:
            current_lr = get_learning_rate(
                epoch=ep,
                initial_lr=lr,
                T_max=T_max,
                lr_min=lr_min,
                lr_warmup_epochs=lr_warmup_epochs,
                pos_satisfied=avg_sat,  # 使用平均满足度
                adaptive_lr=True,
                sat_threshold_high=0.95,
                sat_threshold_low=0.5,
                lr_scale_high=0.5,
                lr_scale_low=1.2
            )
        else:
            current_lr = get_learning_rate(
                epoch=ep,
                initial_lr=lr,
                T_max=T_max,
                lr_min=lr_min,
                lr_warmup_epochs=lr_warmup_epochs
            )
        
        # K_scale 退火
        if K_scale_annealing:
            current_K_scale = get_K_scale_annealed(
                epoch=ep,
                K_scale_start=K_scale_start,
                K_scale_end=K_scale_end,
                T_max=K_scale_T_max_val,
                warmup_epochs=K_scale_warmup_epochs,
                annealing_type=K_scale_annealing_type
            )
            model.update_K_scale(current_K_scale)
        else:
            current_K_scale = model.K_scale
        
        model.riemannian_step(current_lr)

        # 早停检查
        if early_stopping and ep >= early_stop_min_epochs:
            if avg_sat >= early_stop_threshold:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    if verbose:
                        print(f"\n{'='*60}")
                        print(f"Early stopping triggered at epoch {ep}!")
                        print(f"  Average pos_satisfied: {avg_sat*100:.2f}% >= {early_stop_threshold*100:.2f}%")
                        print(f"  Maintained for {patience_counter} epochs")
                        print(f"{'='*60}\n")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break
            else:
                patience_counter = 0
            
            # 保存最佳模型
            if avg_sat > best_sat:
                best_sat = avg_sat
                best_epoch = ep
                best_model_state = model.state_dict().copy()

        # 打印信息
        if ep % 20 == 0 or ep == 1:
            if verbose:
                print(f"[epoch {ep:4d}] loss={loss.item():.6f}  "
                      f"pos_satisfied(batch)={sat*100:.2f}%  "
                      f"pos_satisfied(avg)={avg_sat*100:.2f}%  "
                      f"neg_unsatisfied={neg_unsat*100:.2f}%  "
                      f"lr={current_lr:.6f}  "
                      f"K_scale={current_K_scale:.4f}")
                if early_stopping and ep >= early_stop_min_epochs:
                    print(f"  Early stop: patience={patience_counter}/{early_stop_patience}, "
                          f"best_sat={best_sat*100:.2f}% @ epoch {best_epoch}")
                print(f"\nGradient stats (before update):")
                print(f"  grad_norm={grad_norm:.8f}, grad_max={grad_max:.8f}, grad_mean={grad_mean:.8f}")
                if initial_emb is not None:
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

def load_icd10_codes(icd10_file_path: str) -> List[str]:
    all_conditions = []
    with open(icd10_file_path, 'r') as f:
        for line in f:
            code = line.split()[0]
            all_conditions.append(code)
    return all_conditions


def extract_icd_parent_child_pair(code: str) -> List[Tuple[str, str]]:
    pairs = []
    
    if len(code) > 3:
        parent = code[:-1]  # Remove last character to get direct parent
        if len(parent) >= 3:  # Ensure parent is at least 3 characters (ICD-10 minimum)
            pairs.append((parent, code))
    
    return pairs


def build_parent_child_edges_from_codes(codes: List[str]) -> Tuple[List[Tuple[str, str]], List[str]]:
    pairs = []
    all_codes = set(codes)
    for code in codes:
        code_pairs = extract_icd_parent_child_pair(code)
        for parent, child in code_pairs:
            pairs.append((parent, child))
            all_codes.add(parent)
            all_codes.add(child)
    pairs = list(set(pairs))
    return pairs, list(all_codes)


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
    device: str = "cpu",
    T_max: float = 200,
    lr_min: float = 1e-6,
    lr_warmup_epochs: int = 0,
    # 新增参数：动态调整相关
    adaptive_lr: bool = True,
    early_stopping: bool = True,
    early_stop_patience: int = 50,
    early_stop_threshold: float = 0.98,
    early_stop_min_epochs: int = 100,
    sat_history_window: int = 10,
    # K_scale 退火相关参数
    K_scale_annealing: bool = True,
    K_scale_start: float = 0.99,
    K_scale_end: float = 0.9,
    K_scale_T_max: float = None,
    K_scale_warmup_epochs: int = 0,
    K_scale_annealing_type: str = "linear"
):
    codes = load_icd10_codes(icd10_file_path)
    
    print("Building parent-child edges from code hierarchy...")
    parent_child_edges, all_codes = build_parent_child_edges_from_codes(codes)
    print(f"Built {len(parent_child_edges)} parent-child edges")
    
    if len(parent_child_edges) > 0:
        sample_edges = random.sample(parent_child_edges, min(5, len(parent_child_edges)))
        print(f"Sample edges (random 5): {sample_edges}")
  
    print(f"Training hyperbolic entailment cones model...")
    print(f"  dim={dim}, epochs={epochs}, batch_size={batch_size}, lr={lr}")
    if K_scale_annealing:
        print(f"  neg_ratio={neg_ratio}, margin={margin}, eps={eps}")
        print(f"  K_scale: annealing from {K_scale_start} to {K_scale_end} (type={K_scale_annealing_type})")
    else:
        print(f"  neg_ratio={neg_ratio}, margin={margin}, eps={eps}, K_scale={K_scale}")
    print(f"  lr_decay: cosine, T_max={T_max}, lr_min={lr_min}, warmup={lr_warmup_epochs}")
    print(f"  adaptive_lr={adaptive_lr}, early_stopping={early_stopping}")
    if early_stopping:
        print(f"    early_stop_threshold={early_stop_threshold}, patience={early_stop_patience}, min_epochs={early_stop_min_epochs}")
    
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
        device=device,
        T_max=T_max,
        lr_min=lr_min,
        lr_warmup_epochs=lr_warmup_epochs,
        adaptive_lr=adaptive_lr,
        early_stopping=early_stopping,
        early_stop_patience=early_stop_patience,
        early_stop_threshold=early_stop_threshold,
        early_stop_min_epochs=early_stop_min_epochs,
        sat_history_window=sat_history_window,
        K_scale_annealing=K_scale_annealing,
        K_scale_start=K_scale_start,
        K_scale_end=K_scale_end,
        K_scale_T_max=K_scale_T_max,
        K_scale_warmup_epochs=K_scale_warmup_epochs,
        K_scale_annealing_type=K_scale_annealing_type
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


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train hyperbolic entailment cones for ICD-10 codes')
    
    # Data parameters
    parser.add_argument('--icd10_file', type=str, 
                       default="/data/yuyu/project1/cond_hist_codes.txt",
                       help='Path to ICD-10 codes file')
    parser.add_argument('--output_file', type=str, default='hyperbolic_cones_embeddings.pkl',
                       help='Output file to save model')
    
    # Model parameters
    parser.add_argument('--dim', type=int, default=128,
                       help='Dimension of hyperbolic embeddings')
    parser.add_argument('--eps', type=float, default=0.15,
                       help='Epsilon parameter for entailment cones')
    parser.add_argument('--K_scale', type=float, default=0.99,
                       help='K scale parameter')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20000,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=10,
                       help='Learning rate for training')
    parser.add_argument('--neg_ratio', type=int, default=30,
                       help='Negative sampling ratio')
    parser.add_argument('--margin', type=float, default=0.15,
                       help='Margin for max-margin loss')
    
    # Learning rate decay parameters (cosine annealing)
    parser.add_argument('--T_max', type=float, default=20000,
                       help='Maximum number of epochs for cosine decay (period)')
    parser.add_argument('--lr_min', type=float, default=1.0,
                       help='Minimum learning rate')
    parser.add_argument('--lr_warmup_epochs', type=int, default=0,
                       help='Number of warmup epochs with linear warmup')
    
    # Dynamic adjustment parameters
    parser.add_argument('--adaptive_lr', action='store_true', default=True,
                       help='Enable adaptive learning rate based on pos_satisfied')
    parser.add_argument('--no_adaptive_lr', dest='adaptive_lr', action='store_false',
                       help='Disable adaptive learning rate')
    parser.add_argument('--early_stopping', action='store_true', default=True,
                       help='Enable early stopping based on pos_satisfied')
    parser.add_argument('--no_early_stopping', dest='early_stopping', action='store_false',
                       help='Disable early stopping')
    parser.add_argument('--early_stop_patience', type=int, default=50,
                       help='Number of epochs to wait before early stopping')
    parser.add_argument('--early_stop_threshold', type=float, default=0.98,
                       help='pos_satisfied threshold for early stopping (0-1)')
    parser.add_argument('--early_stop_min_epochs', type=int, default=100,
                       help='Minimum number of epochs before early stopping can trigger')
    parser.add_argument('--sat_history_window', type=int, default=10,
                       help='Window size for computing average pos_satisfied')
    
    # K_scale annealing parameters
    parser.add_argument('--K_scale_annealing', action='store_true',
                       help='Enable K_scale annealing')
    parser.add_argument('--no_K_scale_annealing', dest='K_scale_annealing', action='store_false',
                       help='Disable K_scale annealing')
    parser.add_argument('--K_scale_start', type=float, default=0.995,
                       help='Initial K_scale value for annealing')
    parser.add_argument('--K_scale_end', type=float, default=0.9,
                       help='Final K_scale value for annealing')
    parser.add_argument('--K_scale_T_max', type=float, default=None,
                       help='Maximum epochs for K_scale annealing (None uses T_max)')
    parser.add_argument('--K_scale_warmup_epochs', type=int, default=0,
                       help='Number of warmup epochs for K_scale annealing')
    parser.add_argument('--K_scale_annealing_type', type=str, default='linear',
                       choices=['linear', 'cosine'],
                       help='Type of K_scale annealing: linear or cosine')
    
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
        device=args.device,
        T_max=args.T_max,
        lr_min=args.lr_min,
        lr_warmup_epochs=args.lr_warmup_epochs,
        adaptive_lr=args.adaptive_lr,
        early_stopping=args.early_stopping,
        early_stop_patience=args.early_stop_patience,
        early_stop_threshold=args.early_stop_threshold,
        early_stop_min_epochs=args.early_stop_min_epochs,
        sat_history_window=args.sat_history_window,
        K_scale_annealing=args.K_scale_annealing,
        K_scale_start=args.K_scale_start,
        K_scale_end=args.K_scale_end,
        K_scale_T_max=args.K_scale_T_max,
        K_scale_warmup_epochs=args.K_scale_warmup_epochs,
        K_scale_annealing_type=args.K_scale_annealing_type
    )
    
    print("Hyperbolic entailment cones training completed!")
