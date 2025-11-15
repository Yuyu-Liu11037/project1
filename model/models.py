"""
Model definition module
Contains MLP and Transformer model definitions
"""
import torch
import torch.nn as nn
import geoopt
import math
from model.hierarchical_embedding import HierarchicalHyperbolicEmbedding


def project_to_ball(x, c, eps=1e-5):
    """Project points to Poincaré ball."""
    r = 1.0 / (c**0.5)
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    max_norm = (1 - eps) * r
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale


def artanh(x):  # 数值稳定的 atanh
    """Numerically stable arctanh."""
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def logmap0_poincare(x, c):
    """Logarithmic map from Poincaré ball to tangent space at origin."""
    x = project_to_ball(x, c)
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    return (2.0 / (c**0.5)) * artanh((c**0.5) * norm) * (x / norm)


class PoincareToEuclid(nn.Module):
    """Project from Poincaré ball to Euclidean space with learnable curvature."""
    def __init__(self, in_dim, out_dim, c_init=1.0):
        super().__init__()
        self.logit_c = nn.Parameter(torch.tensor(float(c_init)).log())  # learnable curvature
        self.norm = nn.LayerNorm(out_dim)
        self.linear = nn.Linear(in_dim, out_dim)  # Linear projection to hidden dimension

    @property
    def c(self):
        return self.logit_c.exp()

    def forward(self, x_h):  # x_h in Poincaré ball
        x_tan = logmap0_poincare(x_h, self.c)
        x_proj = self.linear(x_tan)
        return self.norm(x_proj)


class HyperbolicEmbedding(nn.Module):
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

    def forward(self, idx):
        # idx: (...,) 其中0表示padding/不属于该段
        x_h = self.weight[idx]
        x_e = self.ball.logmap0(x_h)
        if self.padding_idx is not None:
            mask = (idx == self.padding_idx).unsqueeze(-1)
            x_e = x_e.masked_fill(mask, 0.)
        return x_e
    
    def get_hyperbolic_embeddings(self, idx):
        """Get embeddings in hyperbolic space (Poincaré ball)."""
        x_h = self.weight[idx]
        if self.padding_idx is not None:
            mask = (idx == self.padding_idx).unsqueeze(-1)
            x_h = x_h.masked_fill(mask, 0.)
        return x_h

    @torch.no_grad()
    def reproject_(self):
        self.weight.data = self.ball.projx(self.weight.data)


class TransformerModel(nn.Module):
    def __init__(self, x_vocab_size, hidden, out_dim, *,
                 diag_size, proc_size, embed_dim=256,
                 num_heads=8, num_layers=3, p=0.3,
                 c_diag=1.0, c_proc=1.0, c_third=1.0,
                 max_diag_len=None, max_proc_len=None, max_drug_len=None,
                 use_hierarchical_structure=False,
                 ancestors_dict=None, diag_itos=None,
                 hierarchical_mode="weighted_sum", max_depth=None):
        super().__init__()
        V = x_vocab_size
        D = diag_size
        P = proc_size
        T = V - (D + P)  # 第三段大小

        # Conditional hierarchical embedding for diagnosis codes
        self.use_hierarchical_structure = use_hierarchical_structure
        if use_hierarchical_structure and ancestors_dict is not None and diag_itos is not None:
            # Build itos mapping: index 0 = padding, index 1+ = codes from diag_itos
            # diag_itos is a list where diag_itos[i] is the code string at vocab index i
            # In the model, index 0 is padding, index 1 maps to diag_itos[0], etc.
            itos = {0: "<pad>"}
            for i in range(len(diag_itos)):
                itos[i + 1] = diag_itos[i]
            self.emb_diag = HierarchicalHyperbolicEmbedding(
                D + 1, embed_dim, ancestors_dict=ancestors_dict,
                itos=itos, c=c_diag, padding_idx=0,
                mode=hierarchical_mode, max_depth=max_depth
            )
        else:
            # Standard hyperbolic embedding
            self.emb_diag = HyperbolicEmbedding(D + 1, embed_dim, c=c_diag, padding_idx=0)
        
        # Procedure and drug codes remain standard
        self.emb_proc  = HyperbolicEmbedding(P + 1, embed_dim, c=c_proc,  padding_idx=0)
        self.emb_third = HyperbolicEmbedding(T + 1, embed_dim, c=c_third, padding_idx=0)

        # Project from Poincaré ball to Euclidean space with learnable curvature
        # Use average curvature of the three embedding types as initial value
        c_avg = (c_diag + c_proc + c_third) / 3.0
        self.input_projection = PoincareToEuclid(embed_dim, hidden, c_init=c_avg)
        # self.pos_encoding = PositionalEncoding(hidden, p)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=num_heads,
            dim_feedforward=hidden * 4, dropout=p, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden))
        self.output_projection = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(p)

        # 记录边界方便 forward
        self.D = D
        self.P = P
        self.V = V
        self.max_diag_len = max_diag_len
        self.max_proc_len = max_proc_len
        self.max_drug_len = max_drug_len

    def forward(self, x_diag, x_proc, x_drug):
        """
        x_diag: (B, L_diag), diagnosis codes, 0=padding
        x_proc: (B, L_proc), procedure codes, 0=padding
        x_drug: (B, L_drug), drug codes, 0=padding
        Each code type uses local indexing (0-indexed in its own vocabulary)
        """
        device = x_diag.device
        B = x_diag.shape[0]
        
        e_diag_h = self.emb_diag.get_hyperbolic_embeddings(x_diag)   # (B, L_diag, E)
        e_proc_h = self.emb_proc.get_hyperbolic_embeddings(x_proc)   # (B, L_proc, E)
        e_third_h = self.emb_third.get_hyperbolic_embeddings(x_drug) # (B, L_drug, E)

        # Concatenate the three sequences along the sequence dimension
        x_embedded_h = torch.cat([e_diag_h, e_proc_h, e_third_h], dim=1)  # (B, L_total, E)
        
        # Get padding masks for each type
        diag_mask = (x_diag != 0)   # (B, L_diag)
        proc_mask = (x_proc != 0)   # (B, L_proc)
        drug_mask = (x_drug != 0)   # (B, L_drug)
        
        # Concatenate masks
        padding_mask = torch.cat([diag_mask, proc_mask, drug_mask], dim=1)  # (B, L_total)

        # Project from Poincaré ball to Euclidean space and then to hidden dimension
        x_projected = self.input_projection(x_embedded_h)  # (B, L_total, H)
        
        # Apply padding mask (set padding positions to zero)
        padding_mask_expanded = padding_mask.unsqueeze(-1)  # (B, L_total, 1)
        x_projected = x_projected.masked_fill(~padding_mask_expanded, 0.)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, 1, -1)
        x_seq = torch.cat([cls_tokens, x_projected], dim=1)  # (B, L_total+1, H)

        # Extend padding mask for CLS token (always valid)
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
        key_padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
        src_key_padding_mask = ~key_padding_mask  # True = mask out

        # x_seq = self.pos_encoding(x_seq)
        x_seq = self.transformer(x_seq, src_key_padding_mask=src_key_padding_mask)

        x_cls = self.dropout(x_seq[:, 0, :])
        return self.output_projection(x_cls)

    @torch.no_grad()
    def reproject_hyperbolic_(self):
        # Handle both standard and hierarchical embeddings
        if hasattr(self.emb_diag, 'reproject_'):
            self.emb_diag.reproject_()
        self.emb_proc.reproject_()
        self.emb_third.reproject_()


def create_model(model_type, x_vocab_size, hidden, out_dim, **kwargs):
    """Model factory function"""
    return TransformerModel(x_vocab_size, hidden, out_dim, **kwargs)

