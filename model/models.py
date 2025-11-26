import torch
import torch.nn as nn
import geoopt
import math
import pickle
import re
import numpy as np
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler

from hypercore.nn.linear import LorentzLinear
from hypercore.nn.conv import LResNet, LorentzRMSNorm, LorentzActivation
from hypercore.nn.attention import LorentzMultiheadAttention, LorentzEmbeddings
from hypercore.manifolds import Lorentz


class LorentzFeedForward(torch.nn.Module):
    """Feed-forward network in Lorentz space."""
    def __init__(self, manifold, d_model, d_ff):
        super().__init__()
        self.manifold = manifold
        self.linear1 = LorentzLinear(manifold, d_model - 1, d_ff - 1)
        self.linear2 = LorentzLinear(manifold, d_ff - 1, d_model - 1)
        self.activation = LorentzActivation(manifold, nn.ReLU())
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x
from geoopt import ManifoldParameter
import warnings
import math

class TransformerEncoder(nn.Module):
    def __init__(self, x_vocab_size, hidden=390, out_dim=None, *,
                 diag_size, proc_size,
                 num_heads=6, num_layers=3, p=0.3,
                 diag_itos=None, c=1.0, max_diag_len=None):
        super().__init__()
        self.diag_itos = diag_itos
        self.emb_diag  = nn.Embedding(diag_size + 1, hidden, padding_idx=0)
        # self.emb_proc  = nn.Embedding(proc_size + 1, embed_dim, padding_idx=0)
        # self.emb_third = nn.Embedding(x_vocab_size - (diag_size + proc_size) + 1, embed_dim, padding_idx=0)

        enc_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=num_heads, dim_feedforward=hidden * 4, dropout=p, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden))
        self.output_projection = nn.Linear(hidden, out_dim, bias=False)
        self.dropout = nn.Dropout(p)

    def encode(self, x_diag, x_proc, x_drug):
        device = x_diag.device
        B = x_diag.shape[0]
        
        e_diag = self.emb_diag(x_diag)   # (B, L_diag, E)
        # e_proc = self.emb_proc(x_proc)   # (B, L_proc, E)
        # e_third = self.emb_third(x_drug) # (B, L_drug, E)
        
        diag_mask = (x_diag != 0)   # (B, L_diag)
        # proc_mask = (x_proc != 0)   # (B, L_proc)
        # drug_mask = (x_drug != 0)   # (B, L_drug)
        # padding_mask = torch.cat([diag_mask, proc_mask, drug_mask], dim=1)  # (B, L_total)
    
        padding_mask_expanded = diag_mask.unsqueeze(-1)  # (B, L_total, 1)
        e_diag = e_diag.masked_fill(~padding_mask_expanded, 0.)

        cls_tokens = self.cls_token.expand(B, 1, -1)        # (B, 1, H)
        x_seq = torch.cat([cls_tokens, e_diag], dim=1) # (B, L_total+1, H)
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
        key_padding_mask = torch.cat([cls_mask, diag_mask], dim=1)
        src_key_padding_mask = ~key_padding_mask  # True = mask out

        x_seq = self.transformer(x_seq, src_key_padding_mask=src_key_padding_mask)
        x_cls = self.dropout(x_seq[:, 0, :])  # (B, H)
        return x_cls

    def forward(self, x_diag, x_proc, x_drug):
        x_cls = self.encode(x_diag, x_proc, x_drug)  # (B, H)
        logits = self.output_projection(x_cls)  # (B, out_dim)
        return logits

def precompute_theta_pos_frequencies(head_dim, seq_len, theta: float = 10000.0):
    head_dim -= 1
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)) # (Head_Dim / 2)
    m = torch.arange(seq_len)
    freqs = torch.outer(m, theta).float()
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

class _LTransformerDecoderBlock(torch.nn.Module):
    """
    A single Transformer block for the decoder.
    - Uses **masked** self-attention with padding mask.
    - Uses hyperbolic normalization and activation.
    """

    def __init__(self, manifold, d_model: int, n_head: int):
        super().__init__()
        dim_per_head = d_model // n_head
        self.manifold = manifold

        self.attn = LorentzMultiheadAttention(
            manifold, dim_per_head, dim_per_head, n_head,
            attention_type='full', trans_heads_concat=True
        )

        self.ln_1 = LorentzRMSNorm(manifold, d_model - 1)

        # MLP (Feed-forward network)
        self.mlp = LorentzFeedForward(manifold, d_model, d_model * 4)

        self.ln_2 = LorentzRMSNorm(manifold, d_model - 1)
        self.res1 = LResNet(manifold, use_scale=True, scale=4.0 * math.sqrt(d_model))
        self.res2 = LResNet(manifold, use_scale=True, scale=4.0 * math.sqrt(d_model))

    def forward(self, x, attn_mask=None, rope=None):
        lx = self.ln_1(x)
        ax = self.attn(lx, lx, output_attentions=False, mask=attn_mask, rot_pos=rope)  # Masked attention
        x = self.res1(x, ax)    
        x = self.res2(x, self.mlp(self.ln_2(x)))
        return x
    
class LTransformerDecoder(torch.nn.Module):
    """
    A decoder-only Transformer (like LLAMA) that:
    - Uses **causal + padding-based attention mask**:
      - causal mask: prevent attending to future positions
      - padding mask: mask out padding tokens (id == 0)
    - Outputs **logits** for next-token prediction.
    """

    def __init__(
        self,
        manifold_in = Lorentz(1.0),
        manifold_hidden = Lorentz(1.0),
        manifold_out = Lorentz(1.0),
        arch = "L3_W390_A6",
        vocab_size = None,
        context_length = None,
        out_dim = None,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.out_dim = out_dim
        # Effective context length including CLS token
        self.max_seq_len = context_length + 1
        # Parse architecture string
        self.layers = int(re.search(r"L(\d+)", arch).group(1))
        self.width = int(re.search(r"W(\d+)", arch).group(1))
        _attn = re.search(r"A(\d+)", arch)
        self.heads = int(_attn.group(1)) if _attn else self.width // 64
        # Token Embeddings (Lorentz)
        self.token_embed = LorentzEmbeddings(manifold_in, vocab_size, self.width, manifold_out=manifold_hidden, posit_embed=False, padding_idx=0)  # +1 for padding token
        self.cls_token = ManifoldParameter(
            manifold_hidden.random_normal((1, 1, self.width), std=0.02),
            manifold=manifold_hidden
        )

        # Transformer Blocks (Decoder Only)
        self.resblocks = torch.nn.ModuleList([
            _LTransformerDecoderBlock(manifold_hidden, self.width, self.heads)
            for _ in range(self.layers)
        ])

        # Final normalization and projection
        self.ln_final = LorentzRMSNorm(manifold_hidden, self.width - 1)
        self.final_proj = LorentzLinear(manifold_hidden, self.width - 1, self.width - 1, manifold_out=manifold_hidden)
        self.mapping = torch.nn.Linear(self.width, self.out_dim, bias=False)

        rope_vals = precompute_theta_pos_frequencies(
            self.width // self.heads,
            self.max_seq_len,
        )
        self.register_buffer("freqs_complex", rope_vals)

    def forward(self, x_diag, x_proc, x_drug, attn_mask = None):
        batch_size, max_len = x_diag.shape
        device = x_diag.device
        # CLS is never padding; only mask out padding token 0
        cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
        padding_mask_1d = (x_diag == 0)  # (B, max_len) - True where padding
        padding_mask_1d = torch.cat([cls_mask, padding_mask_1d], dim=1)  # (B, max_len+1)
        # Convert token indices to Lorentz manifold embeddings
        # This projects discrete token IDs into continuous hyperbolic space representations
        # Shape: (batch_size, context_length, width) where width includes time+space dimensions
        cls = self.cls_token.expand(batch_size, -1, -1)
        token_embeddings = self.token_embed(x_diag)
        token_embeddings = torch.cat([cls, token_embeddings], dim=1)
        # RoPE frequencies need to match full sequence length (including CLS)
        seq_len = token_embeddings.shape[1]
        freqs_cis = self.freqs_complex[:seq_len]

        # Build causal mask (L+1, L+1): True where positions are NOT allowed to attend (future)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1
        )  # (L+1, L+1)
        padding_mask_2d = padding_mask_1d.unsqueeze(1).expand(-1, seq_len, -1)  # (B, L+1, L+1)
        # attn_mask = causal_mask.unsqueeze(0) | padding_mask_2d  # (B, L+1, L+1)
        attn_mask = padding_mask_2d
        _attn_mask = attn_mask
        
        # Forward pass through all transformer decoder blocks
        # Each block applies: Lorentz normalization -> bidirectional self-attention -> residual connection
        #                    -> Lorentz normalization -> feed-forward network -> residual connection
        for block in self.resblocks:
            token_embeddings = block(token_embeddings, _attn_mask, freqs_cis)

        token_embeddings = self.final_proj(token_embeddings)
        token_embeddings = self.ln_final(token_embeddings)
        cls_state = token_embeddings[:, 0, :]
        logits = self.mapping(cls_state)
        
        # Return logits for multi-label classification
        # Shape: (batch_size, out_dim)
        return logits

class TransformerDecoder(torch.nn.Module):
    """
    A standard decoder-only Transformer in Euclidean space.
    - Same high-level interface as `LTransformerDecoder`.
    - Uses PyTorch's `nn.TransformerEncoderLayer` with a padding mask (no causal masking).
    """

    def __init__(
        self,
        manifold_in=None,
        manifold_hidden=None,
        manifold_out=None,
        arch: str = "L3_W390_A6",
        vocab_size: int = None,
        context_length: int = None,
        out_dim: int = None,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.out_dim = out_dim
        # Effective context length including CLS token
        self.max_seq_len = context_length + 1

        # Parse architecture string (same as LTransformerDecoder)
        self.layers = int(re.search(r"L(\d+)", arch).group(1))
        self.width = int(re.search(r"W(\d+)", arch).group(1))
        _attn = re.search(r"A(\d+)", arch)
        self.heads = int(_attn.group(1)) if _attn else self.width // 64

        # Token embeddings in Euclidean space
        self.token_embed = nn.Embedding(vocab_size, self.width, padding_idx=0)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.width) * 0.02)

        # Standard Transformer blocks (encoder layers; no causal mask, only padding mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.width,
            nhead=self.heads,
            dim_feedforward=self.width * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.resblocks = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.layers,
        )

        self.ln_final = nn.LayerNorm(self.width)
        self.mapping = nn.Linear(self.width, self.out_dim, bias=False)

    def forward(self, x_diag, x_proc, x_drug, attn_mask=None):
        """
        x_* shapes: (batch_size, seq_len)
        Only `x_diag` is used, to mirror `LTransformerDecoder` behaviour.
        """
        batch_size, max_len = x_diag.shape

        # Build padding mask including CLS (CLS is never padding)
        cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=x_diag.device)
        padding_mask_1d = (x_diag == 0)  # True where padding
        padding_mask_1d = torch.cat([cls_mask, padding_mask_1d], dim=1)  # (B, L+1)

        # Embeddings + prepend CLS
        token_embeddings = self.token_embed(x_diag)  # (B, L, D)
        cls = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, D)
        x = torch.cat([cls, token_embeddings], dim=1)  # (B, L+1, D)

        # Pass through Transformer encoder stack with padding masks only
        x = self.resblocks(
            x,
            src_key_padding_mask=padding_mask_1d,
        )

        x = self.ln_final(x)
        cls_state = x[:, 0, :]
        logits = self.mapping(cls_state)
        return logits


def create_model(model_type, x_vocab_size, out_dim, **kwargs):
    if model_type == 'transformer_encoder':
        return TransformerEncoder(x_vocab_size=x_vocab_size, out_dim=out_dim, **kwargs)
    elif model_type == 'ltransformer_decoder':
        return LTransformerDecoder(
            vocab_size=x_vocab_size,
            context_length=kwargs.get('max_diag_len'),
            out_dim=out_dim  # Output vocabulary size for final mapping
        )
    elif model_type == 'transformer_decoder':
        return TransformerDecoder(
            vocab_size=x_vocab_size,
            context_length=kwargs.get('max_diag_len'),
            out_dim=out_dim
        )