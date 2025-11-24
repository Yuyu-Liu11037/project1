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


class MLP2D(nn.Module):
    def __init__(self, diag_vocab_size, hidden=32, num_classes=None, diag_itos=None):
        super().__init__()
        self.diag_itos = diag_itos
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )
        with open('sarkar_embeddings.pkl', 'rb') as f:
            self.sarkar_embeddings = pickle.load(f)
            
    def forward(self, x_diag, x_proc, x_drug):
        """
        x_diag: (B, L_diag) - batch of diagnosis code sequences
        Returns: (B, num_classes) - logits for each class
        """
        device = x_diag.device
        B = x_diag.shape[0]
        L_diag = x_diag.shape[1]
        
        # Convert each code to its 2D Sarkar embedding
        e_diag_list = []
        for batch_item in x_diag:  # Iterate over batch
            batch_embeddings = []
            for x in batch_item:  # Iterate over sequence
                x_orig = x.item()
                if x_orig == 0:  # padding token
                    batch_embeddings.append(np.zeros(2, dtype=np.float32))
                else:
                    x_val = x_orig - 1  # Convert to 0-based index for diag_itos
                    code = self.diag_itos[x_val]
                    if code in self.sarkar_embeddings:
                        batch_embeddings.append(np.array(self.sarkar_embeddings[code], dtype=np.float32))
                    else:
                        batch_embeddings.append(np.zeros(2, dtype=np.float32))
            e_diag_list.append(batch_embeddings)
        
        # Convert to tensor: shape (B, L_diag, 2)
        e_diag_sarkar = torch.tensor(e_diag_list, device=device, dtype=torch.float32)
        
        # Aggregate sequence dimension: (B, L_diag, 2) -> (B, 2)
        # Use mean pooling over non-padding tokens
        diag_mask = (x_diag != 0).unsqueeze(-1).float()  # (B, L_diag, 1)
        masked_embeddings = e_diag_sarkar * diag_mask  # (B, L_diag, 2)
        seq_lengths = diag_mask.sum(dim=1).clamp(min=1.0)  # (B, 1) - avoid division by zero
        e_diag_aggregated = masked_embeddings.sum(dim=1) / seq_lengths  # (B, 2)
        
        # Pass through MLP: (B, 2) -> (B, num_classes)
        return self.net(e_diag_aggregated)


class MLPDiagLearned(nn.Module):
    def __init__(self, diag_vocab_size, emb_dim=16, hidden=32, num_classes=None, diag_itos=None):
        """
        diag_vocab_size: 诊断码的词表大小（不含 padding，通常 = len(diag_itos)）
        emb_dim        : 可学习 embedding 维度
        hidden         : MLP 的隐藏层维度
        num_classes    : 输出类别数
        """
        super().__init__()
        # 0 作为 padding，所以 vocab 是 diag_vocab_size + 1
        self.emb_diag = nn.Embedding(
            num_embeddings=diag_vocab_size + 1,
            embedding_dim=emb_dim,
            padding_idx=0
        )

        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x_diag, x_proc=None, x_drug=None):
        """
        x_diag: (B, L_diag)，整数 token，0 为 padding
        返回: (B, num_classes)
        """
        device = x_diag.device
        # (B, L_diag, emb_dim)
        e_diag = self.emb_diag(x_diag)

        # mean pooling over non-padding tokens
        diag_mask = (x_diag != 0).unsqueeze(-1).float()  # (B, L_diag, 1)
        masked_embeddings = e_diag * diag_mask           # (B, L_diag, emb_dim)
        seq_lengths = diag_mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
        e_diag_aggregated = masked_embeddings.sum(dim=1) / seq_lengths  # (B, emb_dim)

        # MLP
        logits = self.net(e_diag_aggregated)  # (B, num_classes)
        return logits


class TransformerModel(nn.Module):
    def __init__(self, x_vocab_size, hidden, out_dim, *,
                 diag_size, proc_size,
                 num_heads=8, num_layers=3, p=0.3,
                 diag_itos=None, c=1.0, use_prior=True, max_diag_len=None):
        super().__init__()
        self.use_prior = use_prior
        self.diag_itos = diag_itos
        self.emb_diag  = nn.Embedding(diag_size + 1, hidden-2, padding_idx=0) if use_prior else nn.Embedding(diag_size + 1, hidden, padding_idx=0)
        # self.emb_proc  = nn.Embedding(proc_size + 1, embed_dim, padding_idx=0)
        # self.emb_third = nn.Embedding(x_vocab_size - (diag_size + proc_size) + 1, embed_dim, padding_idx=0)
        with open('sarkar_embeddings.pkl', 'rb') as f:
            self.sarkar_embeddings = pickle.load(f)

        enc_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=num_heads, dim_feedforward=hidden * 4, dropout=p, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden))
        self.output_projection = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(p)

        self.c = c  # 曲率 c > 0, Poincaré ball 半径 ~ 1/√c
        # 每个诊断 CCS class 的“超平面参数”：这里实现为：一个中心点 z_i（在球内）+ 一个偏置 r_i
        self.class_centers = nn.Parameter(1e-3 * torch.randn(out_dim, hidden))
        self.class_bias = nn.Parameter(torch.ones(out_dim) * 10.0)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def encode(self, x_diag, x_proc, x_drug):
        device = x_diag.device
        B = x_diag.shape[0]
        
        e_diag = self.emb_diag(x_diag)   # (B, L_diag, E)
        # e_proc = self.emb_proc(x_proc)   # (B, L_proc, E)
        # e_third = self.emb_third(x_drug) # (B, L_drug, E)
        if self.use_prior:
            e_diag_list = []
            for batch_item in x_diag:
                batch_embeddings = []
                for x in batch_item:
                    x_orig = x.item()
                    if x_orig == 0:  # padding token
                        batch_embeddings.append(np.zeros(2, dtype=np.float32))
                    else:
                        x_val = x_orig - 1
                        code = self.diag_itos[x_val]
                        batch_embeddings.append(np.array(self.sarkar_embeddings[code], dtype=np.float32))
                e_diag_list.append(batch_embeddings)
            e_diag_sarkar = torch.tensor(e_diag_list, device=device, dtype=torch.float32)
            e_diag = torch.cat([e_diag, e_diag_sarkar], dim=2)
        
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

        if self.use_prior:
            logits = self.hyperbolic_logits(x_cls)  # (B, out_dim)
        else:
            logits = self.output_projection(x_cls)  # (B, out_dim)

        return logits

    def _project_to_ball(self, x, eps=1e-5):
        """
        将向量投影回 Poincaré 球内，避免数值溢出：
        ||x|| < 1 / sqrt(c)
        """
        c = self.c
        sqrt_c = c ** 0.5
        norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=eps)
        max_norm = (1.0 - eps) / sqrt_c
        factor = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
        return x * factor

    def _expmap0(self, v, eps=1e-5):
        """
        以 0 为基点的 exp map:
        exp_0(v) = tanh(√c * ||v||) * v / (√c * ||v||)
        v: (..., dim)
        返回: (..., dim)，在 Poincaré 球内
        """
        c = self.c
        sqrt_c = c ** 0.5
        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=eps)
        factor = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)
        return factor * v

    def _poincare_dist(self, x, y, eps=1e-5):
        """
        Poincaré distance d_c(x, y):
        d(x, y) = arcosh(1 + 2c ||x-y||^2 / ((1 - c||x||^2)(1 - c||y||^2)))
        x: (B, E)
        y: (C, E)
        返回: (B, C)
        """
        c = self.c

        # (B, 1), (C,)
        x2 = (x ** 2).sum(dim=-1, keepdim=True)  # (B, 1)
        y2 = (y ** 2).sum(dim=-1)                # (C,)

        # pairwise ||x - y||^2, shape (B, C)
        diff2 = (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(dim=-1)

        denom = (1 - c * x2) * (1 - c * y2).unsqueeze(0)  # (B, C)
        denom = denom.clamp(min=eps)

        arg = 1 + 2 * c * diff2 / denom
        arg = arg.clamp(min=1 + 1e-7)  # acosh 的定义域 >= 1

        dist = torch.acosh(arg)
        return dist  # (B, C)

    def hyperbolic_logits(self, x_cls):
        """
        x_cls: (B, H)
        """
        x_cls = self._expmap0(x_cls)
        x_ball = self._project_to_ball(x_cls)
        centers = self._project_to_ball(self.class_centers)
        dist = self._poincare_dist(x_ball, centers)
        logits = -dist / self.temperature + self.class_bias.unsqueeze(0)
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
    - Uses **masked** self-attention (causal).
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
    - Uses **causal attention mask** (future tokens are masked).
    - Outputs **logits** for next-token prediction.
    """

    def __init__(
        self,
        manifold_in = Lorentz(1.0),
        manifold_hidden = Lorentz(1.0),
        manifold_out = Lorentz(1.0),
        arch = "L6_W390_A6",
        vocab_size = None,
        context_length = None,
        out_dim = None,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.grad_checkpointing = grad_checkpointing
        self.manifold_in = manifold_in
        self.manifold_hidden = manifold_hidden
        self.manifold_out = manifold_out
        self.out_dim = out_dim
        # Parse architecture string
        self.layers = int(re.search(r"L(\d+)", arch).group(1))
        self.width = int(re.search(r"W(\d+)", arch).group(1))
        _attn = re.search(r"A(\d+)", arch)
        self.heads = int(_attn.group(1)) if _attn else self.width // 64
        # Token Embeddings (Lorentz)
        self.token_embed = LorentzEmbeddings(manifold_in, vocab_size, self.width, manifold_out=manifold_hidden, posit_embed=False, padding_idx=0)  # +1 for padding token

        # Transformer Blocks (Decoder Only)
        self.resblocks = torch.nn.ModuleList([
            _LTransformerDecoderBlock(manifold_hidden, self.width, self.heads)
            for _ in range(self.layers)
        ])

        # Final normalization and projection
        self.ln_final = LorentzRMSNorm(manifold_hidden, self.width - 1)
        self.final_proj = LorentzLinear(manifold_hidden, self.width - 1, self.width - 1, manifold_out=manifold_hidden)

        self.mapping = torch.nn.Linear(self.width, self.out_dim, bias=False)

        # **Causal Attention Mask (Precomputed)**
        attn_mask = torch.triu(
            torch.full((context_length, context_length), float("-inf")), diagonal=1
        )
        self.register_buffer("attn_mask", attn_mask.bool())
        rope_vals = precompute_theta_pos_frequencies(self.width// self.heads, self.context_length)
        self.register_buffer("freqs_complex", rope_vals)

    def forward(self, 
            x_diag, x_proc, x_drug,
            attn_mask = None) -> torch.Tensor:
        batch_size, max_len = x_diag.shape
        
        padding_mask_1d = (x_diag == 0)  # (B, L) - True where padding
        attn_mask = padding_mask_1d.unsqueeze(1) | padding_mask_1d.unsqueeze(2)  # (B, L, L)
        _attn_mask = attn_mask  # (B, L, L)

        # Convert token indices to Lorentz manifold embeddings
        # This projects discrete token IDs into continuous hyperbolic space representations
        # Shape: (batch_size, context_length, width) where width includes time+space dimensions
        token_embeddings = self.token_embed(x_diag)
        freqs_cis = self.freqs_complex[:max_len]
        # freqs_cis = None
        decoder_features = token_embeddings
        
        # Forward pass through all transformer decoder blocks
        # Each block applies: Lorentz normalization -> bidirectional self-attention -> residual connection
        #                    -> Lorentz normalization -> feed-forward network -> residual connection
        for block in self.resblocks:
            decoder_features = block(decoder_features, _attn_mask, freqs_cis)

        # Apply final linear projection in Lorentz space
        # This transforms features while maintaining Lorentz manifold constraints
        # Shape: (batch_size, context_length, width) -> (batch_size, context_length, width-1)
        decoder_features = self.final_proj(decoder_features)
        # Apply final Lorentz RMS normalization
        # This normalizes the spatial components and recomputes time component to satisfy Lorentz constraints
        # Shape: (batch_size, context_length, width-1) -> (batch_size, context_length, width)
        decoder_features = self.ln_final(decoder_features)
        # Map from Lorentz manifold features to vocabulary logits using standard Euclidean linear layer
        # The Lorentz features (width dimensions) are projected to out_dim logits
        # Shape: (batch_size, context_length, hidden_dim(width)+1) -> (batch_size, context_length, out_dim)
        logits_per_position = self.mapping(decoder_features).float()  # (B, L, out_dim)
        # Aggregate sequence-level logits for classification
        # Use mean pooling over non-padding positions
        padding_mask_1d = (x_diag == 0)  # (B, L) - True where padding
        valid_mask = ~padding_mask_1d  # (B, L) - True where real tokens
        valid_mask = valid_mask.unsqueeze(-1).float()  # (B, L, 1)
        masked_logits = logits_per_position * valid_mask  # (B, L, out_dim)
        valid_lengths = valid_mask.sum(dim=1).clamp(min=1.0)  # (B, 1) - number of valid tokens per sample
        logits = masked_logits.sum(dim=1) / valid_lengths  # (B, out_dim)
        
        # Return logits for multi-label classification
        # Shape: (batch_size, out_dim)
        return logits


def create_model(model_type, x_vocab_size, hidden, out_dim, **kwargs):
    if model_type == 'transformer':
        return TransformerModel(x_vocab_size, hidden, out_dim, **kwargs)
    elif model_type == 'mlp':
        return MLPDiagLearned(x_vocab_size, hidden=hidden, num_classes=out_dim, diag_itos=kwargs.get('diag_itos'))
    elif model_type == 'lorentz_transformer':
        diag_size = kwargs.get('diag_size', x_vocab_size)
        return LTransformerDecoder(
            vocab_size=diag_size + 1,  # +1 for padding token (0)
            context_length=kwargs.get('max_diag_len'),
            out_dim=out_dim  # Output vocabulary size for final mapping
        )