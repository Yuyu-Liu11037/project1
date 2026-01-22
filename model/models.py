import torch
import torch.nn as nn
import math
import re

from hypercore.nn.linear import LorentzLinear
from hypercore.nn.conv import LResNet, LorentzRMSNorm, LorentzActivation
from hypercore.nn.attention import LorentzMultiheadAttention, LorentzEmbeddings
from hypercore.manifolds import Lorentz
from geoopt import ManifoldParameter


class TransformerEncoder(nn.Module):
    def __init__(self, x_vocab_size, hidden=390, out_dim=None, *,
                 diag_size, proc_size,
                 num_heads=6, num_layers=3, 
                 diag_itos=None, c=1.0, max_diag_len=None, arch=None):
        super().__init__()
        self.diag_itos = diag_itos
        # Keep Euclidean Transformer configurable via the same `arch` string as LTransformerEncoder.
        # Expected format: "L{layers}_W{width}_A{heads}" (e.g. "L3_W390_A6")
        if arch:
            try:
                num_layers = int(re.search(r"L(\d+)", arch).group(1))
                hidden = int(re.search(r"W(\d+)", arch).group(1))
                _attn = re.search(r"A(\d+)", arch)
                num_heads = int(_attn.group(1)) if _attn else num_heads
            except Exception:
                # Fall back to provided args if arch parsing fails
                pass
        self.token_embed  = nn.Embedding(diag_size + 1, hidden, padding_idx=0)
        # self.emb_proc  = nn.Embedding(proc_size + 1, embed_dim, padding_idx=0)
        # self.emb_third = nn.Embedding(x_vocab_size - (diag_size + proc_size) + 1, embed_dim, padding_idx=0)

        enc_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=num_heads, dim_feedforward=hidden * 4, dropout=0.3, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden))
        self.classifier = nn.Linear(hidden, out_dim, bias=False)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x_diag, x_proc, x_drug, x_visit_ids=None):
        batch_size, max_len = x_diag.shape
        device = x_diag.device

        cls_token = self.cls_token.expand(batch_size, 1, -1)        # (batch_size, 1, H)
        token_embeddings = self.token_embed(x_diag)   # (batch_size, L_diag, E)
        token_embeddings = torch.cat([cls_token, token_embeddings], dim=1) # (batch_size, L_total+1, H)

        # PyTorch transformer expects src_key_padding_mask=True where positions should be masked (i.e. padding).
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)  # CLS is never padding
        padding_mask = (x_diag == 0)   # (batch_size, L_diag) - True where padding
        padding_mask = torch.cat([cls_mask, padding_mask], dim=1)  # (batch_size, L_diag+1)

        token_embeddings = self.transformer(token_embeddings, src_key_padding_mask=padding_mask)

        cls_state = self.dropout(token_embeddings[:, 0, :])  # (batch_size, H)
        logits = self.classifier(cls_state)  # (batch_size, out_dim)

        # Match LTransformerEncoder outputs: (logits, visit_states, visit_padding_mask)
        code_states = token_embeddings[:, 1:, :]  # (B, L, H)
        code_padding_mask = (x_diag == 0)         # (B, L), True where padding

        if x_visit_ids is None:
            # Default: treat all (non-padding) codes as belonging to a single visit (visit_id=0).
            x_visit_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        else:
            # Ensure padding positions are marked as invalid
            x_visit_ids = x_visit_ids.to(device)
            if x_visit_ids.shape != (batch_size, max_len):
                raise ValueError(f"x_visit_ids must have shape {(batch_size, max_len)}, got {tuple(x_visit_ids.shape)}")
            x_visit_ids = x_visit_ids.clone()
        x_visit_ids[code_padding_mask] = -1

        valid_visit_ids = x_visit_ids[x_visit_ids >= 0]
        V = int(valid_visit_ids.max().item() + 1) if valid_visit_ids.numel() > 0 else 1

        visit_states = code_states.new_zeros(batch_size, V, code_states.size(-1))  # (B, V, H)
        visit_counts = code_states.new_zeros(batch_size, V, 1)                    # (B, V, 1)

        for b in range(batch_size):
            valid_mask_b = x_visit_ids[b] >= 0
            if not valid_mask_b.any():
                continue
            ids_b = x_visit_ids[b, valid_mask_b]          # (#codes_b,)
            states_b = code_states[b, valid_mask_b, :]    # (#codes_b, H)
            visit_states[b].index_add_(0, ids_b, states_b)
            ones = torch.ones((ids_b.size(0), 1), dtype=visit_counts.dtype, device=device)
            visit_counts[b].index_add_(0, ids_b, ones)

        visit_states = visit_states / visit_counts.clamp(min=1.0)
        visit_padding_mask = (visit_counts.squeeze(-1) == 0)  # (B, V), True where empty visit
        return logits, visit_states, visit_padding_mask


class MLP(nn.Module):
    def __init__(self, x_vocab_size, hidden=390, out_dim=None, *,
                 diag_size, proc_size,
                 num_layers=3,
                 diag_itos=None, c=1.0, max_diag_len=None, arch=None, dropout=0.3):
        super().__init__()
        self.diag_itos = diag_itos
        self.token_embed = nn.Embedding(diag_size + 1, hidden, padding_idx=0)
        # self.emb_proc  = nn.Embedding(proc_size + 1, embed_dim, padding_idx=0)
        # self.emb_third = nn.Embedding(x_vocab_size - (diag_size + proc_size) + 1, embed_dim, padding_idx=0)

        # Build MLP layers
        mlp_layers = []
        for i in range(num_layers):
            mlp_layers.append(nn.Linear(hidden, hidden))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*mlp_layers)

        self.classifier = nn.Linear(hidden, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_diag, x_proc, x_drug):
        batch_size, max_len = x_diag.shape
        device = x_diag.device

        # Embed tokens
        token_embeddings = self.token_embed(x_diag)   # (batch_size, L_diag, H)
        
        # Create mask for padding tokens (0 is padding)
        padding_mask = (x_diag != 0).float()  # (batch_size, L_diag)
        padding_mask = padding_mask.unsqueeze(-1)  # (batch_size, L_diag, 1)
        
        # Masked mean pooling: average over non-padding tokens
        masked_embeddings = token_embeddings * padding_mask  # (batch_size, L_diag, H)
        seq_lengths = padding_mask.sum(dim=1, keepdim=True)  # (batch_size, 1, 1)
        seq_lengths = torch.clamp(seq_lengths, min=1.0)  # Avoid division by zero
        pooled = masked_embeddings.sum(dim=1) / seq_lengths.squeeze(-1)  # (batch_size, H)
        
        # Apply MLP layers
        pooled = self.mlp(pooled)  # (batch_size, H)
        
        # Final classification
        pooled = self.dropout(pooled)  # (batch_size, H)
        logits = self.classifier(pooled)  # (batch_size, out_dim)
        return logits


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


class _LTransformerEncoderBlock(torch.nn.Module):
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
        ax = self.attn(lx, lx, output_attentions=False, mask=attn_mask, rot_pos=rope) 
        x = self.res1(x, ax)    
        x = self.res2(x, self.mlp(self.ln_2(x)))
        return x
    

class LTransformerEncoder(torch.nn.Module):
    def __init__(
        self,
        manifold_in = Lorentz(1.0),
        manifold_hidden = Lorentz(1.0),
        manifold_output = Lorentz(1.0),
        arch = "L3_W390_A6",
        vocab_size = None,
        context_length = None,
        out_dim = None,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.manifold_hidden = manifold_hidden
        # Parse architecture string
        self.layers = int(re.search(r"L(\d+)", arch).group(1))
        self.width = int(re.search(r"W(\d+)", arch).group(1))
        _attn = re.search(r"A(\d+)", arch)
        self.heads = int(_attn.group(1)) if _attn else self.width // 64
        # Token Embeddings (Lorentz)
        self.token_embed = LorentzEmbeddings(manifold_in, vocab_size,  self.width, manifold_out=manifold_hidden, padding_idx=0) 
        self.cls_token = ManifoldParameter(manifold_hidden.random_normal((1, 1, self.width), std=0.02), manifold=manifold_hidden)

        self.resblocks = torch.nn.ModuleList([
            _LTransformerEncoderBlock(manifold_hidden, self.width, self.heads)
            for _ in range(self.layers)
        ])

        # Final normalization and projection
        self.ln_final = LorentzRMSNorm(manifold_hidden, self.width - 1)
        self.final_proj = LorentzLinear(manifold_hidden, self.width - 1, self.width - 1, manifold_out=manifold_hidden)
        self.dropout = nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(self.width, out_dim)

    def forward(self, x_diag, x_proc, x_drug, x_visit_ids=None):
        batch_size, max_len = x_diag.shape
        device = x_diag.device

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        token_embeddings = self.token_embed(x_diag)   # (batch_size, max_len, width)
        token_embeddings = torch.cat([cls_token, token_embeddings], dim=1)
        
        cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
        padding_mask = (x_diag == 0)  # (batch_size, max_len) - True where padding
        padding_mask = torch.cat([cls_mask, padding_mask], dim=1)  # (batch_size, max_len+1)

        _attn_mask = padding_mask.unsqueeze(1).expand(-1, max_len+1, -1)  # (batch_size, max_len+1, max_len+1)
        # Each block applies: Lorentz normalization -> bidirectional self-attention -> residual connection
        #                    -> Lorentz normalization -> feed-forward network -> residual connection
        for block in self.resblocks:
            token_embeddings = block(token_embeddings, _attn_mask)
        token_embeddings = self.final_proj(token_embeddings)
        token_embeddings = self.ln_final(token_embeddings)

        cls_state = self.dropout(token_embeddings[:, 0, :])
        logits = self.classifier(cls_state)

        # token_embeddings: (B, L+1, D)
        # 去掉第 0 个 CLS，只看 code token
        code_states = token_embeddings[:, 1:, :]          # (B, L, D)
        B, L, D = code_states.shape                  # (B, L)

        # padding_mask 包含 CLS token，形状为 (B, L+1)
        # 需要去掉 CLS token 部分，只保留 code tokens 的 mask，形状为 (B, L)
        code_padding_mask = padding_mask[:, 1:]  # (B, L), True where padding

        # 当前 batch 中的最大 visit index
        # 注意：如果你用 -1 表示 padding visit，要先把无效位置 mask 掉再取 max
        masked_visit_ids = x_visit_ids.clone()
        # code_padding_mask 是 True where padding，所以 ~code_padding_mask 是 True where valid
        masked_visit_ids[code_padding_mask] = -1  # 将 padding 位置设为 -1
        # 只考虑非 padding 的位置来取 max
        valid_visit_ids = masked_visit_ids[masked_visit_ids >= 0]
        max_visits = valid_visit_ids.max().item() + 1 if len(valid_visit_ids) > 0 else 1   # V

        V = max_visits

        manifold = self.manifold_hidden
        code_states_tan = manifold.logmap0(code_states)  # (B, L, D') —— 假设返回同维度
        visit_states_tan = code_states_tan.new_zeros(B, V, code_states_tan.size(-1))  # (B, V, D')
        visit_counts = code_states_tan.new_zeros(B, V, 1)                             # (B, V, 1)

        for b in range(B):
            # code_padding_mask[b] 形状为 (L,), True where padding
            # 我们需要 valid_mask_b 是 True where valid (not padding)
            valid_mask_b = ~code_padding_mask[b]  # (L,), True where valid
            if not valid_mask_b.any():
                continue

            ids_b = x_visit_ids[b, valid_mask_b]             # (#codes_b,)
            states_b = code_states_tan[b, valid_mask_b, :]   # (#codes_b, D')
            
            # 确保只使用有效的 visit ids (>= 0)，过滤掉 -1
            valid_ids_mask = ids_b >= 0
            if not valid_ids_mask.any():
                continue
            ids_b = ids_b[valid_ids_mask]
            states_b = states_b[valid_ids_mask, :]

            visit_states_tan[b].index_add_(0, ids_b, states_b)

            ones = torch.ones_like(ids_b, dtype=visit_counts.dtype, device=device).unsqueeze(-1)  # (#codes_b, 1)
            visit_counts[b].index_add_(0, ids_b, ones)

        visit_counts_clamped = visit_counts.clamp(min=1.0)
        visit_states_tan = visit_states_tan / visit_counts_clamped   # (B, V, D')
        visit_states = manifold.expmap0(visit_states_tan)            # (B, V, D)
        visit_padding_mask = (visit_counts.squeeze(-1) == 0)         # (B, V), True 表示这个 visit 其实是“空”的
        return logits, visit_states, visit_padding_mask


def create_model(model_type, x_vocab_size, out_dim, **kwargs):
    if model_type == 'transformer_encoder':
        return TransformerEncoder(x_vocab_size=x_vocab_size, out_dim=out_dim, **kwargs)
    elif model_type == 'mlp':
        return MLP(x_vocab_size=x_vocab_size, out_dim=out_dim, **kwargs)
    elif model_type == 'ltransformer_encoder':
        return LTransformerEncoder(
            vocab_size=x_vocab_size,
            context_length=kwargs.get('max_diag_len'),
            out_dim=out_dim,  # Output vocabulary size for final mapping
            arch=kwargs.get('arch'),
        )
