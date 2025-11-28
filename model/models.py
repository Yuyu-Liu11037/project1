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
                 diag_itos=None, c=1.0, max_diag_len=None):
        super().__init__()
        self.diag_itos = diag_itos
        self.token_embed  = nn.Embedding(diag_size + 1, hidden, padding_idx=0)
        # self.emb_proc  = nn.Embedding(proc_size + 1, embed_dim, padding_idx=0)
        # self.emb_third = nn.Embedding(x_vocab_size - (diag_size + proc_size) + 1, embed_dim, padding_idx=0)

        enc_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=num_heads, dim_feedforward=hidden * 4, dropout=p, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden))
        self.classifier = nn.Linear(hidden, out_dim, bias=False)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x_diag, x_proc, x_drug):
        batch_size, max_len = x_diag.shape
        device = x_diag.device

        cls_token = self.cls_token.expand(batch_size, 1, -1)        # (batch_size, 1, H)
        token_embeddings = self.token_embed(x_diag)   # (batch_size, L_diag, E)
        token_embeddings = torch.cat([cls_token, token_embeddings], dim=1) # (batch_size, L_total+1, H)

        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        padding_mask = (x_diag != 0)   # (batch_size, L_diag)
        padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        token_embeddings = self.transformer(token_embeddings, src_key_padding_mask=~padding_mask)

        cls_state = self.dropout(token_embeddings[:, 0, :])  # (batch_size, H)
        logits = self.classifier(cls_state)  # (batch_size, out_dim)
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
        manifold = Lorentz(1.0),
        arch = "L3_W390_A6",
        vocab_size = None,
        context_length = None,
        out_dim = None,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        # Parse architecture string
        self.layers = int(re.search(r"L(\d+)", arch).group(1))
        self.width = int(re.search(r"W(\d+)", arch).group(1))
        _attn = re.search(r"A(\d+)", arch)
        self.heads = int(_attn.group(1)) if _attn else self.width // 64
        # Token Embeddings (Lorentz)
        self.token_embed = LorentzEmbeddings(manifold, vocab_size, self.width, padding_idx=0) 
        self.cls_token = ManifoldParameter(manifold.random_normal((1, 1, self.width), std=0.02), manifold=manifold)

        self.resblocks = torch.nn.ModuleList([
            _LTransformerEncoderBlock(manifold, self.width, self.heads)
            for _ in range(self.layers)
        ])

        # Final normalization and projection
        self.ln_final = LorentzRMSNorm(manifold, self.width - 1)
        self.final_proj = LorentzLinear(manifold, self.width - 1, self.width - 1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(self.width, out_dim)

    def forward(self, x_diag, x_proc, x_drug):
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
        return logits


def create_model(model_type, x_vocab_size, out_dim, **kwargs):
    if model_type == 'transformer_encoder':
        return TransformerEncoder(x_vocab_size=x_vocab_size, out_dim=out_dim, **kwargs)
    elif model_type == 'ltransformer_encoder':
        return LTransformerEncoder(
            vocab_size=x_vocab_size,
            context_length=kwargs.get('max_diag_len'),
            out_dim=out_dim,  # Output vocabulary size for final mapping
            arch=kwargs.get('arch'),
        )
