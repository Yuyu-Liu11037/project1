"""
Model definition module
Contains MLP and Transformer model definitions
"""
import torch
import torch.nn as nn
import geoopt
import math


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

    @torch.no_grad()
    def reproject_(self):
        self.weight.data = self.ball.projx(self.weight.data)


class TransformerModel(nn.Module):
    """
    三段区间分别做 hyperbolic embedding 的多标签分类模型
    区间：
      1) 1..D
      2) D+1..D+P
      3) D+P+1..V
    其余补0
    """
    def __init__(self, x_vocab_size, hidden, out_dim, *,
                 diag_size, proc_size, embed_dim=256,
                 num_heads=8, num_layers=3, p=0.3,
                 c_diag=1.0, c_proc=1.0, c_third=1.0,
                 max_diag_len=None, max_proc_len=None, max_drug_len=None):
        super().__init__()
        V = x_vocab_size
        D = diag_size
        P = proc_size
        T = V - (D + P)  # 第三段大小

        # 三段各一套 hyperbolic embedding（局部索引0留给padding）
        self.emb_diag  = HyperbolicEmbedding(D + 1, embed_dim, c=c_diag,  padding_idx=0)
        self.emb_proc  = HyperbolicEmbedding(P + 1, embed_dim, c=c_proc,  padding_idx=0)
        self.emb_third = HyperbolicEmbedding(T + 1, embed_dim, c=c_third, padding_idx=0)

        # 之后仍在欧式空间里跑
        self.input_projection = nn.Linear(embed_dim, hidden)
        self.pos_encoding = PositionalEncoding(hidden, p)

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
        
        # Embed each type separately (already using local indices)
        e_diag = self.emb_diag(x_diag)   # (B, L_diag, E)
        e_proc = self.emb_proc(x_proc)   # (B, L_proc, E)
        e_third = self.emb_third(x_drug) # (B, L_drug, E)

        # Concatenate the three sequences along the sequence dimension
        x_embedded = torch.cat([e_diag, e_proc, e_third], dim=1)  # (B, L_total, E)
        
        # Get padding masks for each type
        diag_mask = (x_diag != 0)   # (B, L_diag)
        proc_mask = (x_proc != 0)   # (B, L_proc)
        drug_mask = (x_drug != 0)   # (B, L_drug)
        
        # Concatenate masks
        padding_mask = torch.cat([diag_mask, proc_mask, drug_mask], dim=1)  # (B, L_total)

        # Project to hidden dimension
        x_projected = self.input_projection(x_embedded)  # (B, L_total, H)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, 1, -1)
        x_seq = torch.cat([cls_tokens, x_projected], dim=1)  # (B, L_total+1, H)

        # Extend padding mask for CLS token (always valid)
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
        key_padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
        src_key_padding_mask = ~key_padding_mask  # True = mask out

        # Positional encoding & Transformer encoder
        x_seq = self.pos_encoding(x_seq)
        x_seq = self.transformer(x_seq, src_key_padding_mask=src_key_padding_mask)

        # Use CLS token for output
        x_cls = self.dropout(x_seq[:, 0, :])
        return self.output_projection(x_cls)

    @torch.no_grad()
    def reproject_hyperbolic_(self):
        self.emb_diag.reproject_()
        self.emb_proc.reproject_()
        self.emb_third.reproject_()


class PositionalEncoding(nn.Module):
    """Positional encoding"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


def create_model(model_type, x_vocab_size, hidden, out_dim, **kwargs):
    """Model factory function"""
    return TransformerModel(x_vocab_size, hidden, out_dim, **kwargs)

