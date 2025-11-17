"""
Model definition module
Contains MLP and Transformer model definitions
"""
import torch
import torch.nn as nn
import geoopt
import math

from torch.nn import functional as F

class PoincareMap(nn.Module):
    def __init__(self, in_dim, hyp_dim, c_init=1.0, gamma_init=0.01):
        super().__init__()
        self.proj = nn.Linear(in_dim, hyp_dim)
        self.log_c = nn.Parameter(torch.log(torch.tensor(c_init, dtype=torch.float32)))
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))
        nn.init.xavier_uniform_(self.proj.weight, gain=0.01)
        nn.init.zeros_(self.proj.bias)

    @property
    def c(self):
        return torch.exp(self.log_c)

    def forward(self, x, eps=1e-6):
        """
        x: (..., in_dim)
        返回: (..., hyp_dim) 的 Poincaré ball 上的点
        """
        # 1) 线性投影到超曲空间维度
        v = self.proj(x)                       # (..., hyp_dim)

        # 2) tanh + L2 normalize -> u
        u = torch.tanh(v)                      # 压缩数值
        u_norm = torch.norm(u, dim=-1, keepdim=True).clamp_min(eps)
        u = u / u_norm                         # 单位方向

        # 3) 乘一个可学习缩放 gamma 得到 h
        h = self.gamma * u                     # (..., hyp_dim)

        # 4) 按论文公式投到 Poincaré ball
        c = self.c
        sqrt_c = torch.sqrt(c)                 # 标量
        h_norm = torch.norm(h, dim=-1, keepdim=True).clamp_min(eps)
        direction = h / h_norm                 # 单位方向

        radial = torch.tanh(h_norm / sqrt_c)   # (..., 1)，控制半径
        z = sqrt_c * radial * direction        # (..., hyp_dim)

        return z


class TransformerModel(nn.Module):
    def __init__(self, x_vocab_size, hidden, out_dim, *,
                 diag_size, proc_size, embed_dim=256,
                 num_heads=8, num_layers=3, p=0.3,
                 max_diag_len=None, max_proc_len=None, max_drug_len=None, hyp_dim=32):
        super().__init__()
        V = x_vocab_size
        D = diag_size
        P = proc_size
        T = V - (D + P)  # 第三段大小
        self.hyp_dim = hyp_dim
        
        self.emb_diag  = nn.Embedding(D + 1, embed_dim, padding_idx=0)
        self.emb_proc  = nn.Embedding(P + 1, embed_dim, padding_idx=0)
        self.emb_third = nn.Embedding(T + 1, embed_dim, padding_idx=0)
        self.input_projection = nn.Linear(embed_dim, hidden)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=num_heads,
            dim_feedforward=hidden * 4, dropout=p, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden))
        self.output_projection = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(p)

        self.D = D
        self.P = P
        self.V = V
        self.max_diag_len = max_diag_len
        self.max_proc_len = max_proc_len
        self.max_drug_len = max_drug_len

        self.patient_hyp_head = PoincareMap(
            in_dim=hidden, 
            hyp_dim=self.hyp_dim
        )
        self.diag_hyp_head = PoincareMap(
            in_dim=embed_dim, 
            hyp_dim=self.hyp_dim
        )

    def encode(self, x_diag, x_proc, x_drug):
        device = x_diag.device
        B = x_diag.shape[0]
        
        e_diag = self.emb_diag(x_diag)   # (B, L_diag, E)
        e_proc = self.emb_proc(x_proc)   # (B, L_proc, E)
        e_third = self.emb_third(x_drug) # (B, L_drug, E)

        # 拼接三段序列
        x_embedded = torch.cat([e_diag, e_proc, e_third], dim=1)  # (B, L_total, E)
        
        # padding mask
        diag_mask = (x_diag != 0)   # (B, L_diag)
        proc_mask = (x_proc != 0)   # (B, L_proc)
        drug_mask = (x_drug != 0)   # (B, L_drug)
        padding_mask = torch.cat([diag_mask, proc_mask, drug_mask], dim=1)  # (B, L_total)

        x_projected = self.input_projection(x_embedded)  # (B, L_total, H)
    
        padding_mask_expanded = padding_mask.unsqueeze(-1)  # (B, L_total, 1)
        x_projected = x_projected.masked_fill(~padding_mask_expanded, 0.)

        cls_tokens = self.cls_token.expand(B, 1, -1)        # (B, 1, H)
        x_seq = torch.cat([cls_tokens, x_projected], dim=1) # (B, L_total+1, H)
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
        key_padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
        src_key_padding_mask = ~key_padding_mask  # True = mask out

        x_seq = self.transformer(x_seq, src_key_padding_mask=src_key_padding_mask)

        x_cls = self.dropout(x_seq[:, 0, :])  # (B, H)
        return x_cls

    def forward(self, x_diag, x_proc, x_drug):
        x_cls = self.encode(x_diag, x_proc, x_drug)  # (B, H)
        logits = self.output_projection(x_cls) 
        z_patient = self.patient_hyp_head(x_cls)        # (B, out_dim)
        return logits, z_patient

    def get_diag_hyperbolic(self):
        E_diag = self.emb_diag.weight         # (D+1, embed_dim)
        Z_diag = self.diag_hyp_head(E_diag)   # (D+1, hyp_dim) in hyperbolic space
        return Z_diag


def create_model(model_type, x_vocab_size, hidden, out_dim, **kwargs):
    return TransformerModel(x_vocab_size, hidden, out_dim, **kwargs)
