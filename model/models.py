"""
Model definition module
Contains MLP and Transformer model definitions
"""
import torch
import torch.nn as nn
import geoopt
import math


class TransformerModel(nn.Module):
    def __init__(self, x_vocab_size, hidden, out_dim, *,
                 diag_size, proc_size, embed_dim=256,
                 num_heads=8, num_layers=3, p=0.3,
                 max_diag_len=None, max_proc_len=None, max_drug_len=None,
                 diag_itos=None, max_depth=None):
        super().__init__()
        V = x_vocab_size
        D = diag_size
        P = proc_size
        T = V - (D + P)  # 第三段大小
        
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

    def forward(self, x_diag, x_proc, x_drug):
        """
        x_diag: (B, L_diag), diagnosis codes, 0=padding
        x_proc: (B, L_proc), procedure codes, 0=padding
        x_drug: (B, L_drug), drug codes, 0=padding
        Each code type uses local indexing (0-indexed in its own vocabulary)
        """
        device = x_diag.device
        B = x_diag.shape[0]
        
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

        x_projected = self.input_projection(x_embedded)  # (B, L_total, H)
    
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


def create_model(model_type, x_vocab_size, hidden, out_dim, **kwargs):
    """Model factory function"""
    return TransformerModel(x_vocab_size, hidden, out_dim, **kwargs)

