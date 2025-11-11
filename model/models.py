import torch
import torch.nn as nn
import torch.nn.functional as F

def project_to_ball(x, c, eps=1e-5):
    r = 1.0 / (c**0.5)
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    max_norm = (1 - eps) * r
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale

def artanh(x):  # 数值稳定的 atanh
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

def logmap0_poincare(x, c):
    x = project_to_ball(x, c)
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    return (2.0 / (c**0.5)) * artanh((c**0.5) * norm) * (x / norm)

class PoincareToEuclid(nn.Module):
    def __init__(self, in_dim, out_dim, c_init=1.0):
        super().__init__()
        self.logit_c = nn.Parameter(torch.tensor(float(c_init)).log())  # 可学习曲率
        self.norm = nn.LayerNorm(out_dim)

    @property
    def c(self):
        return self.logit_c.exp()

    def forward(self, x_h):  # x_h 在 Poincaré ball
        x_tan = logmap0_poincare(x_h, self.c)
        return self.norm(x_tan)


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, hidden, out_dim, num_heads=8, num_layers=3, p=0.3, save_data=None, diag_itos=None):
        super().__init__()
        
        self.embedding_layer = nn.Embedding(vocab_size, hidden)
        self.save_data = save_data
        self.diag_itos = diag_itos  # Mapping from token ID to code string
        # self.pos_encoding = PositionalEncoding(hidden, p)
        
        if self.save_data is not None:
            self.id_map = self.save_data['id_map']  # Maps code strings -> embedding indices
            self.emb_data = self.save_data['model'].emb.data  # (num_codes, dim)
            
            max_token_id = len(self.diag_itos) if self.diag_itos else 0
            token_to_emb_idx = torch.full((max_token_id,), -1, dtype=torch.long)
            
            for token_id in range(max_token_id):
                if token_id == 0:  # Padding token
                    continue  # Keep as -1
                code_str = self.diag_itos[token_id]
                if code_str not in self.id_map:  # skip ICD-9 codes
                    print(code_str)
                    continue
                emb_idx = self.id_map[code_str]
                token_to_emb_idx[token_id] = emb_idx
            
            self.register_buffer('token_to_emb_idx', token_to_emb_idx)
        else:
            self.id_map = None
            self.emb_data = None
            self.token_to_emb_idx = None
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=num_heads,
            dim_feedforward=hidden * 4,
            dropout=p,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.adapter = PoincareToEuclid(in_dim=hidden, out_dim=hidden, c_init=1.0)
        
        self.output_projection = nn.Linear(hidden, out_dim)
        
        self.dropout = nn.Dropout(p)
    
    def forward(self, x):
        # x: (batch_size, seq_len) - contains integer token IDs from diag_stoi
        if self.emb_data is not None:
            # by default, emb_dim = hidden
            batch_size, seq_len = x.size()
            emb_dim = self.emb_data.size(1)
            x_flat = x.view(-1)  # (batch_size * seq_len,)
            
            valid_token_ids = torch.clamp(x_flat, 0, len(self.token_to_emb_idx) - 1)
            emb_indices = self.token_to_emb_idx[valid_token_ids]  # (batch_size * seq_len,)
            
            valid_mask = (x_flat > 0) & (emb_indices >= 0) & (emb_indices < self.emb_data.size(0))
            embeddings = torch.zeros(batch_size * seq_len, emb_dim, device='cuda', dtype=self.emb_data.dtype)
            
            if valid_mask.any():
                valid_emb_indices = emb_indices[valid_mask]
                embeddings[valid_mask] = self.emb_data[valid_emb_indices].to('cuda')
            
            x = embeddings.view(batch_size, seq_len, emb_dim)
        else:
            x = self.embedding_layer(x)  # (batch_size, seq_len, hidden)
        # x = self.pos_encoding(x)
        x = self.adapter(x)
        x = self.transformer(x)
        x_output = x[:, 0, :]  # (batch_size, hidden)
        x_output = self.dropout(x_output)
        output = self.output_projection(x_output)  # (batch_size, out_dim)
        
        return output


def create_model(model_type, vocab_size, hidden, out_dim, **kwargs):
    return TransformerModel(vocab_size, hidden, out_dim, **kwargs)

