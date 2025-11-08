import torch
import torch.nn as nn
import math


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, hidden, out_dim, num_heads=8, num_layers=3, p=0.3, save_data=None, diag_itos=None):
        super().__init__()
        
        self.embedding_layer = nn.Embedding(vocab_size, hidden)
        self.save_data = save_data
        self.diag_itos = diag_itos  # Mapping from token ID to code string
        self.pos_encoding = PositionalEncoding(hidden, p)
        
        # Initialize hyperbolic embeddings data if available
        if self.save_data is not None:
            self.id_map = self.save_data['id_map']  # Maps code strings -> embedding indices
            self.emb_data = self.save_data['model'].emb.data  # (num_codes, dim)
            
            # Pre-build token_id -> emb_idx mapping tensor for vectorized lookup
            # This avoids Python loops and dictionary lookups in forward pass
            max_token_id = len(self.diag_itos) if self.diag_itos else 0
            token_to_emb_idx = torch.full((max_token_id,), -1, dtype=torch.long)
            
            for token_id in range(max_token_id):
                if token_id == 0:  # Padding token
                    continue  # Keep as -1
                code_str = self.diag_itos[token_id]
                if code_str in self.id_map:
                    emb_idx = self.id_map[code_str]
                    if emb_idx < self.emb_data.size(0):
                        token_to_emb_idx[token_id] = emb_idx
            
            # Register as buffer so it moves to correct device automatically
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
        
        self.output_projection = nn.Linear(hidden, out_dim)
        
        self.dropout = nn.Dropout(p)
    
    def forward(self, x):
        # x: (batch_size, seq_len) - contains integer token IDs from diag_stoi
        if self.emb_data is not None:
            batch_size, seq_len = x.size()
            emb_dim = self.emb_data.size(1)
            device = x.device
            
            # Flatten input for batch processing
            x_flat = x.view(-1)  # (batch_size * seq_len,)
            
            # Vectorized lookup: token_id -> emb_idx
            # Clamp token_ids to valid range to avoid index errors
            valid_token_ids = torch.clamp(x_flat, 0, len(self.token_to_emb_idx) - 1)
            emb_indices = self.token_to_emb_idx[valid_token_ids]  # (batch_size * seq_len,)
            
            # Create mask for valid embeddings (not padding and has valid emb_idx)
            valid_mask = (x_flat > 0) & (emb_indices >= 0) & (emb_indices < self.emb_data.size(0))
            
            # Initialize embeddings tensor (padding tokens remain zero)
            embeddings = torch.zeros(batch_size * seq_len, emb_dim, device=device, dtype=self.emb_data.dtype)
            
            # Batch copy valid embeddings using advanced indexing
            if valid_mask.any():
                valid_emb_indices = emb_indices[valid_mask]
                embeddings[valid_mask] = self.emb_data[valid_emb_indices].to(device)
            
            # Reshape back to (batch_size, seq_len, emb_dim)
            x = embeddings.view(batch_size, seq_len, emb_dim)
        else:
            x = self.embedding_layer(x)  # (batch_size, seq_len, hidden)
        # x = self.pos_encoding(x)
        x = self.transformer(x)
        x_output = x[:, 0, :]  # (batch_size, hidden)
        x_output = self.dropout(x_output)
        output = self.output_projection(x_output)  # (batch_size, out_dim)
        
        return output


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


def create_model(model_type, vocab_size, hidden, out_dim, **kwargs):
    return TransformerModel(vocab_size, hidden, out_dim, **kwargs)

