"""
Model definition module
Contains MLP and Transformer model definitions
"""
import torch
import torch.nn as nn
import math


class MLP(nn.Module):
    """Simple multi-label MLP model with embedding layer for index-based inputs"""
    
    def __init__(self, x_vocab_size, hidden, out_dim, embed_dim=256, p=0.3):
        super().__init__()
        # Embedding layer to convert sparse indices to dense vectors
        self.embedding = nn.Embedding(x_vocab_size, embed_dim, padding_idx=0)
        
        # Use EmbeddingBag for efficient handling of variable-length sequences
        # This automatically handles padding and averaging
        self.embedding_bag = nn.EmbeddingBag(x_vocab_size, embed_dim, padding_idx=0, mode='mean')
        
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden), 
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, out_dim)  # logits for each class
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len) where seq_len can vary
        # Use embedding bag to aggregate variable-length sequences
        x_embedded = self.embedding_bag(x)  # (batch_size, embed_dim)
        return self.net(x_embedded)  # (batch_size, out_dim)


class TransformerModel(nn.Module):
    """Transformer-based multi-label classification model with embedding layer"""
    
    def __init__(self, x_vocab_size, hidden, out_dim, embed_dim=256, num_heads=8, num_layers=3, p=0.3):
        super().__init__()
        self.x_vocab_size = x_vocab_size
        self.hidden = hidden
        self.out_dim = out_dim
        
        # Embedding layer for input indices
        self.embedding = nn.Embedding(x_vocab_size, embed_dim, padding_idx=0)
        
        # Project embeddings to hidden dimension
        self.input_projection = nn.Linear(embed_dim, hidden)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden, p)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=num_heads,
            dim_feedforward=hidden * 4,
            dropout=p,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer - use a classification token approach
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden))
        self.output_projection = nn.Linear(hidden, out_dim)
        
        # Dropout
        self.dropout = nn.Dropout(p)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len) - variable length sequences of indices
        batch_size = x.size(0)
        
        # Convert padding to mask for transformer (0 = padding, 1 = valid)
        padding_mask = (x != 0)  # (batch_size, seq_len)
        
        # Embedding: convert indices to dense vectors
        x_embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        
        # Project to hidden dimension
        x_projected = self.input_projection(x_embedded)  # (batch_size, seq_len, hidden)
        
        # Add CLS token at the beginning
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, hidden)
        x_seq = torch.cat([cls_tokens, x_projected], dim=1)  # (batch_size, seq_len+1, hidden)
        
        # Extend padding mask for CLS token (always valid)
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=x.device)
        padding_mask = torch.cat([cls_mask, padding_mask], dim=1)  # (batch_size, seq_len+1)
        
        # Positional encoding
        x_seq = self.pos_encoding(x_seq)
        
        # Transformer encoding
        # Create attention mask: True values will be ignored (padded positions)
        src_key_padding_mask = ~padding_mask  # Invert: True = masked out
        x_seq = self.transformer(x_seq, src_key_padding_mask=src_key_padding_mask)
        
        # Use CLS token (first token) for prediction
        x_output = x_seq[:, 0, :]  # (batch_size, hidden)
        
        # Dropout
        x_output = self.dropout(x_output)
        
        # Output projection
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


def create_model(model_type, x_vocab_size, hidden, out_dim, **kwargs):
    """Model factory function"""
    if model_type.lower() == 'mlp':
        return MLP(x_vocab_size, hidden, out_dim, **kwargs)
    elif model_type.lower() == 'transformer':
        return TransformerModel(x_vocab_size, hidden, out_dim, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: 'mlp', 'transformer'")

