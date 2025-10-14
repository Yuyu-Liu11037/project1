"""
Model definition module
Contains MLP and Transformer model definitions
"""
import torch
import torch.nn as nn
import math


class MLP(nn.Module):
    """Simple multi-label MLP model"""
    
    def __init__(self, in_dim, hidden, out_dim, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), 
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, out_dim)  # logits
        )
    
    def forward(self, x): 
        return self.net(x)


class TransformerModel(nn.Module):
    """Transformer-based multi-label classification model"""
    
    def __init__(self, in_dim, hidden, out_dim, num_heads=8, num_layers=3, p=0.3):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.out_dim = out_dim
        
        # Input projection layer - project input features to hidden dimension
        self.input_projection = nn.Linear(in_dim, hidden)
        
        # Create a learnable sequence of tokens for the transformer
        # We'll use a fixed number of learnable tokens instead of reshaping input
        self.num_tokens = 16  # Fixed number of learnable tokens
        self.token_embeddings = nn.Parameter(torch.randn(self.num_tokens, hidden))
        
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
        
        # Output layer
        self.output_projection = nn.Linear(hidden, out_dim)
        
        # Dropout
        self.dropout = nn.Dropout(p)
        
    def forward(self, x):
        # x shape: (batch_size, in_dim)
        batch_size = x.size(0)
        
        # Project input features to hidden dimension
        # x: (batch_size, in_dim) -> (batch_size, hidden)
        x_projected = self.input_projection(x)  # (batch_size, hidden)
        
        # Create sequence by combining projected input with learnable tokens
        # Expand learnable tokens for batch
        tokens = self.token_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, num_tokens, hidden)
        
        # Add input as first token
        x_input = x_projected.unsqueeze(1)  # (batch_size, 1, hidden)
        
        # Combine input with learnable tokens
        x_seq = torch.cat([x_input, tokens], dim=1)  # (batch_size, num_tokens+1, hidden)
        
        # Positional encoding
        x_seq = self.pos_encoding(x_seq)
        
        # Transformer encoding
        x_seq = self.transformer(x_seq)
        
        # Use the first token (which contains input information) for prediction
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


class SequentialTransformerModel(nn.Module):
    """Transformer model for sequential input (batch_size, seq_len, embedding_dim)"""
    
    def __init__(self, embedding_dim, hidden, out_dim, num_heads=8, num_layers=3, p=0.3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden = hidden
        self.out_dim = out_dim
        
        # Project embedding dimension to hidden dimension
        self.input_projection = nn.Linear(embedding_dim, hidden)
        
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
        
        # Output layer
        self.output_projection = nn.Linear(hidden, out_dim)
        
        # Dropout
        self.dropout = nn.Dropout(p)
        
    def forward(self, x, attention_mask=None):
        # x shape: (batch_size, seq_len, embedding_dim)
        batch_size, seq_len, embedding_dim = x.size()
        
        # Project embeddings to hidden dimension
        x_projected = self.input_projection(x)  # (batch_size, seq_len, hidden)
        
        # Positional encoding
        x_projected = self.pos_encoding(x_projected)
        
        # Transformer encoding
        # If attention mask is provided, convert to the format expected by PyTorch transformer
        if attention_mask is not None:
            # PyTorch transformer expects attention_mask where True means "ignore"
            # Our mask has 1 for "attend" and 0 for "ignore", so we need to invert it
            attention_mask = (attention_mask == 0)
        
        x_encoded = self.transformer(x_projected, src_key_padding_mask=attention_mask)
        
        # Global average pooling over sequence dimension
        # Use attention mask to ignore padding tokens
        if attention_mask is not None:
            # Create mask for valid tokens (invert the attention mask)
            valid_mask = ~attention_mask  # True for valid tokens
            # Set invalid positions to 0
            x_encoded = x_encoded * valid_mask.unsqueeze(-1).float()
            # Sum over sequence dimension
            x_pooled = x_encoded.sum(dim=1)
            # Divide by number of valid tokens per sample
            valid_counts = valid_mask.sum(dim=1, keepdim=True).float()
            x_pooled = x_pooled / valid_counts.clamp(min=1)
        else:
            # Simple average pooling
            x_pooled = x_encoded.mean(dim=1)
        
        # Dropout
        x_pooled = self.dropout(x_pooled)
        
        # Output projection
        return self.output_projection(x_pooled)


def create_model(model_type, in_dim, hidden, out_dim, **kwargs):
    """Model factory function"""
    if model_type.lower() == 'mlp':
        return MLP(in_dim, hidden, out_dim, **kwargs)
    elif model_type.lower() == 'transformer':
        return TransformerModel(in_dim, hidden, out_dim, **kwargs)
    elif model_type.lower() == 'sequential_transformer':
        return SequentialTransformerModel(in_dim, hidden, out_dim, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: 'mlp', 'transformer', 'sequential_transformer'")

