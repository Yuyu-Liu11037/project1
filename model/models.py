"""
Model definition module
Contains MLP and Transformer model definitions
"""
import torch
import torch.nn as nn
import math


class EmbeddingCombinerMLP(nn.Module):
    """MLP to combine multiple embeddings into a single embedding vector
    
    Supports variable-length sequences by processing each embedding and then aggregating.
    """
    
    def __init__(self, embedding_dim, hidden_dim=None, p=0.3):
        """
        Args:
            embedding_dim: Dimension of each embedding vector
            hidden_dim: Hidden dimension for the MLP (default: embedding_dim * 2)
            p: Dropout probability
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        if hidden_dim is None:
            hidden_dim = embedding_dim * 2
        
        # Process each embedding through shared MLP
        self.per_embedding_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Final aggregation layer
        self.aggregation = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, x):
        """
        Args:
            x: List of tensors, each of shape (num_codes_i, embedding_dim) where num_codes_i can vary
        
        Returns:
            (batch_size, embedding_dim)
        """
        if isinstance(x, list):
            # Handle variable-length sequences in a batch
            batch_size = len(x)
            batch_outputs = []
            
            for i, patient_embeddings in enumerate(x):
                # patient_embeddings: (num_codes_i, embedding_dim)
                # Process each embedding through shared MLP
                processed = self.per_embedding_mlp(patient_embeddings)  # (num_codes_i, embedding_dim)
                # Aggregate: mean pooling
                aggregated = processed.mean(dim=0)  # (embedding_dim,)
                # Final transformation
                output = self.aggregation(aggregated)  # (embedding_dim,)
                batch_outputs.append(output)
            
            # Stack to (batch_size, embedding_dim)
            return torch.stack(batch_outputs, dim=0)
        else:
            # Handle fixed-length sequences: (batch_size, num_codes, embedding_dim)
            batch_size, num_codes, embedding_dim = x.shape
            
            # Process each embedding through shared MLP
            # Reshape to process all embeddings at once: (batch_size * num_codes, embedding_dim)
            x_flat = x.view(batch_size * num_codes, embedding_dim)
            x_processed = self.per_embedding_mlp(x_flat)
            # Reshape back: (batch_size, num_codes, embedding_dim)
            x_processed = x_processed.view(batch_size, num_codes, embedding_dim)
            
            # Aggregate: mean pooling followed by linear transformation
            x_aggregated = x_processed.mean(dim=1)  # (batch_size, embedding_dim)
            x_output = self.aggregation(x_aggregated)  # (batch_size, embedding_dim)
            
            return x_output


class TransformerModel(nn.Module):
    """Transformer-based multi-label classification model"""
    
    def __init__(self, in_dim, hidden, out_dim, num_heads=8, num_layers=3, p=0.3, embedding_combiner=None):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.out_dim = out_dim
        self.embedding_combiner = embedding_combiner
        
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
        # If embedding_combiner is provided, apply it first
        # x would be (batch_size, num_codes, embedding_dim) in this case
        if self.embedding_combiner is not None:
            x = self.embedding_combiner(x)  # (batch_size, embedding_dim)
        
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


def create_model(model_type, in_dim, hidden, out_dim, **kwargs):
    return TransformerModel(in_dim, hidden, out_dim, **kwargs)


def create_model_from_data(model_type, X_sample, hidden, out_dim, use_hyperbolic_embeddings=False, **kwargs):
    """
    Create model with automatic input dimension detection
    
    Args:
        model_type: Type of model ('mlp', 'transformer')
        X_sample: Sample input tensor to determine input dimensions
        hidden: Hidden layer dimension
        out_dim: Output dimension
        use_hyperbolic_embeddings: If True and input is 3D, create embedding combiner
        **kwargs: Additional model parameters
        
    Returns:
        Model instance
    """
    if len(X_sample.shape) == 3:
        # Input is 3D: (batch_size, max_num_codes, embedding_dim)
        # This happens when use_hyperbolic_embeddings=True
        embedding_dim = X_sample.shape[2]
        
        if use_hyperbolic_embeddings:
            # Extract parameters for EmbeddingCombinerMLP
            # EmbeddingCombinerMLP only accepts: embedding_dim, hidden_dim, p
            combiner_kwargs = {}
            if 'hidden_dim' in kwargs:
                combiner_kwargs['hidden_dim'] = kwargs.pop('hidden_dim')
            if 'p' in kwargs:
                combiner_kwargs['p'] = kwargs.pop('p')
            elif 'dropout' in kwargs:
                combiner_kwargs['p'] = kwargs.pop('dropout')
            
            # Create embedding combiner MLP with only relevant parameters
            embedding_combiner = EmbeddingCombinerMLP(embedding_dim, **combiner_kwargs)
            # Create base model with embedding_dim as in_dim (after combiner)
            # Remaining kwargs (num_heads, num_layers, etc.) go to base model
            model = create_model(model_type, embedding_dim, hidden, out_dim, **kwargs)
            # Attach combiner to the model
            model.embedding_combiner = embedding_combiner
            print(f"Created {model_type} model with EmbeddingCombinerMLP for hyperbolic embeddings")
            return model
        else:
            raise ValueError(f"3D input detected but use_hyperbolic_embeddings=False. Expected 2D input for regular models.")
    elif len(X_sample.shape) == 2:
        # Flat input: (batch_size, feature_dim)
        in_dim = X_sample.shape[1]
        return create_model(model_type, in_dim, hidden, out_dim, **kwargs)
    else:
        raise ValueError(f"Unsupported input tensor shape: {X_sample.shape}")

