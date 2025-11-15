import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

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
        self.logit_c = nn.Parameter(torch.tensor(float(c_init)).log())  # learnable curvature
        self.norm = nn.LayerNorm(out_dim)

    @property
    def c(self):
        return self.logit_c.exp()

    def forward(self, x_h):  # x_h in Poincaré ball
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
                    # print(code_str)
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
            x = self.adapter(x)
        else:
            x = self.embedding_layer(x)  # (batch_size, seq_len, hidden)
        # x = self.pos_encoding(x)
        x = self.transformer(x)
        x_output = x[:, 0, :]  # (batch_size, hidden)
        x_output = self.dropout(x_output)
        output = self.output_projection(x_output)  # (batch_size, out_dim)
        
        return output


class LinearSVMModel(nn.Module):
    def __init__(self, vocab_size, hidden, out_dim, save_data=None, diag_itos=None, C=1.0, **kwargs):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.out_dim = out_dim
        self.save_data = save_data
        self.diag_itos = diag_itos
        self.C = C
        
        # Initialize SVM model (will be fit during training)
        self.svm_model = OneVsRestClassifier(LinearSVC(C=C, max_iter=1000, random_state=42))
        self.is_fitted = False
    
    def _extract_features(self, x):
        """
        Extract features from token sequences using one-hot encoding and average pooling.
        Processes on CPU to avoid GPU memory issues with large vocabularies.
        
        Args:
            x: (num_samples, seq_len) tensor of token IDs
            
        Returns:
            features: (num_samples, vocab_size) tensor of feature vectors (frequency vectors)
        """
        num_samples, seq_len = x.size()
        
        # Move to CPU for one-hot encoding to avoid GPU OOM
        x_cpu = x.cpu()
        x_flat = x_cpu.view(-1)  # (num_samples * seq_len,)
        x_clamped = torch.clamp(x_flat, 0, self.vocab_size - 1)
        
        # Create one-hot encoding on CPU
        one_hot = torch.zeros(num_samples * seq_len, self.vocab_size, device='cpu', dtype=torch.float32)
        valid_mask = x_flat > 0
        if valid_mask.any():
            valid_indices = torch.arange(num_samples * seq_len, device='cpu')[valid_mask]
            valid_token_ids = x_clamped[valid_mask]
            one_hot[valid_indices, valid_token_ids] = 1.0
        
        embeddings = one_hot.view(num_samples, seq_len, self.vocab_size)
        
        # Average pooling over sequence length (mask out padding tokens)
        mask = (x_cpu > 0).float().unsqueeze(-1)  # (num_samples, seq_len, 1)
        masked_embeddings = embeddings * mask
        feature_sum = masked_embeddings.sum(dim=1)  # (num_samples, vocab_size)
        seq_lengths = mask.sum(dim=1).clamp_min(1.0)  # (num_samples, 1)
        features = feature_sum / seq_lengths  # (num_samples, vocab_size)
        
        return features
    
    def fit(self, X, y, batch_size=1000):
        """
        Fit the SVM model with batch processing to avoid memory issues
        
        Args:
            X: (num_samples, max_seq_len) tensor of token IDs
            y: (num_samples, len_y_stoi) tensor of multi-hot labels
            batch_size: Batch size for feature extraction (default: 1000)
        """
        num_samples = X.size(0)
        all_features = []
        
        # Process in batches to avoid OOM
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                X_batch = X[i:end_idx]
                
                # Extract features for this batch
                features_batch = self._extract_features(X_batch)
                all_features.append(features_batch)
        
        # Concatenate all features
        features = torch.cat(all_features, dim=0)  # (num_samples, vocab_size)
        features_np = features.numpy()  # Already on CPU
        labels_np = y.cpu().numpy()
        
        # Fit SVM
        self.svm_model.fit(features_np, labels_np)
        self.is_fitted = True
    
    def forward(self, x):
        """
        Forward pass - returns logits (decision function values)
        
        Args:
            x: (num_samples, seq_len) tensor of token IDs
            
        Returns:
            logits: (num_samples, len_y_stoi) tensor of logits
        """
        if not self.is_fitted:
            # If not fitted, return zeros (for compatibility during model initialization)
            num_samples = x.size(0)
            logits = torch.zeros(num_samples, self.out_dim, device=x.device)
            return logits
        
        # Extract features (already on CPU)
        features = self._extract_features(x)
        features_np = features.numpy()
        
        # Get logits from SVM
        logits_np = self.svm_model.decision_function(features_np)  # (num_samples, out_dim)
        logits = torch.from_numpy(logits_np).float().to(x.device)
        
        return logits
    
    def parameters(self):
        """
        Return empty iterator for compatibility with PyTorch optimizers
        Note: SVM doesn't have trainable parameters (no embedding layer, embeddings are fixed)
        """
        return iter([])
  
    def to(self, device):
        """
        Move model to device (for compatibility)
        Note: SVM model itself stays on CPU, but embeddings are moved
        """
        return super().to(device)


def create_model(model_type, vocab_size, hidden, out_dim, **kwargs):
    if model_type == "svm":
        return LinearSVMModel(vocab_size, hidden, out_dim, **kwargs)
    else:
        return TransformerModel(vocab_size, hidden, out_dim, **kwargs)
