import torch
import torch.nn as nn
from ...manifolds import Lorentz
from geoopt import ManifoldParameter

class LorentzEmbeddings(nn.Module):
    def __init__(self, manifold_in, num_embeddings, embedding_dim, padding_idx=0, manifold_out=None):
        super().__init__()
        self.manifold_in = manifold_in
        self.manifold_out = manifold_out
        self.padding_idx = padding_idx
        init = manifold_in.random_normal((num_embeddings, embedding_dim))
        init[padding_idx] = manifold_in.origin(embedding_dim)
        self.embedding = ManifoldParameter(init, manifold=manifold_in)
        self.embedding.requires_grad_(True)
        self.embedding.data[padding_idx].requires_grad = False

    def forward(self, input_tokens):
        assert input_tokens.max() < self.embedding.size(0), "Token index cannot exceed vocab size"
        input_tokens = input_tokens.permute(1, 0).contiguous()
        emb = self.embedding.index_select(0, input_tokens.view(-1, )).view(input_tokens.shape + (-1, ))
        if self.padding_idx is not None:
            mask = (input_tokens == self.padding_idx).unsqueeze(-1)  # (L, B, 1)
            pad_emb = self.embedding[self.padding_idx].detach()  # (D,)
            pad_emb = pad_emb.view(1, 1, -1)
            emb = torch.where(mask, pad_emb, emb)
            emb = emb * (self.manifold_out.c / self.manifold_in.c).sqrt()
        return emb.permute(1, 0, 2)
        