import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math


class LorentzLinear(nn.Module):
    def __init__(self, manifold, in_features, out_features, bias=True, manifold_out=None, num_heads=1):
        super().__init__()
        self.manifold = manifold
        self.manifold_out = manifold_out
        self.c = manifold.c
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()
        self.num_heads = num_heads

    def reset_parameters(self):
        init.xavier_uniform_(self.linear.weight, gain=math.sqrt(2))
        init.constant_(self.linear.bias, 0)

    def forward(self, x, return_space=False):
        x_space = self.linear(x[..., 1:])   # TODO: changed from x_space = self.linear(x)
        if self.num_heads > 1:
            dim_per_head = self.out_features // self.num_heads
            x_space = x_space.reshape(x_space.size(0), x_space.size(1), self.num_heads, dim_per_head)
        if return_space:
            x = x_space
        else:
            x_time = ((x_space**2).sum(dim=-1, keepdims=True) + self.c).clamp_min(1e-8).sqrt()
            x = torch.cat([x_time, x_space], dim=-1)
        return x
