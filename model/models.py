"""
模型定义模块
包含MLP模型的定义
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """简单的多标签MLP模型"""
    
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

