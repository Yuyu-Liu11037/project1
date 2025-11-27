import torch
import torch.nn as nn

class TransformerEncoderOnly(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8,
                 num_layers=6, num_classes=10,
                 dim_feedforward=1024, dropout=0.1, max_len=512):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        x: (B, L)
        """
        B, L = x.shape
        device = x.device

        pos = torch.arange(L, device=device).unsqueeze(0)
        x_emb = self.token_emb(x) + self.pos_emb(pos)

        # prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x_emb = torch.cat([cls, x_emb], dim=1)

        enc = self.encoder(x_emb)

        # classification using CLS
        cls_rep = enc[:, 0]       # (B, d_model)
        return self.classifier(cls_rep)
