import torch
import torch.nn as nn

class TransformerDecoderOnly(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8,
                 num_layers=6, dim_feedforward=1024,
                 dropout=0.1, max_len=1024):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_len, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        x: (B, L)
        """
        B, L = x.shape
        device = x.device

        pos = torch.arange(L, device=device).unsqueeze(0)
        x_emb = self.token_emb(x) + self.pos_emb(pos)

        # GPT causal mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(L).to(device)

        # Decoder-only: use itself as both memory and target
        out = self.decoder(
            x_emb, 
            x_emb, 
            tgt_mask=tgt_mask
        )
        logits = self.lm_head(out)
        return logits
