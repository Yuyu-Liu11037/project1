import torch
import torch.nn as nn

class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8,
                 num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=1024, dropout=0.1, max_len=512):
        super().__init__()

        self.d_model = d_model

        # Token + Positional embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_len, d_model)

        # PyTorch built-in Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Final LM head
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        """
        src: (B, S)
        tgt: (B, T)
        """
        B, S = src.shape
        _, T = tgt.shape
        device = src.device

        src_pos = torch.arange(S, device=device).unsqueeze(0)
        tgt_pos = torch.arange(T, device=device).unsqueeze(0)

        src_emb = self.token_emb(src) + self.pos_emb(src_pos)
        tgt_emb = self.token_emb(tgt) + self.pos_emb(tgt_pos)

        # Causal mask for decoder
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)

        out = self.transformer(
            src_emb, 
            tgt_emb,
            tgt_mask=tgt_mask
        )
        logits = self.lm_head(out)
        return logits
