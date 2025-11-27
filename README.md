## An example of training data (X) and label (Y)
X (all codes in history visits of a patient):


    [
        ['I25110', 'E1110', 'E1122', 'E11319', 'M869', 'I5032', 'I130', 'T82855A', 'E11621', 'E1142', 'E1165', 'E1169', 'L97529', 'N183', 'J449', 'B961', 'B951', 'F329', 'K219', 'G4733', 'Z87891', 'Z951', 'Z794', 'Z9114', 'G2581', 'Z955', 'Y840', 'Y929'], 
        ['A419', 'M86672', 'M86172', 'N179', 'L03116', 'J9811', 'I5032', 'I130', 'E1152', 'I96', 'E1169', 'L97524', 'E11621', 'B9561', 'J449', 'E11319', 'E1142', 'E1165', 'I2510', 'I252', 'G4733', 'K219', 'G2581', 'N189', 'E1122', 'Z794', 'Z951', 'Z955', 'Z87891'], 
        ['E1152', 'M869', 'I70268', 'L97528', 'Z1639', 'N179', 'E11621', 'E1169', 'I129', 'E1122', 'I25118', 'N183', 'G2581', 'K219', 'J449', 'G4700', 'B9562', 'Z951'], 
        []
    ]

Y (classes of next-visit codes):

    ['50', '55', '201', '199', '248', '50', '114', '50', '50', '127', '101', '114', '50', '155', '99', '158', '2617', '50', '50', '657', '81', '138', '95', '155', '53', '59', '238', '2617', '2621', '257', '101', '663', '212', '212']

I concatenated all history ICD-10 codes in one sequence, then mapped them to integer tokens. Labels are originally constituted by ICD-10 codes in the next visit; they are mapped to some clinical classes through a mapping called "CCS". In my data there are 275 possible distinct labels.

The feed-forward step is line 184 in training/training.py:
```logits = model(batch_X_diag, batch_X_proc, batch_X_drug)```, but ```batch_X_proc, batch_X_drug``` are not used in the current procedure.

## Models
I use an Encoder-only model for this multi-class classification task. The Euclidean Transformer model and Lorentz Transformer model share the same inputs and outputs, in model/models.py:
```    
    # Euclidean Transformer
    def forward(self, x_diag, x_proc, x_drug):
        batch_size, max_len = x_diag.shape
        device = x_diag.device

        cls_token = self.cls_token.expand(batch_size, 1, -1)        # (batch_size, 1, H)
        token_embeddings = self.token_embed(x_diag)   # (batch_size, L_diag, E)
        token_embeddings = torch.cat([cls_token, token_embeddings], dim=1) # (batch_size, L_total+1, H)

        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        padding_mask = (x_diag != 0)   # (batch_size, L_diag)
        padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        token_embeddings = self.transformer(token_embeddings, src_key_padding_mask=~padding_mask)

        cls_state = self.dropout(token_embeddings[:, 0, :])  # (batch_size, H)
        logits = self.classifier(cls_state)  # (batch_size, out_dim)
        return logits
```
```
    # Lorentz Transformer
    def forward(self, x_diag, x_proc, x_drug, attn_mask = None):
        batch_size, max_len = x_diag.shape
        device = x_diag.device

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        token_embeddings = self.token_embed(x_diag)   # (batch_size, max_len, width)
        token_embeddings = torch.cat([cls_token, token_embeddings], dim=1)
        
        cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
        padding_mask = (x_diag == 0)  # (batch_size, max_len) - True where padding
        padding_mask = torch.cat([cls_mask, padding_mask], dim=1)  # (batch_size, max_len+1)

        _attn_mask = padding_mask.unsqueeze(1).expand(-1, max_len+1, -1)  # (batch_size, max_len+1, max_len+1)
        # Each block applies: Lorentz normalization -> bidirectional self-attention -> residual connection
        #                    -> Lorentz normalization -> feed-forward network -> residual connection
        for block in self.resblocks:
            token_embeddings = block(token_embeddings, _attn_mask)
        token_embeddings = self.final_proj(token_embeddings)
        token_embeddings = self.ln_final(token_embeddings)

        cls_state = self.dropout(token_embeddings[:, 0, :])
        logits = self.classifier(cls_state)
        return logits
```
They are both optimized by ```opt = geoopt.optim.RiemannianAdam(model.parameters(), lr=lr, weight_decay=wd)```.

## Results
My results are recorded in results.md. You can see that Transformer encoder performs slightly better than Lorentz Transformer.