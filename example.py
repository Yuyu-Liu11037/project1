import torch

from model.models import LTransformerDecoder


def run_ltransformerdecoder_example():
    """
    Minimal example showing how to construct and run `LTransformerDecoder`.

    The model expects three inputs:
      - x_diag: LongTensor of shape (batch_size, seq_len)
      - x_proc: LongTensor of shape (batch_size, seq_len)  (currently unused but required)
      - x_drug: LongTensor of shape (batch_size, seq_len)  (currently unused but required)

    Each entry in x_diag is an integer token id in [0, vocab_size-1], where 0 is reserved for padding.
    """
    # Hyperparameters for the example
    batch_size = 1
    context_length = 5  # max sequence length
    vocab_size = 10      # size of diagnosis vocabulary (including padding id 0)
    out_dim = 6         # number of output labels / logits per sample

    # Create a random batch of diagnosis token ids in [1, vocab_size-1]
    # 0 is reserved for padding. Here we explicitly set the last 2 positions to 0
    # to mimic padding tokens.
    x_diag = torch.randint(
        low=1,
        high=vocab_size,
        size=(batch_size, context_length),
        dtype=torch.long,
    )
    x_diag[:, -2:] = 0  # last two positions are padding

    # Procedure and drug sequences are required by the forward signature,
    # but are not used inside `LTransformerDecoder.forward` at the moment.
    # We simply pass zero tensors with the same shape.
    x_proc = torch.zeros_like(x_diag)
    x_drug = torch.zeros_like(x_diag)

    # Instantiate the decoder-only Lorentz transformer
    model = LTransformerDecoder(
        vocab_size=vocab_size,
        context_length=context_length,
        out_dim=out_dim,
    )

    # Run a forward pass
    logits = model(x_diag, x_proc, x_drug)  # shape: (batch_size, out_dim)

    print("x_diag shape:", x_diag.shape)
    print("logits shape:", logits.shape)
    print("logits (first sample):")
    print(logits[0])


if __name__ == "__main__":
    run_ltransformerdecoder_example()
