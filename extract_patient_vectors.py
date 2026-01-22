"""
Extract patient vectors (CLS token representations) from trained model
Processes all _processed.json files in the dataset directory
"""
import torch
import torch.nn as nn
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from functools import partial

from model.models import create_model, LTransformerEncoder
from util.data_processing import create_tokenizers


class CLSExtractorWrapper(nn.Module):
    """Wrapper to extract CLS representation from LTransformerEncoder or TransformerEncoder"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Detect model type
        self.is_lorentz = hasattr(model, 'resblocks') and hasattr(model, 'manifold_hidden')
        self.is_euclidean = hasattr(model, 'transformer') and not self.is_lorentz
    
    def extract_cls(self, x_diag, x_proc, x_drug, x_visit_ids=None):
        """
        Extract CLS token representation (patient vector)
        Returns: (batch_size, hidden_dim) tensor
        """
        if self.is_lorentz:
            return self._extract_cls_lorentz(x_diag, x_proc, x_drug, x_visit_ids)
        elif self.is_euclidean:
            return self._extract_cls_euclidean(x_diag, x_proc, x_drug, x_visit_ids)
        else:
            raise ValueError(f"Unknown model type. Model should be LTransformerEncoder or TransformerEncoder")
    
    def _extract_cls_lorentz(self, x_diag, x_proc, x_drug, x_visit_ids=None):
        """Extract CLS from LTransformerEncoder (hyperbolic space)"""
        batch_size, max_len = x_diag.shape
        device = x_diag.device

        cls_token = self.model.cls_token.expand(batch_size, -1, -1)
        token_embeddings = self.model.token_embed(x_diag)   # (batch_size, max_len, width)
        token_embeddings = torch.cat([cls_token, token_embeddings], dim=1)
        
        cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
        padding_mask = (x_diag == 0)  # (batch_size, max_len) - True where padding
        padding_mask = torch.cat([cls_mask, padding_mask], dim=1)  # (batch_size, max_len+1)

        _attn_mask = padding_mask.unsqueeze(1).expand(-1, max_len+1, -1)  # (batch_size, max_len+1, max_len+1)
        
        # Apply transformer blocks
        for block in self.model.resblocks:
            token_embeddings = block(token_embeddings, _attn_mask)
        
        token_embeddings = self.model.final_proj(token_embeddings)
        token_embeddings = self.model.ln_final(token_embeddings)

        # Extract CLS representation (before dropout for consistency)
        cls_state = token_embeddings[:, 0, :]  # (batch_size, hidden_dim)
        
        return cls_state
    
    def _extract_cls_euclidean(self, x_diag, x_proc, x_drug, x_visit_ids=None):
        """Extract CLS from TransformerEncoder (Euclidean space)"""
        batch_size, max_len = x_diag.shape
        device = x_diag.device

        cls_token = self.model.cls_token.expand(batch_size, 1, -1)  # (batch_size, 1, H)
        token_embeddings = self.model.token_embed(x_diag)   # (batch_size, L_diag, E)
        token_embeddings = torch.cat([cls_token, token_embeddings], dim=1)  # (batch_size, L_total+1, H)

        # PyTorch transformer expects src_key_padding_mask=True where positions should be masked (i.e. padding).
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)  # CLS is never padding
        padding_mask = (x_diag == 0)   # (batch_size, L_diag) - True where padding
        padding_mask = torch.cat([cls_mask, padding_mask], dim=1)  # (batch_size, L_diag+1)

        # Apply transformer (without dropout for consistency with Lorentz version)
        token_embeddings = self.model.transformer(token_embeddings, src_key_padding_mask=padding_mask)

        # Extract CLS representation (before dropout for consistency)
        cls_state = token_embeddings[:, 0, :]  # (batch_size, hidden_dim)
        
        return cls_state


def load_checkpoint(checkpoint_path):
    """Load checkpoint and return model, vocabs, and config"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    model_type = checkpoint['model_type']
    vocabs = checkpoint['vocabs']  # (diag_stoi, proc_stoi, drug_stoi)
    arch = checkpoint['arch']
    diag_vocab_size = checkpoint['diag_vocab_size']
    max_diag_len = checkpoint['max_diag_len']
    max_proc_len = checkpoint['max_proc_len']
    max_drug_len = checkpoint['max_drug_len']
    out_dim = len(vocabs[0])  # Output dimension is diag vocab size
    
    print(f"Model type: {model_type}")
    print(f"Architecture: {arch}")
    print(f"Diag vocab size: {diag_vocab_size}")
    print(f"Max lengths - Diag: {max_diag_len}, Proc: {max_proc_len}, Drug: {max_drug_len}")
    
    # Recreate model
    model_kwargs = {
        'diag_size': len(vocabs[0]),
        'proc_size': len(vocabs[1]),
        'diag_itos': None,  # Not needed for inference
        'max_diag_len': max_diag_len,
        'arch': arch
    }
    
    model = create_model(model_type, x_vocab_size=diag_vocab_size, out_dim=out_dim, **model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, vocabs, {
        'max_diag_len': max_diag_len,
        'max_proc_len': max_proc_len,
        'max_drug_len': max_drug_len
    }


def convert_json_entry_to_sample(entry):
    """
    Convert JSON entry to sample format expected by model
    JSON entry has: icd_codes (list), visit_ids (list), patient_id
    Returns: sample dict with cond_hist (list of lists), procedures, drugs
    """
    icd_codes = entry.get('icd_codes', [])
    visit_ids = entry.get('visit_ids', [])
    patient_id = entry.get('patient_id')
    
    # Group codes by visit_id
    visit_dict = defaultdict(list)
    for code, visit_id in zip(icd_codes, visit_ids):
        visit_dict[visit_id].append(code)
    
    # Convert to list of lists, sorted by visit_id
    sorted_visits = sorted(visit_dict.keys())
    cond_hist = [visit_dict[vid] for vid in sorted_visits]
    
    # Create sample format
    sample = {
        'patient_id': patient_id,
        'cond_hist': cond_hist,
        'procedures': [],  # JSON doesn't have procedure codes
        'drugs': [],       # JSON doesn't have drug codes
        'visit_id': entry.get('visit_id', f"patient_{patient_id}"),
    }
    
    return sample


def prepare_input_from_sample(sample, vocabs, max_diag_len, max_proc_len, max_drug_len):
    """
    Convert sample to model input tensors
    Returns: (x_diag, x_proc, x_drug, x_visit_ids) tensors
    """
    diag_stoi, proc_stoi, drug_stoi = vocabs
    diag_tokenizer, proc_tokenizer, drug_tokenizer = create_tokenizers(vocabs)
    
    # Tokenize codes
    cond_hist = sample.get('cond_hist', [])
    x_diag_indices = diag_tokenizer.encode(cond_hist)
    x_proc_indices = proc_tokenizer.encode(sample.get('procedures', []))
    x_drug_indices = drug_tokenizer.encode(sample.get('drugs', []))
    
    # Generate visit_ids for diag codes
    x_visit_ids = []
    for visit_idx, visit_codes in enumerate(cond_hist):
        for c in visit_codes:
            if c in diag_stoi:
                x_visit_ids.append(visit_idx)
    
    # Convert to tensors
    if len(x_diag_indices) == 0:
        x_diag_indices = [0]
        x_visit_ids = [-1]
    
    X_diag = torch.tensor(x_diag_indices, dtype=torch.long)
    X_proc = torch.tensor(x_proc_indices, dtype=torch.long) if len(x_proc_indices) > 0 else torch.tensor([0], dtype=torch.long)
    X_drug = torch.tensor(x_drug_indices, dtype=torch.long) if len(x_drug_indices) > 0 else torch.tensor([0], dtype=torch.long)
    X_visit_ids = torch.tensor(x_visit_ids, dtype=torch.long) if len(x_visit_ids) > 0 else torch.tensor([-1], dtype=torch.long)
    
    # Pad/truncate to max lengths
    if len(X_diag) > max_diag_len:
        X_diag = X_diag[:max_diag_len]
        X_visit_ids = X_visit_ids[:max_diag_len]
    elif len(X_diag) < max_diag_len:
        padding_len = max_diag_len - len(X_diag)
        X_diag = torch.cat([X_diag, torch.zeros(padding_len, dtype=torch.long)])
        X_visit_ids = torch.cat([X_visit_ids, torch.full((padding_len,), -1, dtype=torch.long)])
    
    if len(X_proc) > max_proc_len:
        X_proc = X_proc[:max_proc_len]
    elif len(X_proc) < max_proc_len:
        padding_len = max_proc_len - len(X_proc)
        X_proc = torch.cat([X_proc, torch.zeros(padding_len, dtype=torch.long)])
    
    if len(X_drug) > max_drug_len:
        X_drug = X_drug[:max_drug_len]
    elif len(X_drug) < max_drug_len:
        padding_len = max_drug_len - len(X_drug)
        X_drug = torch.cat([X_drug, torch.zeros(padding_len, dtype=torch.long)])
    
    # Add batch dimension
    X_diag = X_diag.unsqueeze(0)  # (1, max_diag_len)
    X_proc = X_proc.unsqueeze(0)  # (1, max_proc_len)
    X_drug = X_drug.unsqueeze(0)  # (1, max_drug_len)
    X_visit_ids = X_visit_ids.unsqueeze(0)  # (1, max_diag_len)
    
    return X_diag, X_proc, X_drug, X_visit_ids


def process_json_file(json_path, model, vocabs, config, device='cuda', batch_size=32):
    """
    Process a single JSON file and extract patient vectors
    Returns: list of dicts with original entry + patient_vector
    """
    print(f"\nProcessing {json_path}")
    
    with open(json_path, 'r') as f:
        entries = json.load(f)
    
    print(f"Found {len(entries)} entries")
    
    # Wrap model for CLS extraction
    extractor = CLSExtractorWrapper(model).to(device)
    extractor.eval()
    
    results = []
    
    # Process in batches
    for i in range(0, len(entries), batch_size):
        batch_entries = entries[i:i+batch_size]
        batch_samples = [convert_json_entry_to_sample(entry) for entry in batch_entries]
        
        # Prepare batch inputs
        batch_x_diag = []
        batch_x_proc = []
        batch_x_drug = []
        batch_x_visit_ids = []
        
        for sample in batch_samples:
            x_diag, x_proc, x_drug, x_visit_ids = prepare_input_from_sample(
                sample, vocabs, config['max_diag_len'], config['max_proc_len'], config['max_drug_len']
            )
            batch_x_diag.append(x_diag)
            batch_x_proc.append(x_proc)
            batch_x_drug.append(x_drug)
            batch_x_visit_ids.append(x_visit_ids)
        
        # Stack into batch tensors
        batch_x_diag = torch.cat(batch_x_diag, dim=0).to(device)
        batch_x_proc = torch.cat(batch_x_proc, dim=0).to(device)
        batch_x_drug = torch.cat(batch_x_drug, dim=0).to(device)
        batch_x_visit_ids = torch.cat(batch_x_visit_ids, dim=0).to(device)
        
        # Extract CLS representations
        with torch.no_grad():
            patient_vectors = extractor.extract_cls(
                batch_x_diag, batch_x_proc, batch_x_drug, batch_x_visit_ids
            )
        
        # Convert to numpy and store
        patient_vectors_np = patient_vectors.cpu().numpy()
        
        for j, entry in enumerate(batch_entries):
            result = {
                **entry,  # Keep original entry fields
                'patient_vector': patient_vectors_np[j].tolist()  # Convert to list for JSON serialization
            }
            results.append(result)
        
        if (i + batch_size) % 100 == 0 or (i + batch_size) >= len(entries):
            print(f"  Processed {min(i + batch_size, len(entries))}/{len(entries)} entries")
    
    return results


def main():
    # Configuration
    checkpoint_path = Path("checkpoints/ltransformer_encoder_next_20251220_103843.pth")
    dataset_dir = Path("/data/yuyu/data/EHRXQA/ehrxqa/dataset")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    model, vocabs, config = load_checkpoint(checkpoint_path)
    model = model.to(device)
    
    # Find all _processed.json files
    json_files = list(dataset_dir.glob("*_processed.json"))
    print(f"\nFound {len(json_files)} JSON files:")
    for f in json_files:
        print(f"  - {f.name}")
    
    # Process each file
    all_results = {}
    for json_file in json_files:
        results = process_json_file(json_file, model, vocabs, config, device=device)
        
        # Save results
        output_file = output_dir / f"{json_file.stem}_patient_vectors.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} patient vectors to {output_file}")
        
        all_results[json_file.name] = results
    
    # Also save a summary
    summary = {
        'total_files': len(json_files),
        'total_entries': sum(len(r) for r in all_results.values()),
        'vector_dim': len(results[0]['patient_vector']) if results else 0,
        'files_processed': [f.name for f in json_files]
    }
    
    summary_file = output_dir / "extraction_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_file}")
    print(f"Total entries processed: {summary['total_entries']}")
    print(f"Patient vector dimension: {summary['vector_dim']}")


if __name__ == "__main__":
    main()

