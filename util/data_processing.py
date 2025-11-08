"""
Data processing module
Contains data preprocessing, vectorization, vocabulary building functions
"""
from collections import defaultdict, Counter
import torch
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from pyhealth.data import Patient
from pyhealth.medcode import CrossMap
from util.hyperbolic_conditions import ConditionsHyperbolicEmbedder
from eval_embedding import load_pkl_file

# Import HyperbolicEntailmentCones for pickle loading
try:
    from hyperbolic_entailment_cones import HyperbolicEntailmentCones
except ImportError:
    # If import fails, we'll handle it dynamically during loading
    HyperbolicEntailmentCones = None


mapping = CrossMap("ICD10CM", "CCSCM")

# Global cache for hyperbolic embeddings
_hyperbolic_embeddings_cache = None
_visit_sep_embedding = None
# Cache for load_pkl_file results (keyed by file path)
_pkl_file_cache = {}


class HyperbolicEntailmentConesAdapter:
    """
    Adapter class to wrap hyperbolic_entailment_cones.py saved data
    and provide the same interface as ConditionsHyperbolicEmbedder
    """
    def __init__(self, save_data):
        """
        Initialize adapter from hyperbolic_entailment_cones.py saved data
        
        Args:
            save_data: Dictionary containing 'model', 'id_map', 'codes', 'dim', etc.
        """
        self.model = save_data['model']
        self.id_map = save_data['id_map']  # code -> id mapping
        self.codes = save_data['codes']  # List of all codes
        self.dim = save_data['dim']
        self.conditions_codes = save_data.get('original_codes', self.codes)
        
        # Build code2embedding dictionary from model.emb and id_map
        # model.emb is a Parameter tensor of shape (num_codes, dim)
        self.code2embedding = {}
        with torch.no_grad():
            emb_tensor = self.model.emb.data.cpu()  # (num_codes, dim)
            # Create reverse mapping: id -> code
            id_to_code = {idx: code for code, idx in self.id_map.items()}
            
            # Build code2embedding dictionary
            for code, code_id in self.id_map.items():
                if code_id < emb_tensor.shape[0]:
                    self.code2embedding[code] = emb_tensor[code_id].clone()
    
    def get_embedding_dim(self) -> int:
        """Get the total embedding dimension for a single condition"""
        return self.dim
    
    def get_embedding_vector(self, conditions_list):
        """
        Get hyperbolic embedding vector for a list of conditions
        
        Args:
            conditions_list: List of condition codes
            
        Returns:
            Fixed-size embedding vector by averaging all condition embeddings
        """
        embeddings = []
        for cond in conditions_list:
            if cond in self.code2embedding:
                embeddings.append(self.code2embedding[cond])
            else:
                # Use zero embedding for unknown codes
                embeddings.append(torch.zeros(self.dim))
        
        if len(embeddings) == 0:
            # Return zero vector if no conditions
            return torch.zeros(self.dim)
        
        # Average all embeddings to get a fixed-size representation
        return torch.stack(embeddings).mean(dim=0)
    
    def get_embedding_sequences(self, conditions_list):
        """
        Get hyperbolic embedding sequences for a list of conditions (for transformer)
        
        Args:
            conditions_list: List of condition codes
            
        Returns:
            Embedding tensor of shape [n, embedding_dim] where n is the number of conditions
        """
        embeddings = []
        for cond in conditions_list:
            if cond in self.code2embedding:
                embeddings.append(self.code2embedding[cond])
            else:
                # Use zero embedding for unknown codes
                embeddings.append(torch.zeros(self.dim))
        
        if len(embeddings) == 0:
            # Return empty tensor with correct shape
            return torch.zeros(0, self.dim)
        
        # Return sequence of embeddings without averaging
        return torch.stack(embeddings)


def load_hyperbolic_embeddings(embedding_file="hyperbolic_embeddings.pkl"):
    """
    Load hyperbolic embeddings from file and cache globally.
    DEPRECATED: This function now uses load_pkl_file internally.
    For new code, use load_pkl_file directly.
    
    Supports two formats:
    1. ConditionsHyperbolicEmbedder instance (from train_hyperbolic_embeddings_icd10.py)
    2. Dictionary format (from hyperbolic_entailment_cones.py)
    
    Args:
        embedding_file: Path to the hyperbolic embeddings pickle file
        
    Returns:
        ConditionsHyperbolicEmbedder instance or HyperbolicEntailmentConesAdapter instance
    """
    global _hyperbolic_embeddings_cache, _visit_sep_embedding
    
    if _hyperbolic_embeddings_cache is None:
        # Use load_pkl_file to load the data
        loaded_data = load_pkl_file(embedding_file)
        
        # Check if it's a dictionary format (from hyperbolic_entailment_cones.py)
        if isinstance(loaded_data, dict) and 'model' in loaded_data and 'id_map' in loaded_data:
            print("Detected hyperbolic_entailment_cones.py format")
            _hyperbolic_embeddings_cache = HyperbolicEntailmentConesAdapter(loaded_data)
            embedding_dim = _hyperbolic_embeddings_cache.get_embedding_dim()
            print(f"Loaded hyperbolic entailment cones embeddings with dimension: {embedding_dim}")
            print(f"Number of codes: {len(_hyperbolic_embeddings_cache.codes)}")
            print(f"Number of condition codes: {len(_hyperbolic_embeddings_cache.conditions_codes)}")
        # Check if it's a ConditionsHyperbolicEmbedder instance
        elif isinstance(loaded_data, ConditionsHyperbolicEmbedder):
            print("Detected ConditionsHyperbolicEmbedder format")
            _hyperbolic_embeddings_cache = loaded_data
            embedding_dim = _hyperbolic_embeddings_cache.get_embedding_dim()
            print(f"Loaded hyperbolic embeddings with dimension: {embedding_dim}")
            print(f"Number of condition codes: {len(_hyperbolic_embeddings_cache.conditions_codes)}")
        else:
            raise ValueError(f"Unknown embedding file format. Expected ConditionsHyperbolicEmbedder or dict with 'model' and 'id_map' keys, got {type(loaded_data)}")
        
        # Create visit separator embedding (same dimension as condition embeddings)
        embedding_dim = _hyperbolic_embeddings_cache.get_embedding_dim()
        # Initialize with small random values (similar to hyperbolic embedding init)
        _visit_sep_embedding = torch.randn(embedding_dim) * 0.01
    
    return _hyperbolic_embeddings_cache


def create_embedding_sequence_with_visit_markers(cond_hist, loaded_data, max_seq_length=200):
    """
    Create embedding sequence from condition history with visit boundary markers
    
    Args:
        cond_hist: List of lists, each inner list contains codes for one visit
                  Last visit is empty to prevent leakage
        loaded_data: Dictionary format (with 'model' and 'id_map') or ConditionsHyperbolicEmbedder instance
        max_seq_length: Maximum sequence length (for padding)
        
    Returns:
        Tuple of (sequence_tensor, attention_mask)
        - sequence_tensor: (seq_len, embedding_dim) padded tensor
        - attention_mask: (seq_len,) tensor with 1 for real tokens, 0 for padding
    """
    global _visit_sep_embedding
    
    if _visit_sep_embedding is None:
        raise ValueError("Visit separator embedding not initialized. Call load_hyperbolic_embeddings() first.")
    
    embedding_dim = get_embedding_dim_from_data(loaded_data)
    sequence_embeddings = []
    
    # Process each visit (except the last empty one)
    for visit_idx, visit_codes in enumerate(cond_hist[:-1]):  # Skip last empty visit
        # Add embeddings for each code in this visit
        for code in visit_codes:
            embedding = get_embedding_from_data(loaded_data, code)
            if embedding is not None:
                sequence_embeddings.append(embedding)
            else:
                # Use zero embedding for unknown codes
                sequence_embeddings.append(torch.zeros(embedding_dim))
        
        # Add visit separator after each visit (except the last one)
        if visit_idx < len(cond_hist) - 2:  # Don't add separator after last visit
            sequence_embeddings.append(_visit_sep_embedding)
    
    if len(sequence_embeddings) == 0:
        # Handle case where no conditions exist
        sequence_embeddings = [torch.zeros(embedding_dim)]
    
    # Convert to tensor
    sequence_tensor = torch.stack(sequence_embeddings)  # (seq_len, embedding_dim)
    
    # Create attention mask (1 for real tokens, 0 for padding)
    seq_len = sequence_tensor.size(0)
    attention_mask = torch.ones(seq_len, dtype=torch.long)
    
    # Pad sequence if needed
    if seq_len < max_seq_length:
        padding_length = max_seq_length - seq_len
        padding = torch.zeros(padding_length, embedding_dim)
        sequence_tensor = torch.cat([sequence_tensor, padding], dim=0)
        
        # Extend attention mask with zeros for padding
        padding_mask = torch.zeros(padding_length, dtype=torch.long)
        attention_mask = torch.cat([attention_mask, padding_mask], dim=0)
    elif seq_len > max_seq_length:
        # Truncate if too long
        sequence_tensor = sequence_tensor[:max_seq_length]
        attention_mask = attention_mask[:max_seq_length]
    
    return sequence_tensor, attention_mask


def diag_prediction_mimic4_fn(patient: Patient):
    """Data processing function for MIMIC-IV diagnosis prediction task"""
    samples = []
    # Sort visits by encounter time to ensure chronological order
    visit_ls = sorted(patient.visits.keys(), key=lambda vid: patient.visits[vid].encounter_time)
    
    for i in range(len(visit_ls)):
        visit = patient.visits[visit_ls[i]]
        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]
        
        cond_ccs = []
        for con in conditions:
            if mapping.map(con):
                cond_ccs.append(mapping.map(con)[0]) 

        if len(cond_ccs) * len(procedures) * len(drugs) == 0:
            continue
            
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": cond_ccs,
                "procedures": procedures,
                "adm_time" : visit.encounter_time.strftime("%Y-%m-%d %H:%M"),
                "drugs": drugs,
                "cond_hist": conditions,
            }
        )
    
    # exclude: patients with less than 2 visits
    if len(samples) < 2:
        return []
    
    # add history
    samples[0]["cond_hist"] = [samples[0]["cond_hist"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs"] = [samples[0]["drugs"]]
    samples[0]["adm_time"] = [samples[0]["adm_time"]]

    for i in range(1, len(samples)):
        samples[i]["drugs"] = samples[i - 1]["drugs"] + [samples[i]["drugs"]]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [samples[i]["procedures"]]
        samples[i]["cond_hist"] = samples[i - 1]["cond_hist"] + [samples[i]["cond_hist"]]
        samples[i]["adm_time"] = samples[i - 1]["adm_time"] + [samples[i]["adm_time"]]

    for i in range(len(samples)):
        samples[i]["cond_hist"][i] = []

    return samples


def sort_samples_within_patient(samples):
    """Group by patient ID and sort by admission time"""
    by_pid = defaultdict(list)
    for s in samples:
        by_pid[s["patient_id"]].append(s)
    
    for pid in by_pid:
        # If adm_time is string, sort directly; if needed, convert to datetime for sorting
        by_pid[pid] = sorted(by_pid[pid], key=lambda x: x["adm_time"][-1])
    
    return by_pid


def build_pairs(samples_by_pid):
    pairs = []
    for pid, seq in samples_by_pid.items():
        for i in range(1, len(seq)):
                pairs.append((seq[i], seq[i]["conditions"]))
    return pairs


def build_vocab_from_pairs(pairs):
    """Build vocabulary from training pairs"""
    diag_c, proc_c, drug_c, y_c = Counter(), Counter(), Counter(), Counter()
    
    for s, y in pairs:
        for visit_codes in s["cond_hist"]:   # Historical ICD diagnoses (last step is empty to prevent leakage)
            diag_c.update(visit_codes)
        for visit_codes in s["procedures"]:  # Each step is a procedure code list
            proc_c.update(visit_codes)
        for visit_codes in s["drugs"]:       # Each step is an ATC3 list
            drug_c.update(visit_codes)
        y_c.update(y)                        # Labels (CCS)
    
    def mk_vocab(cnt):
        itos = [c for c, _ in cnt.most_common()]
        stoi = {c:i for i,c in enumerate(itos)}
        return stoi, itos
    
    return mk_vocab(diag_c), mk_vocab(proc_c), mk_vocab(drug_c), mk_vocab(y_c)


def multihot_from_sequence(seq_of_lists, stoi):
    """Convert sequence to multi-hot vector"""
    x = torch.zeros(len(stoi), dtype=torch.float32)
    for codes in seq_of_lists:
        for c in codes:
            if c in stoi: 
                x[stoi[c]] = 1.0
    return x


def split_by_patient(pairs, test_size=0.2, val_size=0.1, seed=42):
    """Split dataset by patient ID to avoid data leakage"""
    pid2pairs = defaultdict(list)
    for s, y in pairs:
        pid2pairs[s["patient_id"]].append((s, y))
    
    pids = list(pid2pairs.keys())
    tr_pids, te_pids = train_test_split(pids, test_size=test_size, random_state=seed)
    tr_pids, va_pids = train_test_split(tr_pids, test_size=val_size, random_state=seed)
    
    def collect(pid_list):
        out = []
        for pid in pid_list: 
            out.extend(pid2pairs[pid])
        return out
    
    return collect(tr_pids), collect(va_pids), collect(te_pids)
