"""
Data processing module
Contains data preprocessing, vectorization, vocabulary building functions
"""
from collections import defaultdict, Counter
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from pyhealth.data import Patient
from pyhealth.medcode import CrossMap


mapping = CrossMap("ICD10CM", "CCSCM")


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


def build_pairs(samples_by_pid, task="current"):
    """
    Build training pairs
    task="current": Use sample's own conditions as labels
    task="next":    Strictly follow paper, features from time t, labels from time t+1 conditions
    Returns pairs: list of (X_sample_dict, y_codes_list)
    """
    pairs = []
    for pid, seq in samples_by_pid.items():
        if task == "current":
            for s in seq:
                pairs.append((s, s["conditions"]))
        elif task == "next":
            # Must have at least t and t+1
            for i in range(len(seq) - 1):
                s_t = seq[i]
                y_next = seq[i + 1]["conditions"]
                pairs.append((s_t, y_next))
        else:
            raise ValueError("task must be 'current' or 'next'")
    return pairs


def build_vocab_from_pairs(pairs):
    """Build vocabulary from training pairs
    
    Note: Y (labels) and diagnosis share the same vocab. 
    Labels (CCS codes) are merged into diagnosis vocab.
    """
    diag_c, proc_c, drug_c = Counter(), Counter(), Counter()
    
    for s, y in pairs:
        for visit_codes in s["cond_hist"]:   # Historical ICD diagnoses (last step is empty to prevent leakage)
            diag_c.update(visit_codes)
        for visit_codes in s["procedures"]:  # Each step is a procedure code list
            proc_c.update(visit_codes)
        for visit_codes in s["drugs"]:       # Each step is an ATC3 list
            drug_c.update(visit_codes)
        # Merge labels into diagnosis vocab since they share the same tokenizer
        diag_c.update(y)                     # Labels (CCS) added to diagnosis vocab
    
    def mk_vocab(cnt):
        itos = [c for c, _ in cnt.most_common()]
        stoi = {c:i for i,c in enumerate(itos)}
        return stoi, itos
    
    # Y vocab is now the same as diag vocab
    diag_stoi, diag_itos = mk_vocab(diag_c)
    proc_stoi, proc_itos = mk_vocab(proc_c)
    drug_stoi, drug_itos = mk_vocab(drug_c)
    
    # Return diag_vocab as both diag_stoi and y_stoi since they share the same tokenizer
    return (diag_stoi, diag_itos), (proc_stoi, proc_itos), (drug_stoi, drug_itos), (diag_stoi, diag_itos)


class CodeTokenizer:
    """Tokenizer for medical codes (diagnosis, procedure, drug)"""
    
    def __init__(self, stoi, offset=0):
        """
        Args:
            stoi: String to index mapping dictionary
            offset: Starting index for this tokenizer (to reserve 0 for padding)
        """
        self.stoi = stoi
        self.itos = {i: code for code, i in stoi.items()}  # Index to string mapping
        self.offset = offset
        self.vocab_size = len(stoi)
    
    def encode(self, seq_of_lists):
        """
        Convert code sequences to indices
        
        Args:
            seq_of_lists: List of lists of codes (e.g., [[code1, code2], [code3]])
        
        Returns:
            List of indices
        """
        indices = []
        for codes in seq_of_lists:
            for c in codes:
                if c in self.stoi:
                    indices.append(self.stoi[c] + self.offset)
        return indices
    
    def decode(self, indices, remove_padding=True):
        """
        Convert indices back to code strings
        
        Args:
            indices: List of indices
            remove_padding: Whether to remove 0 (padding) indices
        
        Returns:
            List of code strings
        """
        codes = []
        for idx in indices:
            if remove_padding and idx == 0:
                continue
            idx_without_offset = idx - self.offset
            if idx_without_offset in self.itos:
                codes.append(self.itos[idx_without_offset])
        return codes


def indices_from_sequence(seq_of_lists, stoi, offset=0):
    """Convert sequence to list of indices (with optional offset to reserve 0 for padding)
    
    DEPRECATED: Use CodeTokenizer.encode() instead
    """
    indices = []
    for codes in seq_of_lists:
        for c in codes:
            if c in stoi: 
                indices.append(stoi[c] + offset)  # Add offset to reserve 0 for padding
    return indices


def create_tokenizers(vocabs):
    """
    Create tokenizers for diagnosis, procedure, drug codes, and labels
    
    Args:
        vocabs: Tuple of (diag_stoi, proc_stoi, drug_stoi, y_stoi)
        Note: y_stoi is the same as diag_stoi (they share the same vocab)
    
    Returns:
        Tuple of (diag_tokenizer, proc_tokenizer, drug_tokenizer, y_tokenizer)
    """
    diag_stoi, proc_stoi, drug_stoi, y_stoi = vocabs
    
    # Reserve 0 for padding, so start at 1
    # Y (labels) shares the same tokenizer as diagnosis since they use the same vocab
    diag_tokenizer = CodeTokenizer(diag_stoi, offset=1)
    y_tokenizer = diag_tokenizer  # Same tokenizer for Y and diagnosis
    
    # Procedure codes come after diagnosis codes
    proc_tokenizer = CodeTokenizer(proc_stoi, offset=len(diag_stoi) + 1)
    
    # Drug codes come after procedure codes
    drug_tokenizer = CodeTokenizer(drug_stoi, offset=len(diag_stoi) + len(proc_stoi) + 1)
    
    return diag_tokenizer, proc_tokenizer, drug_tokenizer, y_tokenizer


def vectorize_pair(s, y_codes, vocabs, use_current_step=False):
    """Vectorize sample pair - returns indices instead of multi-hot vectors"""
    diag_stoi, proc_stoi, drug_stoi, y_stoi = vocabs
    
    # Admission prediction: don't look at current step's proc/drug; discharge prediction can look
    if use_current_step:
        proc_hist = s["procedures"]
        drug_hist = s["drugs"]
    else:
        proc_hist = s["procedures"][:-1] if len(s["procedures"])>0 else []
        drug_hist = s["drugs"][:-1] if len(s["drugs"])>0 else []

    # Create tokenizers (Y shares the same tokenizer as diagnosis)
    diag_tokenizer, proc_tokenizer, drug_tokenizer, y_tokenizer = create_tokenizers(vocabs)
    
    # Tokenize each type of code
    x_diag_indices = diag_tokenizer.encode(s["cond_hist"])
    x_proc_indices = proc_tokenizer.encode(proc_hist)
    x_drug_indices = drug_tokenizer.encode(drug_hist)
    
    # Flatten all indices into a single list
    X_indices = x_diag_indices + x_proc_indices + x_drug_indices
    X = torch.tensor(X_indices, dtype=torch.long) if len(X_indices) > 0 else torch.tensor([0], dtype=torch.long)

    # For Y, use the shared diag_tokenizer (they use the same vocab)
    y_indices = y_tokenizer.encode([y_codes])
    y = torch.tensor(y_indices, dtype=torch.long) if len(y_indices) > 0 else torch.tensor([0], dtype=torch.long)
    
    return X, y


def prepare_XY(pairs, vocabs, use_current_step=False):
    """Prepare training data X and Y as variable-length sequences"""
    Xs, Ys = [], []
    for s, y_codes in pairs:
        X, y = vectorize_pair(s, y_codes, vocabs, use_current_step=use_current_step)
        Xs.append(X)
        Ys.append(y)
    return Xs, Ys


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

