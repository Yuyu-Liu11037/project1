"""
Data processing module
Contains data preprocessing, vectorization, vocabulary building functions
"""
from collections import defaultdict, Counter
import torch
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from pyhealth.data import Patient
from pyhealth.medcode import CrossMap
from pathlib import Path


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
        
        # cond_ccs = []
        # for con in conditions:
        #     if mapping.map(con):
        #         cond_ccs.append(mapping.map(con)[0]) 

        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
            
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
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
    diag_c, proc_c, drug_c= Counter(), Counter(), Counter()
    with open('/data/yuyu/project1/cond_hist_codes.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            codes = line.split()
            diag_c.update(codes)
    
    for s, y in pairs:
        for visit_codes in s["procedures"]:  # Each step is a procedure code list
            proc_c.update(visit_codes)
        for visit_codes in s["drugs"]:       # Each step is an ATC3 list
            drug_c.update(visit_codes)
    
    def mk_vocab(cnt):
        itos = [c for c, _ in cnt.most_common()]
        stoi = {c:i for i,c in enumerate(itos)}
        return stoi, itos
    
    # Separate vocabularies: ICD for input, CCS for output
    diag_stoi, diag_itos = mk_vocab(diag_c)  # ICD codes for cond_hist
    proc_stoi, proc_itos = mk_vocab(proc_c)
    drug_stoi, drug_itos = mk_vocab(drug_c)
    
    return (diag_stoi, diag_itos), (proc_stoi, proc_itos), (drug_stoi, drug_itos)


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
                if c not in self.stoi:
                    # print(c)
                    continue
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
    diag_stoi, proc_stoi, drug_stoi= vocabs
    
    # Reserve 0 for padding, so start at 1
    # With separate embeddings, each type uses local indexing (0-indexed in its own vocab)
    diag_tokenizer = CodeTokenizer(diag_stoi, offset=1) 
    
    # Each code type now uses local indexing
    proc_tokenizer = CodeTokenizer(proc_stoi, offset=1)
    drug_tokenizer = CodeTokenizer(drug_stoi, offset=1)
    
    return diag_tokenizer, proc_tokenizer, drug_tokenizer


def vectorize_pair(s, y_codes, vocabs, use_current_step=False):
    diag_stoi, proc_stoi, drug_stoi= vocabs
    
    # Admission prediction: don't look at current step's proc/drug; discharge prediction can look
    cond_hist = s["cond_hist"]
    if use_current_step:
        proc_hist = s["procedures"]
        drug_hist = s["drugs"]
    else:
        proc_hist = s["procedures"][:-1] if len(s["procedures"])>0 else []
        drug_hist = s["drugs"][:-1] if len(s["drugs"])>0 else []

    diag_tokenizer, proc_tokenizer, drug_tokenizer = create_tokenizers(vocabs)
    
    # Tokenize each type of code
    x_diag_indices = diag_tokenizer.encode(cond_hist) 
    x_proc_indices = proc_tokenizer.encode(proc_hist)
    x_drug_indices = drug_tokenizer.encode(drug_hist)
    
    # Return three separate tensors (use 0 as padding)
    X_diag = torch.tensor(x_diag_indices, dtype=torch.long) if len(x_diag_indices) > 0 else torch.tensor([0], dtype=torch.long)
    X_proc = torch.tensor(x_proc_indices, dtype=torch.long) if len(x_proc_indices) > 0 else torch.tensor([0], dtype=torch.long)
    X_drug = torch.tensor(x_drug_indices, dtype=torch.long) if len(x_drug_indices) > 0 else torch.tensor([0], dtype=torch.long)

    y_indices = diag_tokenizer.encode([y_codes])
    y_multi_hot = torch.zeros(len(diag_stoi), dtype=torch.float)
    valid_indices = [idx - 1 for idx in y_indices if idx > 0]
    y_multi_hot[valid_indices] = 1.0
    y = y_multi_hot
    
    return (X_diag, X_proc, X_drug), y


def prepare_XY(pairs, vocabs, use_current_step=False):
    """Prepare training data X and Y as variable-length sequences"""
    Xs_diag, Xs_proc, Xs_drug, Ys = [], [], [], []
    for s, y_codes in pairs:
        (X_diag, X_proc, X_drug), y = vectorize_pair(s, y_codes, vocabs, use_current_step=use_current_step)
        Xs_diag.append(X_diag)
        Xs_proc.append(X_proc)
        Xs_drug.append(X_drug)
        Ys.append(y)
    return (Xs_diag, Xs_proc, Xs_drug), Ys


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


def load_preprocessed_data(data_path):
    """
    Load preprocessed data from the specified directory and convert to samples format
    
    Args:
        data_path: Path to the preprocessed data directory
        
    Returns:
        List of samples in the format expected by train_model_on_samples
    """
    data_path = Path(data_path)
    
    # Load the main CSV file with patient data
    csv_file = data_path / "full_preprocessing_sample_977.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"Required file not found: {csv_file}")
    
    print(f"Loading data from {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Load mapping files
    with open(data_path / "diag_code_to_idx.json", 'r') as f:
        diag_code_to_idx = json.load(f)
    with open(data_path / "med_code_to_idx.json", 'r') as f:
        med_code_to_idx = json.load(f)
    with open(data_path / "proc_code_to_idx.json", 'r') as f:
        proc_code_to_idx = json.load(f)
    
    samples = []
    
    for _, row in df.iterrows():
        patient_id = row['patient_id']
        
        # Parse diagnosis codes
        diag_codes = []
        if pd.notna(row['diagnosis_codes']) and row['diagnosis_codes']:
            diag_codes = [code.strip() for code in str(row['diagnosis_codes']).split(';')]
        
        # Parse medication codes  
        med_codes = []
        if pd.notna(row['medication_codes']) and row['medication_codes']:
            med_codes = [code.strip() for code in str(row['medication_codes']).split(';')]
        
        # Parse procedure codes
        proc_codes = []
        if pd.notna(row['procedure_codes']) and row['procedure_codes']:
            proc_codes = [code.strip() for code in str(row['procedure_codes']).split(';')]
        
        # Skip if any code list is empty
        if len(diag_codes) == 0 or len(med_codes) == 0 or len(proc_codes) == 0:
            continue
        
        # For dialysis prediction, we'll create two visits per patient:
        # Visit 1: Historical data (input features)
        # Visit 2: Current data with dialysis flag as target
        
        # Split diagnosis codes into historical and current
        mid_point = len(diag_codes) // 2 if len(diag_codes) > 1 else 1
        hist_diag = diag_codes[:mid_point]
        curr_diag = diag_codes[mid_point:] if len(diag_codes) > 1 else diag_codes
        
        # Split other codes similarly
        mid_med = len(med_codes) // 2 if len(med_codes) > 1 else 1
        hist_med = med_codes[:mid_med]
        curr_med = med_codes[mid_med:] if len(med_codes) > 1 else med_codes
        
        mid_proc = len(proc_codes) // 2 if len(proc_codes) > 1 else 1
        hist_proc = proc_codes[:mid_proc]
        curr_proc = proc_codes[mid_proc:] if len(proc_codes) > 1 else proc_codes
        
        # Create historical visit (input)
        hist_sample = {
            "visit_id": f"{patient_id}_hist",
            "patient_id": patient_id,
            "conditions": hist_diag,
            "procedures": hist_proc,
            "drugs": hist_med,
            "cond_hist": hist_diag,
            "adm_time": "2023-01-01",
            "lab_dialysis_flag": 0.0,  # Historical visit - no dialysis yet
            "lab_egfr_min": row.get('lab_egfr_min', 0.0),
            "lab_creatinine_max": row.get('lab_creatinine_max', 0.0),
            "lab_bun_max": row.get('lab_bun_max', 0.0),
            "lab_potassium_max": row.get('lab_potassium_max', 0.0),
        }
        
        # Create current visit (target) - this will be used for prediction
        curr_sample = {
            "visit_id": f"{patient_id}_curr",
            "patient_id": patient_id,
            "conditions": curr_diag,
            "procedures": curr_proc,
            "drugs": curr_med,
            "cond_hist": curr_diag,
            "adm_time": "2023-06-01",
            "lab_dialysis_flag": row.get('lab_dialysis_flag', 0.0),
            "lab_egfr_min": row.get('lab_egfr_min', 0.0),
            "lab_creatinine_max": row.get('lab_creatinine_max', 0.0),
            "lab_bun_max": row.get('lab_bun_max', 0.0),
            "lab_potassium_max": row.get('lab_potassium_max', 0.0),
        }
        
        samples.extend([hist_sample, curr_sample])
    
    print(f"Loaded {len(samples)} samples from preprocessed data")
    return samples


def dialysis_prediction_fn(data_path):
    """
    Data processing function for dialysis prediction task using preprocessed data
    
    Args:
        data_path: Path to the preprocessed data directory
        
    Returns:
        List of samples formatted for training
    """
    return load_preprocessed_data(data_path)

