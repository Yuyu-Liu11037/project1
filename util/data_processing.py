"""
Data processing module
Contains data preprocessing, vectorization, vocabulary building functions
"""
from collections import defaultdict, Counter
import torch
import numpy as np
import pickle
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from pyhealth.data import Patient
from pyhealth.medcode import CrossMap


mapping = CrossMap("ICD10CM", "CCSCM")


class GEMMapper:
    """
    ICD-9-CM to ICD-10-CM mapping using General Equivalence Mappings (GEM)
    """
    
    def __init__(self, gem_file_path=None):
        """
        Initialize GEM mapper
        
        Args:
            gem_file_path: Path to GEM mapping file (CSV format)
                          If None, will use a built-in sample mapping
        """
        self.icd9_to_icd10 = {}
        self.load_gem_mapping(gem_file_path)
    
    def load_gem_mapping(self, gem_file_path):
        """
        Load GEM mapping from file or use built-in sample mapping
        
        Args:
            gem_file_path: Path to GEM mapping file
        """
        if gem_file_path and os.path.exists(gem_file_path):
            self._load_from_file(gem_file_path)
        else:
            self._load_sample_mapping()
    
    def _load_from_file(self, file_path):
        """
        Load GEM mapping from CSV file
        
        Expected CSV format:
        icd9_code,icd10_code,approximate,no_map,combination,scenario,choice_list
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Loading GEM mapping from: {file_path}")
            
            for _, row in df.iterrows():
                icd9_code = str(row['icd9_code']).strip()
                icd10_code = str(row['icd10_code']).strip()
                
                # Skip invalid mappings
                if icd9_code == 'nan' or icd10_code == 'nan':
                    continue
                
                if icd9_code not in self.icd9_to_icd10:
                    self.icd9_to_icd10[icd9_code] = []
                
                if icd10_code not in self.icd9_to_icd10[icd9_code]:
                    self.icd9_to_icd10[icd9_code].append(icd10_code)
            
            print(f"Loaded {len(self.icd9_to_icd10)} ICD-9-CM to ICD-10-CM mappings")
            
        except Exception as e:
            print(f"Error loading GEM file {file_path}: {e}")
            print("Falling back to sample mapping...")
            self._load_sample_mapping()
    
    def _load_sample_mapping(self):
        """
        Load a sample mapping for common ICD-9-CM codes
        This is a subset of common mappings for demonstration purposes
        """
        print("Using sample ICD-9-CM to ICD-10-CM mapping...")
        
        # Sample mappings for common codes
        sample_mappings = {
            # Diabetes codes
            '250.00': ['E10.9'],
            '250.01': ['E10.9'],
            '250.02': ['E10.9'],
            '250.03': ['E10.9'],
            '250.10': ['E11.9'],
            '250.11': ['E11.9'],
            '250.12': ['E11.9'],
            '250.13': ['E11.9'],
            '250.20': ['E10.9'],
            '250.21': ['E10.9'],
            '250.22': ['E10.9'],
            '250.23': ['E10.9'],
            '250.30': ['E10.9'],
            '250.31': ['E10.9'],
            '250.32': ['E10.9'],
            '250.33': ['E10.9'],
            '250.40': ['E10.9'],
            '250.41': ['E10.9'],
            '250.42': ['E10.9'],
            '250.43': ['E10.9'],
            '250.50': ['E10.9'],
            '250.51': ['E10.9'],
            '250.52': ['E10.9'],
            '250.53': ['E10.9'],
            '250.60': ['E10.9'],
            '250.61': ['E10.9'],
            '250.62': ['E10.9'],
            '250.63': ['E10.9'],
            '250.70': ['E10.9'],
            '250.71': ['E10.9'],
            '250.72': ['E10.9'],
            '250.73': ['E10.9'],
            '250.80': ['E10.9'],
            '250.81': ['E10.9'],
            '250.82': ['E10.9'],
            '250.83': ['E10.9'],
            '250.90': ['E10.9'],
            '250.91': ['E10.9'],
            '250.92': ['E10.9'],
            '250.93': ['E10.9'],
            
            # Hypertension codes
            '401.0': ['I10'],
            '401.1': ['I10'],
            '401.9': ['I10'],
            '402.00': ['I11.9'],
            '402.01': ['I11.9'],
            '402.10': ['I11.9'],
            '402.11': ['I11.9'],
            '402.90': ['I11.9'],
            '402.91': ['I11.9'],
            '403.00': ['I12.9'],
            '403.01': ['I12.9'],
            '403.10': ['I12.9'],
            '403.11': ['I12.9'],
            '403.90': ['I12.9'],
            '403.91': ['I12.9'],
            '404.00': ['I13.9'],
            '404.01': ['I13.9'],
            '404.02': ['I13.9'],
            '404.03': ['I13.9'],
            '404.10': ['I13.9'],
            '404.11': ['I13.9'],
            '404.12': ['I13.9'],
            '404.13': ['I13.9'],
            '404.90': ['I13.9'],
            '404.91': ['I13.9'],
            '404.92': ['I13.9'],
            '404.93': ['I13.9'],
            '405.01': ['I15.1'],
            '405.09': ['I15.9'],
            '405.11': ['I15.1'],
            '405.19': ['I15.9'],
            '405.91': ['I15.1'],
            '405.99': ['I15.9'],
            
            # Heart failure codes
            '428.0': ['I50.9'],
            '428.1': ['I50.9'],
            '428.20': ['I50.9'],
            '428.21': ['I50.9'],
            '428.22': ['I50.9'],
            '428.23': ['I50.9'],
            '428.30': ['I50.9'],
            '428.31': ['I50.9'],
            '428.32': ['I50.9'],
            '428.33': ['I50.9'],
            '428.40': ['I50.9'],
            '428.41': ['I50.9'],
            '428.42': ['I50.9'],
            '428.43': ['I50.9'],
            '428.9': ['I50.9'],
            
            # AKI codes
            '584.5': ['N17.0'],
            '584.6': ['N17.1'],
            '584.7': ['N17.2'],
            '584.8': ['N17.8'],
            '584.9': ['N17.9'],
            
            # Chronic kidney disease codes
            '585.1': ['N18.1'],
            '585.2': ['N18.2'],
            '585.3': ['N18.3'],
            '585.4': ['N18.4'],
            '585.5': ['N18.5'],
            '585.6': ['N18.6'],
            '585.9': ['N18.9'],
            
            # Pneumonia codes
            '481': ['J13'],
            '482.0': ['J15.1'],
            '482.1': ['J15.2'],
            '482.2': ['J15.3'],
            '482.30': ['J15.4'],
            '482.31': ['J15.4'],
            '482.32': ['J15.4'],
            '482.39': ['J15.4'],
            '482.40': ['J15.4'],
            '482.41': ['J15.4'],
            '482.49': ['J15.4'],
            '482.81': ['J15.8'],
            '482.82': ['J15.8'],
            '482.83': ['J15.8'],
            '482.84': ['J15.8'],
            '482.89': ['J15.8'],
            '482.9': ['J15.9'],
            '483.0': ['J15.6'],
            '483.1': ['J15.6'],
            '483.8': ['J15.6'],
            '484.1': ['J15.6'],
            '484.3': ['J15.6'],
            '484.5': ['J15.6'],
            '484.6': ['J15.6'],
            '484.7': ['J15.6'],
            '484.8': ['J15.6'],
            '485': ['J15.9'],
            '486': ['J18.9'],
            
            # COPD codes
            '490': ['J44.1'],
            '491.0': ['J44.1'],
            '491.1': ['J44.1'],
            '491.20': ['J44.1'],
            '491.21': ['J44.1'],
            '491.22': ['J44.1'],
            '491.8': ['J44.1'],
            '491.9': ['J44.1'],
            '492.0': ['J44.1'],
            '492.8': ['J44.1'],
            '493.00': ['J45.9'],
            '493.01': ['J45.9'],
            '493.02': ['J45.9'],
            '493.10': ['J45.9'],
            '493.11': ['J45.9'],
            '493.12': ['J45.9'],
            '493.20': ['J45.9'],
            '493.21': ['J45.9'],
            '493.22': ['J45.9'],
            '493.81': ['J45.9'],
            '493.82': ['J45.9'],
            '493.90': ['J45.9'],
            '493.91': ['J45.9'],
            '493.92': ['J45.9'],
            '494.0': ['J44.1'],
            '494.1': ['J44.1'],
            '495.0': ['J44.1'],
            '495.1': ['J44.1'],
            '495.2': ['J44.1'],
            '495.3': ['J44.1'],
            '495.4': ['J44.1'],
            '495.5': ['J44.1'],
            '495.6': ['J44.1'],
            '495.7': ['J44.1'],
            '495.8': ['J44.1'],
            '495.9': ['J44.1'],
            '496': ['J44.1'],
            
            # Stroke codes
            '430': ['I60.9'],
            '431': ['I61.9'],
            '432.0': ['I62.0'],
            '432.1': ['I62.1'],
            '432.9': ['I62.9'],
            '433.00': ['I63.9'],
            '433.01': ['I63.9'],
            '433.10': ['I63.9'],
            '433.11': ['I63.9'],
            '433.20': ['I63.9'],
            '433.21': ['I63.9'],
            '433.30': ['I63.9'],
            '433.31': ['I63.9'],
            '433.80': ['I63.9'],
            '433.81': ['I63.9'],
            '433.90': ['I63.9'],
            '433.91': ['I63.9'],
            '434.00': ['I63.9'],
            '434.01': ['I63.9'],
            '434.10': ['I63.9'],
            '434.11': ['I63.9'],
            '434.90': ['I63.9'],
            '434.91': ['I63.9'],
            '435.0': ['G45.9'],
            '435.1': ['G45.9'],
            '435.2': ['G45.9'],
            '435.3': ['G45.9'],
            '435.8': ['G45.9'],
            '435.9': ['G45.9'],
            '436': ['I64'],
            '437.0': ['G45.9'],
            '437.1': ['G45.9'],
            '437.2': ['G45.9'],
            '437.3': ['G45.9'],
            '437.4': ['G45.9'],
            '437.5': ['G45.9'],
            '437.6': ['G45.9'],
            '437.7': ['G45.9'],
            '437.8': ['G45.9'],
            '437.9': ['G45.9'],
            '438.0': ['I69.9'],
            '438.10': ['I69.9'],
            '438.11': ['I69.9'],
            '438.12': ['I69.9'],
            '438.13': ['I69.9'],
            '438.14': ['I69.9'],
            '438.19': ['I69.9'],
            '438.20': ['I69.9'],
            '438.21': ['I69.9'],
            '438.22': ['I69.9'],
            '438.30': ['I69.9'],
            '438.31': ['I69.9'],
            '438.32': ['I69.9'],
            '438.40': ['I69.9'],
            '438.41': ['I69.9'],
            '438.42': ['I69.9'],
            '438.50': ['I69.9'],
            '438.51': ['I69.9'],
            '438.52': ['I69.9'],
            '438.53': ['I69.9'],
            '438.6': ['I69.9'],
            '438.7': ['I69.9'],
            '438.8': ['I69.9'],
            '438.81': ['I69.9'],
            '438.82': ['I69.9'],
            '438.83': ['I69.9'],
            '438.84': ['I69.9'],
            '438.85': ['I69.9'],
            '438.89': ['I69.9'],
            '438.9': ['I69.9'],
        }
        
        self.icd9_to_icd10 = sample_mappings
        print(f"Loaded {len(self.icd9_to_icd10)} sample ICD-9-CM to ICD-10-CM mappings")
    
    def is_icd9_code(self, code):
        """
        Check if a code is in ICD-9-CM format
        
        Args:
            code: Code string to check
            
        Returns:
            True if code appears to be ICD-9-CM format
        """
        if not code or not isinstance(code, str):
            return False
        
        # Remove any decimal points for checking
        clean_code = code.replace('.', '')
        
        # ICD-9-CM codes are typically 3-5 digits
        if len(clean_code) < 3 or len(clean_code) > 5:
            return False
        
        # Check if it's numeric (ICD-9-CM codes are numeric)
        if not clean_code.isdigit():
            return False
        
        # Additional heuristic: ICD-9-CM codes typically start with certain ranges
        # This is a simplified check
        first_digit = clean_code[0]
        if first_digit in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            return True
        
        return False
    
    def map_to_icd10(self, icd9_code):
        """
        Map ICD-9-CM code to ICD-10-CM code
        
        Args:
            icd9_code: ICD-9-CM code string
            
        Returns:
            ICD-10-CM code string, or original code if no mapping found
        """
        if not self.is_icd9_code(icd9_code):
            return icd9_code
        
        # Try exact match first
        if icd9_code in self.icd9_to_icd10:
            # Return the first mapping if multiple exist
            return self.icd9_to_icd10[icd9_code][0]
        
        # Try without decimal point
        clean_code = icd9_code.replace('.', '')
        if clean_code in self.icd9_to_icd10:
            return self.icd9_to_icd10[clean_code][0]
        
        # Try with decimal point added
        if len(clean_code) >= 3:
            formatted_code = clean_code[:3] + '.' + clean_code[3:] if len(clean_code) > 3 else clean_code
            if formatted_code in self.icd9_to_icd10:
                return self.icd9_to_icd10[formatted_code][0]
        
        # No mapping found, return original code
        return icd9_code
    
    def get_mapping_stats(self):
        """
        Get statistics about the mapping
        
        Returns:
            Dictionary with mapping statistics
        """
        total_mappings = len(self.icd9_to_icd10)
        single_mappings = sum(1 for codes in self.icd9_to_icd10.values() if len(codes) == 1)
        multiple_mappings = sum(1 for codes in self.icd9_to_icd10.values() if len(codes) > 1)
        
        return {
            'total_icd9_codes': total_mappings,
            'single_mappings': single_mappings,
            'multiple_mappings': multiple_mappings,
            'avg_icd10_per_icd9': sum(len(codes) for codes in self.icd9_to_icd10.values()) / total_mappings if total_mappings > 0 else 0
        }


def load_hyperbolic_embeddings(embedding_file: str):
    """
    Load pre-trained hyperbolic embeddings from file
    
    Args:
        embedding_file: Path to the saved embeddings file
        
    Returns:
        ConditionsHyperbolicEmbedder instance with loaded embeddings
    """
    with open(embedding_file, 'rb') as f:
        conditions_embedder = pickle.load(f)
    
    print(f"Loaded hyperbolic embeddings from: {embedding_file}")
    print(f"Embedding dimension: {conditions_embedder.get_embedding_dim()}")
    print(f"Number of conditions: {len(conditions_embedder.conditions_codes)}")
    
    return conditions_embedder


def preprocess_conditions_codes(samples, gem_mapper=None):
    """
    Preprocess conditions codes by:
    1. Converting ICD-9-CM codes to ICD-10-CM codes using GEM mapping
    2. Adding decimal points to codes longer than 3 characters that don't already contain decimal points
    
    Args:
        samples: List of sample dictionaries containing conditions codes
        gem_mapper: GEMMapper instance for ICD-9-CM to ICD-10-CM conversion
        
    Returns:
        List of sample dictionaries with preprocessed conditions codes
    """
    processed_samples = []
    
    # Initialize GEM mapper if not provided
    if gem_mapper is None:
        gem_mapper = GEMMapper()
    
    for sample in samples:
        # Create a copy of the sample to avoid modifying the original
        processed_sample = sample.copy()
        
        # Process conditions field (CCS codes)
        if 'conditions' in processed_sample and processed_sample['conditions']:
            processed_conditions = []
            for code in processed_sample['conditions']:
                # First convert ICD-9-CM to ICD-10-CM if needed
                converted_code = gem_mapper.map_to_icd10(code)
                # Then add decimal point if needed
                processed_code = add_decimal_point_if_needed(converted_code)
                processed_conditions.append(processed_code)
            processed_sample['conditions'] = processed_conditions
        
        # Process cond_hist field (ICD codes)
        if 'cond_hist' in processed_sample and processed_sample['cond_hist']:
            processed_cond_hist = []
            for visit_conditions in processed_sample['cond_hist']:
                if visit_conditions:  # Only process non-empty visits
                    processed_visit_conditions = []
                    for code in visit_conditions:
                        # First convert ICD-9-CM to ICD-10-CM if needed
                        converted_code = gem_mapper.map_to_icd10(code)
                        # Then add decimal point if needed
                        processed_code = add_decimal_point_if_needed(converted_code)
                        processed_visit_conditions.append(processed_code)
                    processed_cond_hist.append(processed_visit_conditions)
                else:
                    processed_cond_hist.append(visit_conditions)  # Keep empty visits as is
            processed_sample['cond_hist'] = processed_cond_hist
        
        processed_samples.append(processed_sample)
    
    return processed_samples


def add_decimal_point_if_needed(code):
    """
    Add decimal point to a condition code if it's longer than 3 characters 
    and doesn't already contain a decimal point.
    
    Args:
        code: Condition code string
        
    Returns:
        Processed condition code string
    """
    if len(code) > 3 and '.' not in code:
        # Insert decimal point after the third character
        return code[:3] + '.' + code[3:]
    return code


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


def vectorize_pair(s, y_codes, vocabs, use_current_step=False, conditions_embedder=None, max_seq_length=None):
    """Vectorize sample pair"""
    diag_stoi, proc_stoi, drug_stoi, y_stoi = vocabs
    
    # Admission prediction: don't look at current step's proc/drug; discharge prediction can look
    if use_current_step:
        proc_hist = s["procedures"]
        drug_hist = s["drugs"]
    else:
        proc_hist = s["procedures"][:-1] if len(s["procedures"])>0 else []
        drug_hist = s["drugs"][:-1] if len(s["drugs"])>0 else []

    # Use hyperbolic embeddings for conditions if embedder is provided, otherwise multi-hot
    if conditions_embedder is not None:
        # Get sequence of embeddings for each visit in cond_hist
        embeddings_list = []
        for visit_conditions in s["cond_hist"]:
            if visit_conditions:  # Only process non-empty visits
                visit_embeddings = conditions_embedder.get_embedding_sequences(visit_conditions)
                embeddings_list.append(visit_embeddings)
        
        if embeddings_list:
            # Concatenate all visit embeddings into a single sequence
            x_diag = torch.cat(embeddings_list, dim=0)  # Shape: [total_conditions, embedding_dim]
        else:
            # No conditions, return empty tensor with correct shape
            x_diag = torch.zeros(0, conditions_embedder.get_embedding_dim())
        
        # Apply padding/truncation if max_seq_length is specified
        if max_seq_length is not None:
            seq_len = x_diag.size(0)
            if seq_len > max_seq_length:
                # Truncate to max length
                x_diag = x_diag[:max_seq_length]
            elif seq_len < max_seq_length:
                # Pad with zeros
                padding = torch.zeros(max_seq_length - seq_len, x_diag.size(1))
                x_diag = torch.cat([x_diag, padding], dim=0)
    else:
        # Fallback to multi-hot encoding
        x_diag = multihot_from_sequence(s["cond_hist"], diag_stoi)  # Historical ICD (current step is empty)
    
    # x_proc = multihot_from_sequence(proc_hist, proc_stoi)
    # x_drug = multihot_from_sequence(drug_hist, drug_stoi)
    # X = torch.cat([x_diag, x_proc, x_drug], dim=0)
    X = x_diag

    y = torch.zeros(len(y_stoi), dtype=torch.float32)
    for c in y_codes:
        if c in y_stoi: 
            y[y_stoi[c]] = 1.0
    return X, y


def prepare_XY(pairs, vocabs, use_current_step=False, conditions_embedder=None, max_seq_length=None):
    """Prepare training data X and Y"""
    Xs, Ys = [], []
    for s, y_codes in pairs:
        X, y = vectorize_pair(s, y_codes, vocabs, use_current_step=use_current_step, 
                             conditions_embedder=conditions_embedder, max_seq_length=max_seq_length)
        Xs.append(X)
        Ys.append(y)
    
    # Handle variable-length sequences for hyperbolic embeddings
    if conditions_embedder is not None:
        # For hyperbolic embeddings, we need to pad sequences to the same length
        if max_seq_length is None:
            # Find the maximum sequence length
            max_len = max(x.size(0) for x in Xs) if Xs else 0
            print(f"Auto-determined max sequence length: {max_len}")
        else:
            max_len = max_seq_length
        
        # Pad all sequences to the same length
        padded_Xs = []
        for x in Xs:
            if x.size(0) < max_len:
                padding = torch.zeros(max_len - x.size(0), x.size(1))
                x_padded = torch.cat([x, padding], dim=0)
            else:
                x_padded = x[:max_len]  # Truncate if too long
            padded_Xs.append(x_padded)
        
        return torch.stack(padded_Xs), torch.stack(Ys)
    else:
        # For multi-hot encoding, sequences are already fixed length
        return torch.stack(Xs), torch.stack(Ys)


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


def dialysis_prediction_mimic4_fn(patient: Patient):
    """
    Data processing function for MIMIC-IV dialysis prediction task for AKI patients
    Based on the approach from aki.ipynb but adapted for MIMIC-IV structure
    """
    samples = []
    
    # AKI ICD codes for MIMIC-IV (ICD-9 format)
    # Based on debug analysis: 5849, 5845, 5848 are the most common AKI codes
    aki_codes = ["584", "584.5", "584.6", "584.7", "584.8", "584.9", "5849", "5845", "5848"]
    
    # Dialysis procedure codes for MIMIC-IV (ICD-9 format)
    # Based on debug analysis: 3995 is the main dialysis procedure code
    dialysis_codes_cpt = ['3995', '3996']  # Hemodialysis and peritoneal dialysis
    dialysis_codes_icd = ['585', '585.1', '585.2', '585.3', '585.4', '585.5', '585.6', '585.9',  # Chronic kidney disease
                         '586', 'V45.1', 'V45.11', 'V45.12', 'V58.61', 'V58.66', 'V58.67']  # Dialysis-related codes
    dialysis_codes_hcpcs = ['6909', '0SP909Z']  # Other dialysis-related procedures
    
    # Sort visits by encounter time
    visit_ls = sorted(patient.visits.keys(), key=lambda vid: patient.visits[vid].encounter_time)
    
    # Check if patient has AKI diagnosis
    has_aki = False
    aki_first_date = None
    
    for visit_id in visit_ls:
        visit = patient.visits[visit_id]
        conditions = visit.get_code_list(table="diagnoses_icd")
        
        # Check for AKI diagnosis
        for condition in conditions:
            if condition in aki_codes:
                has_aki = True
                if aki_first_date is None:
                    aki_first_date = visit.encounter_time
                break
        
        if has_aki:
            break
    
    if not has_aki:
        return []
    
    # Check for dialysis procedures after AKI diagnosis
    has_dialysis = False
    dialysis_date = None
    
    for visit_id in visit_ls:
        visit = patient.visits[visit_id]
        
        # Skip visits before AKI diagnosis
        if visit.encounter_time < aki_first_date:
            continue
            
        procedures = visit.get_code_list(table="procedures_icd")
        
        # Check for dialysis procedures
        for procedure in procedures:
            if (procedure in dialysis_codes_cpt or 
                procedure in dialysis_codes_icd or 
                procedure in dialysis_codes_hcpcs):
                has_dialysis = True
                dialysis_date = visit.encounter_time
                break
        
        if has_dialysis:
            break
    
    # Collect medication data for AKI patients
    # We'll use all medications from visits around AKI diagnosis
    medications = []
    conditions_history = []
    procedures_history = []
    visit_times = []
    
    for visit_id in visit_ls:
        visit = patient.visits[visit_id]
        
        # Include medications from visits within a reasonable timeframe around AKI
        # (e.g., 30 days before AKI to 30 days after AKI)
        time_diff = (visit.encounter_time - aki_first_date).days
        
        if -30 <= time_diff <= 30:  # 30 days before and after AKI
            drugs = visit.get_code_list(table="prescriptions")
            # Convert to ATC 3 level (first 4 characters)
            drugs_atc3 = [drug[:4] for drug in drugs if len(drug) >= 4]
            
            conditions = visit.get_code_list(table="diagnoses_icd")
            procedures = visit.get_code_list(table="procedures_icd")
            
            medications.extend(drugs_atc3)
            conditions_history.extend(conditions)
            procedures_history.extend(procedures)
            visit_times.append(visit.encounter_time)
    
    # Remove duplicates while preserving order
    medications = list(dict.fromkeys(medications))
    conditions_history = list(dict.fromkeys(conditions_history))
    procedures_history = list(dict.fromkeys(procedures_history))
    
    if len(medications) == 0:
        return []
    
    # Create sample for dialysis prediction
    sample = {
        "patient_id": patient.patient_id,
        "visit_id": f"{patient.patient_id}_aki_visit",
        "medications": medications,
        "conditions": conditions_history,
        "procedures": procedures_history,
        "aki_date": aki_first_date.strftime("%Y-%m-%d %H:%M"),
        "dialysis_date": dialysis_date.strftime("%Y-%m-%d %H:%M") if dialysis_date else "None",
        "dialysis_label": int(has_dialysis)
    }
    
    return [sample]


def build_dialysis_pairs(samples):
    """
    Build training pairs for dialysis prediction
    Returns pairs: list of (X_sample_dict, y_label)
    """
    pairs = []
    for sample in samples:
        # Create feature vector from medications, conditions, and procedures
        features = {
            "patient_id": sample["patient_id"],
            "medications": sample["medications"],
            "conditions": sample["conditions"], 
            "procedures": sample["procedures"]
        }
        label = sample["dialysis_label"]
        pairs.append((features, label))
    
    return pairs


def build_dialysis_vocab_from_pairs(pairs):
    """Build vocabulary from dialysis prediction training pairs"""
    med_c, cond_c, proc_c = Counter(), Counter(), Counter()
    
    for features, label in pairs:
        med_c.update(features["medications"])
        cond_c.update(features["conditions"])
        proc_c.update(features["procedures"])
    
    def mk_vocab(cnt):
        itos = [c for c, _ in cnt.most_common()]
        stoi = {c: i for i, c in enumerate(itos)}
        return stoi, itos
    
    return mk_vocab(med_c), mk_vocab(cond_c), mk_vocab(proc_c)


def vectorize_dialysis_pair(features, label, vocabs):
    """Vectorize dialysis prediction sample pair"""
    med_stoi, cond_stoi, proc_stoi = vocabs
    
    # Create multi-hot vectors for each modality
    x_med = torch.zeros(len(med_stoi), dtype=torch.float32)
    for med in features["medications"]:
        if med in med_stoi:
            x_med[med_stoi[med]] = 1.0
    
    x_cond = torch.zeros(len(cond_stoi), dtype=torch.float32)
    for cond in features["conditions"]:
        if cond in cond_stoi:
            x_cond[cond_stoi[cond]] = 1.0
    
    x_proc = torch.zeros(len(proc_stoi), dtype=torch.float32)
    for proc in features["procedures"]:
        if proc in proc_stoi:
            x_proc[proc_stoi[proc]] = 1.0
    
    # Concatenate all features
    X = torch.cat([x_med, x_cond, x_proc], dim=0)
    
    # Binary label
    y = torch.tensor(label, dtype=torch.float32)
    
    return X, y


def prepare_dialysis_XY(pairs, vocabs):
    """Prepare dialysis prediction training data X and Y"""
    Xs, Ys = [], []
    for features, label in pairs:
        X, y = vectorize_dialysis_pair(features, label, vocabs)
        Xs.append(X)
        Ys.append(y)
    return torch.stack(Xs), torch.stack(Ys)

