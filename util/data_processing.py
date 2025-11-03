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
        # Import HyperbolicEntailmentCones if not already imported
        # This is needed for pickle to deserialize the model object
        import sys
        import importlib.util
        
        if HyperbolicEntailmentCones is None:
            try:
                # Try direct import first
                import hyperbolic_entailment_cones
                sys.modules['hyperbolic_entailment_cones'] = hyperbolic_entailment_cones
                # Make HyperbolicEntailmentCones available in this module's namespace
                globals()['HyperbolicEntailmentCones'] = hyperbolic_entailment_cones.HyperbolicEntailmentCones
            except ImportError:
                # If direct import fails, try using importlib
                try:
                    spec = importlib.util.find_spec("hyperbolic_entailment_cones")
                    if spec is not None:
                        hyperbolic_module = importlib.util.module_from_spec(spec)
                        sys.modules['hyperbolic_entailment_cones'] = hyperbolic_module
                        spec.loader.exec_module(hyperbolic_module)
                        # Make HyperbolicEntailmentCones available in this module's namespace
                        globals()['HyperbolicEntailmentCones'] = hyperbolic_module.HyperbolicEntailmentCones
                except Exception as e:
                    print(f"Warning: Could not import HyperbolicEntailmentCones: {e}")
                    print("This may cause issues if loading hyperbolic_entailment_cones.py format files")
        
        print(f"Loading hyperbolic embeddings from: {embedding_file}")
        
        # Create a custom unpickler that can find the class
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Try to find HyperbolicEntailmentCones class
                if name == 'HyperbolicEntailmentCones':
                    # First try the hyperbolic_entailment_cones module
                    if 'hyperbolic_entailment_cones' in sys.modules:
                        mod = sys.modules['hyperbolic_entailment_cones']
                        if hasattr(mod, 'HyperbolicEntailmentCones'):
                            return mod.HyperbolicEntailmentCones
                    # Also try importing it if not already imported
                    try:
                        import hyperbolic_entailment_cones
                        if hasattr(hyperbolic_entailment_cones, 'HyperbolicEntailmentCones'):
                            return hyperbolic_entailment_cones.HyperbolicEntailmentCones
                    except:
                        pass
                    # Fall back to default behavior (try original module path)
                # Use default behavior for other classes
                try:
                    return super().find_class(module, name)
                except AttributeError:
                    # If class is not found in original module, try hyperbolic_entailment_cones
                    if module == '__main__' and name == 'HyperbolicEntailmentCones':
                        if 'hyperbolic_entailment_cones' in sys.modules:
                            mod = sys.modules['hyperbolic_entailment_cones']
                            if hasattr(mod, 'HyperbolicEntailmentCones'):
                                return mod.HyperbolicEntailmentCones
                    raise
        
        with open(embedding_file, 'rb') as f:
            unpickler = CustomUnpickler(f)
            loaded_data = unpickler.load()
        
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


def create_embedding_sequence_with_visit_markers(cond_hist, embeddings_cache, max_seq_length=200):
    """
    Create embedding sequence from condition history with visit boundary markers
    
    Args:
        cond_hist: List of lists, each inner list contains codes for one visit
                  Last visit is empty to prevent leakage
        embeddings_cache: ConditionsHyperbolicEmbedder instance
        max_seq_length: Maximum sequence length (for padding)
        
    Returns:
        Tuple of (sequence_tensor, attention_mask)
        - sequence_tensor: (seq_len, embedding_dim) padded tensor
        - attention_mask: (seq_len,) tensor with 1 for real tokens, 0 for padding
    """
    global _visit_sep_embedding
    
    if _visit_sep_embedding is None:
        raise ValueError("Visit separator embedding not initialized. Call load_hyperbolic_embeddings() first.")
    
    embedding_dim = embeddings_cache.get_embedding_dim()
    sequence_embeddings = []
    
    # Process each visit (except the last empty one)
    for visit_idx, visit_codes in enumerate(cond_hist[:-1]):  # Skip last empty visit
        # Add embeddings for each code in this visit
        for code in visit_codes:
            if code in embeddings_cache.code2embedding:
                sequence_embeddings.append(embeddings_cache.code2embedding[code])
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


def vectorize_pair(s, y_codes, vocabs, use_current_step=False, use_hyperbolic_embeddings=False, embedding_file="hyperbolic_embeddings.pkl", max_seq_length=200):
    """Vectorize sample pair"""
    diag_stoi, proc_stoi, drug_stoi, y_stoi = vocabs
    
    if use_hyperbolic_embeddings:
        # Load hyperbolic embeddings if not already cached
        embeddings_cache = load_hyperbolic_embeddings(embedding_file)
        
        # Create embedding sequence with visit markers
        X, attention_mask = create_embedding_sequence_with_visit_markers(
            s["cond_hist"], embeddings_cache, max_seq_length
        )
        
        # Return both sequence and attention mask
        return X, attention_mask, y_codes
    else:
        # Original multi-hot implementation
        # Admission prediction: don't look at current step's proc/drug; discharge prediction can look
        # if use_current_step:
        #     proc_hist = s["procedures"]
        #     drug_hist = s["drugs"]
        # else:
        #     proc_hist = s["procedures"][:-1] if len(s["procedures"])>0 else []
        #     drug_hist = s["drugs"][:-1] if len(s["drugs"])>0 else []

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


def prepare_XY(pairs, vocabs, use_current_step=False, use_hyperbolic_embeddings=False, embedding_file="hyperbolic_embeddings.pkl", max_seq_length=200):
    """Prepare training data X and Y"""
    if use_hyperbolic_embeddings:
        # Sequential data preparation
        Xs, masks, Ys = [], [], []
        for s, y_codes in pairs:
            X, attention_mask, y_codes_list = vectorize_pair(s, y_codes, vocabs, use_current_step=use_current_step, 
                                                           use_hyperbolic_embeddings=True, embedding_file=embedding_file, 
                                                           max_seq_length=max_seq_length)
            Xs.append(X)
            masks.append(attention_mask)
            Ys.append(y_codes_list)
        
        # Convert to tensors
        X_tensor = torch.stack(Xs)  # (batch_size, max_seq_length, embedding_dim)
        mask_tensor = torch.stack(masks)  # (batch_size, max_seq_length)
        
        # Convert y_codes to multi-hot vectors
        diag_stoi, proc_stoi, drug_stoi, y_stoi = vocabs
        Y_tensor = torch.zeros(len(Ys), len(y_stoi), dtype=torch.float32)
        for i, y_codes_list in enumerate(Ys):
            for c in y_codes_list:
                if c in y_stoi: 
                    Y_tensor[i, y_stoi[c]] = 1.0
        
        return X_tensor, mask_tensor, Y_tensor
    else:
        # Original multi-hot implementation
        Xs, Ys = [], []
        for s, y_codes in pairs:
            X, y = vectorize_pair(s, y_codes, vocabs, use_current_step=use_current_step, 
                                use_hyperbolic_embeddings=False)
            Xs.append(X)
            Ys.append(y)
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

