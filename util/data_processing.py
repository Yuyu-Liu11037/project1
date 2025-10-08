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


def vectorize_pair(s, y_codes, vocabs, use_current_step=False):
    """Vectorize sample pair"""
    diag_stoi, proc_stoi, drug_stoi, y_stoi = vocabs
    
    # Admission prediction: don't look at current step's proc/drug; discharge prediction can look
    if use_current_step:
        proc_hist = s["procedures"]
        drug_hist = s["drugs"]
    else:
        proc_hist = s["procedures"][:-1] if len(s["procedures"])>0 else []
        drug_hist = s["drugs"][:-1] if len(s["drugs"])>0 else []

    x_diag = multihot_from_sequence(s["cond_hist"], diag_stoi)  # Historical ICD (current step is empty)
    x_proc = multihot_from_sequence(proc_hist, proc_stoi)
    x_drug = multihot_from_sequence(drug_hist, drug_stoi)
    X = torch.cat([x_diag, x_proc, x_drug], dim=0)

    y = torch.zeros(len(y_stoi), dtype=torch.float32)
    for c in y_codes:
        if c in y_stoi: 
            y[y_stoi[c]] = 1.0
    return X, y


def prepare_XY(pairs, vocabs, use_current_step=False):
    """Prepare training data X and Y"""
    Xs, Ys = [], []
    for s, y_codes in pairs:
        X, y = vectorize_pair(s, y_codes, vocabs, use_current_step=use_current_step)
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
        "has_dialysis": int(has_dialysis),  # Convert to int for pyhealth compatibility
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

