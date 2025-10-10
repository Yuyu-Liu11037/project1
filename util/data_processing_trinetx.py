"""
TriNetX Data processing module
Contains data preprocessing, vectorization, vocabulary building functions for TriNetX dataset
"""
from collections import defaultdict, Counter
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
from tqdm import tqdm


def load_trinetx_data(data_path):
    """
    Load TriNetX dataset from CSV files
    
    Args:
        data_path: Path to TriNetX data directory
        
    Returns:
        Dictionary containing loaded DataFrames
    """
    print("Loading TriNetX dataset...")
    
    data = {}
    
    # Load core tables
    data['patients'] = pd.read_csv(f"{data_path}/patient.csv")
    data['encounters'] = pd.read_csv(f"{data_path}/encounter.csv")
    data['diagnoses'] = pd.read_csv(f"{data_path}/diagnosis.csv")
    data['procedures'] = pd.read_csv(f"{data_path}/procedure.csv")
    data['medications'] = pd.read_csv(f"{data_path}/medication_drug.csv")
    
    print(f"Loaded {len(data['patients'])} patients")
    print(f"Loaded {len(data['encounters'])} encounters")
    print(f"Loaded {len(data['diagnoses'])} diagnoses")
    print(f"Loaded {len(data['procedures'])} procedures")
    print(f"Loaded {len(data['medications'])} medications")
    
    return data


def process_trinetx_patients(data):
    """
    Process TriNetX data into patient-centric samples for diagnosis prediction
    
    Args:
        data: Dictionary containing loaded DataFrames
        
    Returns:
        List of patient samples
    """
    samples = []
    
    # Convert date columns to datetime
    data['encounters']['start_date'] = pd.to_datetime(data['encounters']['start_date'], format='%Y%m%d', errors='coerce')
    data['diagnoses']['date'] = pd.to_datetime(data['diagnoses']['date'], format='%Y%m%d', errors='coerce')
    data['procedures']['date'] = pd.to_datetime(data['procedures']['date'], format='%Y%m%d', errors='coerce')
    data['medications']['start_date'] = pd.to_datetime(data['medications']['start_date'], format='%Y%m%d', errors='coerce')
    
    # Group encounters by patient
    patient_encounters = data['encounters'].groupby('patient_id')
    
    # Process patients with progress bar
    for patient_id, encounters in tqdm(patient_encounters, desc="Processing patients", total=len(patient_encounters)):
        # Sort encounters by start date
        encounters = encounters.sort_values('start_date')
        
        if len(encounters) < 2:  # Need at least 2 encounters
            continue
            
        patient_samples = []
        
        for _, encounter in encounters.iterrows():
            encounter_id = encounter['encounter_id']
            encounter_date = encounter['start_date']
            
            # Get diagnoses for this encounter
            encounter_diagnoses = data['diagnoses'][data['diagnoses']['encounter_id'] == encounter_id]
            diagnoses = encounter_diagnoses['code'].tolist()
            
            # Get procedures for this encounter
            encounter_procedures = data['procedures'][data['procedures']['encounter_id'] == encounter_id]
            procedures = encounter_procedures['code'].tolist()
            
            # Get medications for this encounter
            encounter_medications = data['medications'][data['medications']['encounter_id'] == encounter_id]
            medications = encounter_medications['code'].tolist()
            
            # Filter out empty encounters
            if len(diagnoses) == 0 and len(procedures) == 0 and len(medications) == 0:
                continue
                
            sample = {
                "visit_id": encounter_id,
                "patient_id": patient_id,
                "conditions": diagnoses,
                "procedures": procedures,
                "adm_time": encounter_date.strftime("%Y-%m-%d %H:%M") if pd.notna(encounter_date) else "Unknown",
                "drugs": medications,
                "cond_hist": diagnoses,
            }
            patient_samples.append(sample)
        
        if len(patient_samples) < 2:  # Need at least 2 visits
            continue
            
        # Add history to samples
        patient_samples[0]["cond_hist"] = [patient_samples[0]["cond_hist"]]
        patient_samples[0]["procedures"] = [patient_samples[0]["procedures"]]
        patient_samples[0]["drugs"] = [patient_samples[0]["drugs"]]
        patient_samples[0]["adm_time"] = [patient_samples[0]["adm_time"]]

        for i in range(1, len(patient_samples)):
            patient_samples[i]["drugs"] = patient_samples[i - 1]["drugs"] + [patient_samples[i]["drugs"]]
            patient_samples[i]["procedures"] = patient_samples[i - 1]["procedures"] + [patient_samples[i]["procedures"]]
            patient_samples[i]["cond_hist"] = patient_samples[i - 1]["cond_hist"] + [patient_samples[i]["cond_hist"]]
            patient_samples[i]["adm_time"] = patient_samples[i - 1]["adm_time"] + [patient_samples[i]["adm_time"]]

        # Clear current step conditions to prevent leakage
        for i in range(len(patient_samples)):
            patient_samples[i]["cond_hist"][i] = []
            
        samples.extend(patient_samples)
    
    print(f"Processed {len(samples)} samples from {len(set(s['patient_id'] for s in samples))} patients")
    return samples


def process_trinetx_dialysis_patients(data: dict,
    *,
    random_state: int = 42,
    max_window_days: int = 1100
):
    """
    构造与 aki.ipynb 同构的数据处理，并返回样本列表 samples（每个样本是一个字典），字段包含：
      - patient_id (str)
      - visit_id (str) = "{patient_id}_aki_visit"
      - medications (list[str])  # AKI→透析（或人工窗口）期间的药物代码
      - conditions  (list[str])  # 同窗口内诊断代码
      - procedures  (list[str])  # 同窗口内程序/处置代码
      - aki_date (str: "YYYY-MM-DD HH:MM")
      - dialysis_date (str: "YYYY-MM-DD HH:MM" 或 "None")
      - dialysis_label (int: 1/0)

    依赖的 data 字典需包含：
      data['patients'], data['encounters'], data['diagnoses'], data['procedures'], data['medications']
    其中字段至少包括：
      diagnoses:  patient_id, code_system, code, date(YYYYMMDD或datetime)
      procedures: patient_id, code_system, code, date(YYYYMMDD或datetime)
      medications:patient_id, code_system, brand(可为空), code, start_date(YYYYMMDD或datetime)
    """

    rng = np.random.default_rng(random_state)

    diagnoses  = data['diagnoses'].copy()
    procedures = data['procedures'].copy()
    meds       = data['medications'].copy()

    # ---- 0) 日期列转 datetime（Notebook 使用 '%Y%m%d'） ----
    def _to_dt(s):
        return pd.to_datetime(s, format='%Y%m%d', errors='coerce')

    if 'date' in diagnoses and not pd.api.types.is_datetime64_ns_dtype(diagnoses['date']):
        diagnoses['date'] = _to_dt(diagnoses['date'])
    if 'date' in procedures and not pd.api.types.is_datetime64_ns_dtype(procedures['date']):
        procedures['date'] = _to_dt(procedures['date'])
    if 'start_date' in meds and not pd.api.types.is_datetime64_ns_dtype(meds['start_date']):
        meds['start_date'] = _to_dt(meds['start_date'])

    # ---- 1) AKI 患者识别（与 notebook 的 ICD-10-CM 列表同构）----
    aki_icd10cm_codes = [
        "A98.5","D59.3","K76.7","N00.8","N00.9","N01.3","N04.2","N04.7",
        "N17","N17.0","N17.1","N17.9","N28.9","O90.4","S37.0"
    ]
    aki_dx = diagnoses[
        (diagnoses['code_system'] == 'ICD-10-CM') &
        (diagnoses['code'].isin(aki_icd10cm_codes))
    ].copy()
    if aki_dx.empty:
        return []

    idx = aki_dx.groupby('patient_id')['date'].idxmin()
    aki_first = aki_dx.loc[idx, ['patient_id', 'date']].rename(columns={'date': 'aki_date'})

    # ---- 2) 透析识别（CPT/HCPCS/ICD-10-CM）----
    dialysis_codes_cpt   = ['90935', '1012752', '90999', '90937', '90947', '90945']
    dialysis_codes_hcpcs = ['G0257']
    dialysis_codes_icd10 = ['I12.0', 'N18.6', 'Z99.2', 'D63.1']  # 与 notebook 保持一致

    proc_cpt = procedures[
        (procedures['code_system'] == 'CPT') &
        (procedures['code'].isin(dialysis_codes_cpt))
    ][['patient_id', 'date']]

    proc_hcpcs = procedures[
        (procedures['code_system'] == 'HCPCS') &
        (procedures['code'].isin(dialysis_codes_hcpcs))
    ][['patient_id', 'date']]

    diag_dialysis = diagnoses[
        (diagnoses['code_system'] == 'ICD-10-CM') &
        (diagnoses['code'].isin(dialysis_codes_icd10))
    ][['patient_id', 'date']]

    dialysis_all = pd.concat([proc_cpt, proc_hcpcs, diag_dialysis], ignore_index=True)
    if dialysis_all.empty:
        dialysis_dates = pd.DataFrame(columns=['dialysis_date'])
    else:
        dialysis_dates = dialysis_all.groupby('patient_id')['date'].min().rename('dialysis_date').to_frame()

    # ---- 3) day_gap = AKI - Dialysis，并限制 [-1100, 0] ----
    merged = aki_first.merge(dialysis_dates, on='patient_id', how='inner')
    if not merged.empty:
        merged['day_gap'] = (merged['aki_date'] - merged['dialysis_date']).dt.days
        dialysis_patients = merged[(merged['day_gap'] <= 0) & (merged['day_gap'] >= -max_window_days)].copy()
    else:
        dialysis_patients = merged.copy()

    # ---- 4) 计算阴性窗口所需的人工 gap 分布参数 ----
    if not dialysis_patients.empty:
        mean_gap = float((-dialysis_patients['day_gap']).mean())  # 阳性 gap 的正值均值
        std_gap  = float(dialysis_patients['day_gap'].std())      # 用同样的 std（与 notebook 行为一致）
        if not np.isfinite(std_gap) or std_gap <= 0:
            std_gap = 1.0
    else:
        # notebook 的经验默认值
        mean_gap, std_gap = 225.87, 299.61

    # ---- 5) 组装 samples：阳性（有透析）----
    samples = []
    pos_ids = set()

    if not dialysis_patients.empty:
        pos_ids = set(dialysis_patients['patient_id'])

        # 为提升效率，先按患者把诊断/程序/药物分组好
        dx_by_pid  = diagnoses.groupby('patient_id')
        pr_by_pid  = procedures.groupby('patient_id')
        med_by_pid = meds.groupby('patient_id')

        for _, row in dialysis_patients.iterrows():
            pid = row['patient_id']
            aki_date = row['aki_date']
            dial_date = row['dialysis_date']

            # 窗口：AKI ≤ date ≤ Dialysis
            # medications
            if pid in med_by_pid.groups:
                m = med_by_pid.get_group(pid)
                mm = m[(m['start_date']>=aki_date) & (m['start_date']<=dial_date) & (~m['code'].isna())]
                meds_list = mm['code'].astype(str).dropna().unique().tolist()
            else:
                meds_list = []

            # diagnoses (conditions)
            if pid in dx_by_pid.groups:
                d = dx_by_pid.get_group(pid)
                dd = d[(d['date']>=aki_date) & (d['date']<=dial_date) & (~d['code'].isna())]
                cond_list = dd['code'].astype(str).dropna().unique().tolist()
            else:
                cond_list = []

            # procedures
            if pid in pr_by_pid.groups:
                p = pr_by_pid.get_group(pid)
                pp = p[(p['date']>=aki_date) & (p['date']<=dial_date) & (~p['code'].isna())]
                proc_list = pp['code'].astype(str).dropna().unique().tolist()
            else:
                proc_list = []

            # 跳过完全无药物的样本（与 notebook 的实际训练目标一致）
            if len(meds_list) == 0:
                continue

            sample = {
                "patient_id": str(pid),
                "visit_id": f"{pid}_aki_visit",
                "medications": meds_list,
                "conditions": cond_list,
                "procedures": proc_list,
                "aki_date": aki_date.strftime("%Y-%m-%d %H:%M") if pd.notna(aki_date) else "Unknown",
                "dialysis_date": dial_date.strftime("%Y-%m-%d %H:%M") if pd.notna(dial_date) else "None",
                "dialysis_label": 1,
            }
            samples.append(sample)

    # ---- 6) 组装 samples：阴性（无透析）----
    neg_patients = aki_first[~aki_first['patient_id'].isin(pos_ids)]
    if not neg_patients.empty:
        # 预分组
        dx_by_pid  = diagnoses.groupby('patient_id')
        pr_by_pid  = procedures.groupby('patient_id')
        med_by_pid = meds.groupby('patient_id')

        # 为每位阴性患者采样人工 gap，并裁剪到 [0, max_window_days]
        gaps = rng.normal(loc=mean_gap, scale=std_gap, size=len(neg_patients))
        gaps = np.clip(gaps, 0, max_window_days)

        for (pid, aki_date), gap in zip(neg_patients[['patient_id','aki_date']].itertuples(index=False), gaps):
            end_date = aki_date + pd.Timedelta(days=gap)  # 保持 datetime64[ns]

            # 窗口：AKI ≤ date ≤ AKI + artificial_gap
            # medications
            if pid in med_by_pid.groups:
                m = med_by_pid.get_group(pid)
                mm = m[(m['start_date']>=aki_date) & (m['start_date']<=end_date) & (~m['code'].isna())]
                meds_list = mm['code'].astype(str).dropna().unique().tolist()
            else:
                meds_list = []

            # diagnoses
            if pid in dx_by_pid.groups:
                d = dx_by_pid.get_group(pid)
                dd = d[(d['date']>=aki_date) & (d['date']<=end_date) & (~d['code'].isna())]
                cond_list = dd['code'].astype(str).dropna().unique().tolist()
            else:
                cond_list = []

            # procedures
            if pid in pr_by_pid.groups:
                p = pr_by_pid.get_group(pid)
                pp = p[(p['date']>=aki_date) & (p['date']<=end_date) & (~p['code'].isna())]
                proc_list = pp['code'].astype(str).dropna().unique().tolist()
            else:
                proc_list = []

            # 阴性也跳过无药物的样本（与 notebook 训练流程同构）
            if len(meds_list) == 0:
                continue

            sample = {
                "patient_id": str(pid),
                "visit_id": f"{pid}_aki_visit",
                "medications": meds_list,
                "conditions": cond_list,
                "procedures": proc_list,
                "aki_date": aki_date.strftime("%Y-%m-%d %H:%M") if pd.notna(aki_date) else "Unknown",
                "dialysis_date": "None",
                "dialysis_label": 0,
            }
            samples.append(sample)

    return samples


def sort_samples_within_patient(samples):
    """Group by patient ID and sort by admission time"""
    by_pid = defaultdict(list)
    for s in samples:
        by_pid[s["patient_id"]].append(s)
    
    for pid in by_pid:
        # Sort by the last admission time in the list
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
        for visit_codes in s["drugs"]:       # Each step is a drug list
            drug_c.update(visit_codes)
        y_c.update(y)                        # Labels (diagnoses)
    
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


def build_dialysis_vocab_from_samples(samples):
    """Build vocabulary directly from samples"""
    med_c, cond_c, proc_c = Counter(), Counter(), Counter()
    
    for sample in samples:
        med_c.update(sample["medications"])
        cond_c.update(sample["conditions"])
        proc_c.update(sample["procedures"])
    
    def mk_vocab(cnt):
        itos = [c for c, _ in cnt.most_common()]
        stoi = {c: i for i, c in enumerate(itos)}
        return stoi, itos
    
    return mk_vocab(med_c), mk_vocab(cond_c), mk_vocab(proc_c)


def vectorize_dialysis_sample(sample, vocabs):
    """Vectorize dialysis prediction sample directly (more efficient than pairs)"""
    med_stoi, cond_stoi, proc_stoi = vocabs
    
    # Create multi-hot vectors for each modality
    x_med = torch.zeros(len(med_stoi), dtype=torch.float32)
    for med in sample["medications"]:
        if med in med_stoi:
            x_med[med_stoi[med]] = 1.0
    
    x_cond = torch.zeros(len(cond_stoi), dtype=torch.float32)
    for cond in sample["conditions"]:
        if cond in cond_stoi:
            x_cond[cond_stoi[cond]] = 1.0
    
    x_proc = torch.zeros(len(proc_stoi), dtype=torch.float32)
    for proc in sample["procedures"]:
        if proc in proc_stoi:
            x_proc[proc_stoi[proc]] = 1.0
    
    # Concatenate all features
    X = torch.cat([x_med, x_cond, x_proc], dim=0)
    
    # Binary label
    y = torch.tensor(sample["dialysis_label"], dtype=torch.float32)
    
    return X, y


def prepare_dialysis_XY_from_samples(samples, vocabs):
    """Prepare dialysis prediction training data X and Y directly from samples"""
    Xs, Ys = [], []
    for sample in samples:
        X, y = vectorize_dialysis_sample(sample, vocabs)
        Xs.append(X)
        Ys.append(y)
    return torch.stack(Xs), torch.stack(Ys)
