"""
数据处理模块
包含数据预处理、向量化、词表构建等功能
"""
from collections import defaultdict, Counter
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from pyhealth.data import Patient
from pyhealth.medcode import CrossMap


mapping = CrossMap("ICD10CM", "CCSCM")


def diag_prediction_mimic4_fn(patient: Patient):
    """MIMIC-IV诊断预测任务的数据处理函数"""
    samples = []
    visit_ls = list(patient.visits.keys())
    
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
    
    # exclude: patients with less than 2 visit
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
    """按患者ID分组, 并按入院时间排序"""
    by_pid = defaultdict(list)
    for s in samples:
        by_pid[s["patient_id"]].append(s)
    
    for pid in by_pid:
        # 若 adm_time 是字符串，直接排序即可；如有异常可转 datetime 再排
        by_pid[pid] = sorted(by_pid[pid], key=lambda x: x["adm_time"][-1])
    
    return by_pid


def build_pairs(samples_by_pid, task="current"):
    """
    构建训练对
    task="current": 用样本自己的 conditions 做标签
    task="next":    严格按论文，特征来自第 t 次，标签用第 t+1 次的 conditions
    返回 pairs: list of (X_sample_dict, y_codes_list)
    """
    pairs = []
    for pid, seq in samples_by_pid.items():
        if task == "current":
            for s in seq:
                pairs.append((s, s["conditions"]))
        elif task == "next":
            # 至少要有 t 和 t+1
            for i in range(len(seq) - 1):
                s_t = seq[i]
                y_next = seq[i + 1]["conditions"]
                pairs.append((s_t, y_next))
        else:
            raise ValueError("task must be 'current' or 'next'")
    return pairs


def build_vocab_from_pairs(pairs):
    """从训练对构建词表"""
    diag_c, proc_c, drug_c, y_c = Counter(), Counter(), Counter(), Counter()
    
    for s, y in pairs:
        for visit_codes in s["cond_hist"]:   # 历史 ICD 诊断（最后一步为空，已防泄漏）
            diag_c.update(visit_codes)
        for visit_codes in s["procedures"]:  # 每步是一个手术码列表
            proc_c.update(visit_codes)
        for visit_codes in s["drugs"]:       # 每步是一个 ATC3 列表
            drug_c.update(visit_codes)
        y_c.update(y)                        # 标签（CCS）
    
    def mk_vocab(cnt):
        itos = [c for c, _ in cnt.most_common()]
        stoi = {c:i for i,c in enumerate(itos)}
        return stoi, itos
    
    return mk_vocab(diag_c), mk_vocab(proc_c), mk_vocab(drug_c), mk_vocab(y_c)


def multihot_from_sequence(seq_of_lists, stoi):
    """将序列转换为multi-hot向量"""
    x = torch.zeros(len(stoi), dtype=torch.float32)
    for codes in seq_of_lists:
        for c in codes:
            if c in stoi: 
                x[stoi[c]] = 1.0
    return x


def vectorize_pair(s, y_codes, vocabs, use_current_step=False):
    """将样本对向量化"""
    diag_stoi, proc_stoi, drug_stoi, y_stoi = vocabs
    
    # 入院预测(admission-time)：不看当前步的 proc/drug；出院预测(discharge-time)可看
    if use_current_step:
        proc_hist = s["procedures"]
        drug_hist = s["drugs"]
    else:
        proc_hist = s["procedures"][:-1] if len(s["procedures"])>0 else []
        drug_hist = s["drugs"][:-1] if len(s["drugs"])>0 else []

    x_diag = multihot_from_sequence(s["cond_hist"], diag_stoi)  # 历史 ICD（当前步已置空）
    x_proc = multihot_from_sequence(proc_hist, proc_stoi)
    x_drug = multihot_from_sequence(drug_hist, drug_stoi)
    X = torch.cat([x_diag, x_proc, x_drug], dim=0)

    y = torch.zeros(len(y_stoi), dtype=torch.float32)
    for c in y_codes:
        if c in y_stoi: 
            y[y_stoi[c]] = 1.0
    return X, y


def prepare_XY(pairs, vocabs, use_current_step=False):
    """准备训练数据X和Y"""
    Xs, Ys = [], []
    for s, y_codes in pairs:
        X, y = vectorize_pair(s, y_codes, vocabs, use_current_step=use_current_step)
        Xs.append(X)
        Ys.append(y)
    return torch.stack(Xs), torch.stack(Ys)


def split_by_patient(pairs, test_size=0.2, val_size=0.1, seed=42):
    """按患者ID分割数据集, 避免数据泄漏"""
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

