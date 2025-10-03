"""
训练模块
包含模型训练的主要函数
"""
import torch
import torch.nn as nn
import torch.optim as optim

from model.models import MLP
from util.data_processing import (
    sort_samples_within_patient, 
    build_pairs, 
    build_vocab_from_pairs,
    prepare_XY, 
    split_by_patient
)
from metrics.metrics import bce_pos_weight, evaluate


def train_mlp_on_samples(samples,
                         task="next",          # "next" 对齐论文；"current" 用已有标注
                         use_current_step=False, # 入院预测(False) or 出院预测(True)
                         hidden=512, lr=1e-3, wd=1e-5,
                         epochs=10, seed=42):
    """
    训练MLP模型进行诊断预测
    
    Args:
        samples: 样本数据
        task: "next" 或 "current"，预测任务类型
        use_current_step: 是否使用当前步骤的信息
        hidden: 隐藏层维度
        lr: 学习率
        wd: 权重衰减
        epochs: 训练轮数
        seed: 随机种子
    
    Returns:
        model: 训练好的模型
        vocabs: 词表字典
        y_itos: 标签索引到字符串的映射
        test_metrics: 测试集评估指标
    """
    # 1) 排序 + 组装 (current/next)
    # print("Samples:")
    # print(f"{samples[0]}\n")
    by_pid = sort_samples_within_patient(samples)
    # for i in range(3):
    #     print(by_pid[i])
    # print("\n")
    pairs = build_pairs(by_pid, task=task)
    # for i in range(3):
    #     print(pairs[i])
    # print("\n")
    # exit()

    # 2) 患者级划分
    train_pairs, val_pairs, test_pairs = split_by_patient(pairs, seed=seed)

    # 3) 词表
    (diag_stoi,_), (proc_stoi,_), (drug_stoi,_), (y_stoi, y_itos) = build_vocab_from_pairs(train_pairs)
    vocabs = (diag_stoi, proc_stoi, drug_stoi, y_stoi)

    # 4) 向量化
    Xtr, Ytr = prepare_XY(train_pairs, vocabs, use_current_step=use_current_step)
    Xva, Yva = prepare_XY(val_pairs,   vocabs, use_current_step=use_current_step)
    Xte, Yte = prepare_XY(test_pairs,   vocabs, use_current_step=use_current_step)

    # 5) 模型与损失（多标签）
    # input: (batch_size, in_dim), in_dim = 历史诊断码 (ICD)len(diag_stoi) + 历史手术码 (Procedures)len(proc_stoi) + 历史药物码 (ATC-3)len(drug_stoi)
    # output: (batch_size, out_dim), out_dim = 标签码 (CCS)len(y_stoi),  标签词表的大小
    model = MLP(Xtr.size(1), hidden=hidden, out_dim=Ytr.size(1))
    pw = bce_pos_weight(Ytr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # 6) 训练
    for ep in range(1, epochs+1):
        model.train()
        logits = model(Xtr)
        loss = criterion(logits, Ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep % 1 == 0:
            mtr = evaluate(model, Xva, Yva, ks=(10, 20))
            print(f"Epoch {ep:02d} | loss={loss.item():.4f} | "
                  f"val P@10={mtr['P@10']:.4f} Acc@10={mtr['Acc@10']:.4f} "
                  f"P@20={mtr['P@20']:.4f} Acc@20={mtr['Acc@20']:.4f}")

    # 7) 测试集评估（与论文一致：Visit-level P@k、Code-level Acc@k）
    test_metrics = evaluate(model, Xte, Yte, ks=(10, 20))
    print("[TEST]", test_metrics)
    
    return model, vocabs, y_itos, test_metrics

