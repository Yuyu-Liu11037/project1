"""
诊断预测主程序
使用MIMIC-IV数据集进行诊断预测任务
"""
import torch
import numpy as np
import warnings
from pyhealth.datasets import MIMIC4Dataset

from util.data_processing import diag_prediction_mimic4_fn
from training.training import train_mlp_on_samples

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    torch.manual_seed(42); np.random.seed(42)

    print("Loading MIMIC-IV dataset...")
    mimic4_base = MIMIC4Dataset(
        root="/data/yuyu/data/MIMIC_IV/hosp",
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
    )

    mimic4_prediction = mimic4_base.set_task(diag_prediction_mimic4_fn)
    print(f"Total samples: {len(mimic4_prediction.samples)}")

    # 训练一个小模型跑通流程
    model, vocabs, y_itos, test_metrics = train_mlp_on_samples(
        mimic4_prediction.samples,
        task="next",             # 严格“下一次就诊”预测
        use_current_step=False,  # 入院时可用信息（不看当前处方/手术）
        hidden=512,
        lr=1e-3,
        wd=1e-5,
        epochs=10,
        seed=42,
    )
    print("[DONE] Test metrics:", test_metrics)