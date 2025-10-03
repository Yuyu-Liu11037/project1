# 诊断预测项目

## 项目结构

```
project1/
├── diagnosis_prediction.py  # 主程序入口
├── model/                   # 模型定义目录
│   ├── __init__.py
│   └── models.py           # MLP模型定义
├── util/                   # 工具函数目录
│   ├── __init__.py
│   └── data_processing.py  # 数据处理和预处理
├── metrics/                # 评估指标目录
│   ├── __init__.py
│   └── metrics.py         # 评估指标计算
├── training/               # 训练函数目录
│   ├── __init__.py
│   └── training.py        # 训练函数
└── README.md               # 项目说明
```

## 模块说明

### 1. `diagnosis_prediction.py` - 主程序
- 数据加载和预处理
- 调用训练函数
- 输出结果

### 2. `model/models.py` - 模型定义
- `MLP`: 多标签分类的MLP模型

### 3. `util/data_processing.py` - 数据处理
- `diag_prediction_mimic4_fn`: MIMIC-IV数据处理函数
- `sort_samples_within_patient`: 患者内样本排序
- `build_pairs`: 构建训练对
- `build_vocab_from_pairs`: 构建词表
- `vectorize_pair`: 样本向量化
- `prepare_XY`: 准备训练数据
- `split_by_patient`: 按患者分割数据集

### 4. `metrics/metrics.py` - 评估指标
- `precision_at_k_visit`: Visit-level P@k
- `accuracy_at_k_code`: Code-level Acc@k
- `recall_at_k_micro`: 微平均召回率
- `evaluate`: 综合评估函数

### 5. `training/training.py` - 训练函数
- `train_mlp_on_samples`: 主要的训练函数

## 使用方法

```bash
python diagnosis_prediction.py
```

## 依赖

- torch
- numpy
- sklearn
- pyhealth