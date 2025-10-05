# Diagnosis Prediction Project

## Project Structure

```
project1/
├── diagnosis_prediction.py  # Main program entry point
├── model/                   # Model definition directory
│   ├── __init__.py
│   └── models.py           # MLP and Transformer model definitions
├── util/                   # Utility functions directory
│   ├── __init__.py
│   └── data_processing.py  # Data processing and preprocessing
├── metrics/                # Evaluation metrics directory
│   ├── __init__.py
│   └── metrics.py         # Evaluation metric calculations
├── training/               # Training functions directory
│   ├── __init__.py
│   └── training.py        # Training functions
└── README.md               # Project documentation
```

## Module Description

### 1. `diagnosis_prediction.py` - Main Program
- Data loading and preprocessing
- Call training functions
- Output results

### 2. `model/models.py` - Model Definitions
- `MLP`: Multi-label classification MLP model
- `TransformerModel`: Transformer-based multi-label classification model
- `create_model`: Model factory function supporting different model types

### 3. `util/data_processing.py` - Data Processing
- `diag_prediction_mimic4_fn`: MIMIC-IV data processing function
- `sort_samples_within_patient`: Sort samples within patients
- `build_pairs`: Build training pairs
- `build_vocab_from_pairs`: Build vocabulary
- `vectorize_pair`: Vectorize samples
- `prepare_XY`: Prepare training data
- `split_by_patient`: Split dataset by patient

### 4. `metrics/metrics.py` - Evaluation Metrics
- `precision_at_k_visit`: Visit-level P@k
- `accuracy_at_k_code`: Code-level Acc@k
- `recall_at_k_micro`: Micro-average recall
- `evaluate`: Comprehensive evaluation function

### 5. `training/training.py` - Training Functions
- `train_model_on_samples`: Main training function supporting multiple model types
- `train_mlp_on_samples`: Backward compatible MLP training function

## Usage

### Basic Usage
```bash
# Use default MLP model
python diagnosis_prediction.py

# Use Transformer model
python diagnosis_prediction.py --model transformer
```

### Command Line Arguments

#### Model Selection
- `--model`: Model type (`mlp` or `transformer`, default: `mlp`)

#### Training Parameters
- `--task`: Prediction task (`current` or `next`, default: `next`)
- `--use_current_step`: Whether to use current step information (default: False)
- `--hidden`: Hidden layer dimension (default: 512)
- `--lr`: Learning rate (default: 1e-3)
- `--wd`: Weight decay (default: 1e-5)
- `--epochs`: Number of training epochs (default: 10)
- `--seed`: Random seed (default: 42)

#### Transformer Specific Parameters
- `--num_heads`: Number of attention heads (default: 8)
- `--num_layers`: Number of Transformer layers (default: 3)
- `--dropout`: Dropout rate (default: 0.3)

#### Data Path
- `--data_path`: MIMIC-IV data path (default: `/data/yuyu/data/MIMIC_IV/hosp`)

### Usage Examples

```bash
# Train MLP model (default settings)
python diagnosis_prediction.py

# Train Transformer model
python diagnosis_prediction.py --model transformer --num_heads 8 --num_layers 3

# Train Transformer model with more epochs
python diagnosis_prediction.py --model transformer --epochs 20 --lr 5e-4

# Use current step information for prediction
python diagnosis_prediction.py --model transformer --use_current_step

# View all parameters
python diagnosis_prediction.py --help
```

## Dependencies

- torch
- numpy
- sklearn
- pyhealth