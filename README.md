# Medical Diagnosis Prediction Project

The project supports MIMIC-IV and TriNetX datasets and implements two network architectures: MLP and Transformer.

## Project Structure

```
project1/
├── mimic_iv.py             # Main program for MIMIC-IV dataset
├── trinetx.py              # Main program for TriNetX dataset
├── model/                  # Model definition directory
│   ├── __init__.py
│   └── models.py          # MLP and Transformer model definitions
├── util/                   # Utility functions directory
│   ├── __init__.py
│   ├── data_processing.py  # MIMIC-IV data processing functions
│   └── data_processing_trinetx.py  # TriNetX data processing functions
├── metrics/                # Evaluation metrics directory
│   ├── __init__.py
│   └── metrics.py         # Evaluation metric calculations
├── training/               # Training functions directory
│   ├── __init__.py
│   ├── training.py        # MIMIC-IV training functions
│   └── training_trinetx.py # TriNetX training functions
├── data preview/           # Sample data previews
│   ├── mimic_iv/          # MIMIC-IV data samples
│   └── trinetx/           # TriNetX data samples
├── results.md             # My experiment results. You should ignore them.
└── README.md              # Project documentation
```
The main training function is ```train_dialysis_model_on_samples()``` in ```training_trinetx.py```.


## Usage
All command line arguments and their descriptions can be found in the `parse_args()` function in `trinetx.py` or `mimic_iv.py`.

```bash
# Train hyperbolic embedding of conditions code
python train_hyperbolic_embeddings.py
```

```bash
# Train and evaluate Transformer model on TriNetX for dialysis prediction (default settings)
python trinetx.py
# Train and evaluate Transformer model on TriNetX for dialysis prediction (using hyperbolic embedding of conditions code)
python trinetx.py --use_hyperbolic_embeddings
```
## Dependencies

### Installation
My suggestion is to create a new conda environment and install any required packages as you run the code. As a reference, my environment is built on Python 3.9.13, pytorch2.5.1+cu121.

### Data
The `--data_path` in this code is hard-coded. You should change it to the directory which stores the original TriNetX data. As a reference, my data files include:
```
ls ../data/trinetx_data/

chemo_lines.csv     datadictionary.xlsx  encounter.csv  lab_result.csv       medication_ingredient.csv  patient.csv    standardized_terminology.csv       tumor_properties.csv
cohort_details.csv  dataset_details.csv  FAQ.pdf        manifest.csv         oncology_treatment.csv     procedure.csv  trinetx_eGFR_gt20_95KPatients.zip  vitals_signs.csv
datadictionary.pdf  diagnosis.csv        genomic.csv    medication_drug.csv  patient_cohort.csv         readme.txt     tumor.csv
```