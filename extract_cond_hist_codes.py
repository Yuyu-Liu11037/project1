"""
Extract all codes from cond_hist in mimic4_prediction.samples and save to a .txt file
"""
import argparse
from pyhealth.datasets import MIMIC4Dataset
from util.data_processing import diag_prediction_mimic4_fn


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Extract codes from cond_hist')
    parser.add_argument('--data_path', type=str, 
                       default="/data/yuyu/data/MIMIC_IV/hosp",
                       help='MIMIC-IV data path')
    parser.add_argument('--output_file', type=str, 
                       default="cond_hist_codes.txt",
                       help='Output file path (default: cond_hist_codes.txt)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print(f"Loading MIMIC-IV dataset...")
    mimic4_base = MIMIC4Dataset(
        root=args.data_path,
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
    )
    
    print(f"Setting up diagnosis prediction task...")
    mimic4_prediction = mimic4_base.set_task(diag_prediction_mimic4_fn)
    
    print(f"Extracting codes from cond_hist...")
    all_codes = set()
    
    # Iterate through all samples
    for sample in mimic4_prediction.samples:
        cond_hist = sample.get("cond_hist", [])
        
        # cond_hist is a list of lists, where each inner list contains codes for a visit
        # The last visit is always empty to prevent leakage
        for visit_codes in cond_hist:
            if visit_codes:  # Skip empty lists
                all_codes.update(visit_codes)
    
    # Convert to sorted list for consistent output
    sorted_codes = sorted(all_codes)
    
    print(f"Found {len(sorted_codes)} unique codes")
    print(f"Saving codes to {args.output_file}...")
    
    # Save to file
    with open(args.output_file, 'w') as f:
        for code in sorted_codes:
            f.write(f"{code}\n")
    
    print(f"Done! Saved {len(sorted_codes)} codes to {args.output_file}")

