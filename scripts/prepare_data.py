"""Script to prepare and verify the raw industrial dataset."""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DATA_PATH, DATA_PROCESSED_DIR
from src.data_loader import download_dataset_if_needed, load_raw_data, audit_data_quality, save_data_quality_report

def main():
    print("Preparing dataset...")
    data_path = download_dataset_if_needed(RAW_DATA_PATH)
    print(f"Dataset path: {data_path}")
    
    df = load_raw_data(data_path)
    print(f"Loaded DataFrame with shape: {df.shape}")
    
    audit = audit_data_quality(df)
    save_data_quality_report(audit)
    print("Data quality audit completed successfully.")
    print(f"Total records: {audit['total_records']}, Failures: {audit['target_distribution']['counts'].get('1')}")

if __name__ == "__main__":
    main()
