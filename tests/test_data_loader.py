"""Unit tests for data loading and quality audit."""

import pytest
import pandas as pd
from pathlib import Path

from src.config import RAW_DATA_PATH, TARGET_COL, IDENTIFIER_COLS, FAILURE_MODE_COLS
from src.data_loader import load_raw_data, audit_data_quality, EXPECTED_COLUMNS


def test_raw_dataset_exists():
    assert RAW_DATA_PATH.exists(), f"Raw dataset not found at {RAW_DATA_PATH}"
    assert RAW_DATA_PATH.stat().st_size > 100000, "Dataset file appears too small or corrupt."


def test_load_raw_data_schema():
    df = load_raw_data(RAW_DATA_PATH)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10000, f"Expected 10,000 records, got {len(df)}"
    for col in EXPECTED_COLUMNS:
        assert col in df.columns, f"Expected column {col} missing from raw dataset."


def test_target_column_integrity():
    df = load_raw_data(RAW_DATA_PATH)
    assert TARGET_COL in df.columns
    unique_vals = set(df[TARGET_COL].unique())
    assert unique_vals.issubset({0, 1}), f"Target contains invalid classes: {unique_vals}"
    failure_count = df[TARGET_COL].sum()
    assert 300 < failure_count < 400, f"Unexpected failure count: {failure_count}"


def test_data_quality_audit():
    df = load_raw_data(RAW_DATA_PATH)
    audit = audit_data_quality(df)
    assert audit["total_records"] == 10000
    assert audit["total_missing_values"] == 0, "Dataset contains unexpected missing values."
    assert audit["duplicate_records"] == 0, "Dataset contains duplicate records."
    assert "class_imbalance_ratio" in audit
    assert audit["class_imbalance_ratio"] > 20.0
