"""Unit tests for preprocessing, feature engineering, and leakage prevention."""

import pytest
import numpy as np
import pandas as pd

from src.config import RAW_DATA_PATH, TARGET_COL, IDENTIFIER_COLS, FAILURE_MODE_COLS
from src.data_loader import load_raw_data
from src.preprocessor import RootIQPreprocessor, engineer_domain_features


@pytest.fixture
def raw_df():
    return load_raw_data(RAW_DATA_PATH)


@pytest.fixture
def fitted_preprocessor(raw_df):
    """Return a preprocessor already fitted on training data."""
    preprocessor = RootIQPreprocessor(include_engineered_features=True)
    X_train, _, _, _, _, _ = preprocessor.clean_and_split_data(raw_df)
    preprocessor.fit_transform(X_train)
    return preprocessor


# ── Feature engineering ──────────────────────────────────────────────────────

def test_engineer_domain_features(raw_df):
    engineered_df = engineer_domain_features(raw_df)
    assert "Temp difference [K]" in engineered_df.columns
    assert "Power [kW]" in engineered_df.columns
    assert "Wear torque product [min*Nm]" in engineered_df.columns

    row0 = engineered_df.iloc[0]
    expected_temp_diff = round(row0["Process temperature [K]"] - row0["Air temperature [K]"], 2)
    assert row0["Temp difference [K]"] == expected_temp_diff

    omega = row0["Rotational speed [rpm]"] * 2.0 * np.pi / 60.0
    expected_power = round((row0["Torque [Nm]"] * omega) / 1000.0, 3)
    assert row0["Power [kW]"] == expected_power


def test_engineered_features_physically_plausible(raw_df):
    """Derived features must stay within physically meaningful ranges."""
    eng = engineer_domain_features(raw_df)
    # Temperature difference: process always > air in this dataset → ΔT > 0
    assert (eng["Temp difference [K]"] > 0).all(), "All ΔT values should be positive"
    # Mechanical power: always positive (speed and torque are positive)
    assert (eng["Power [kW]"] > 0).all(), "All power values should be positive"
    # Wear-torque product: always non-negative
    assert (eng["Wear torque product [min*Nm]"] >= 0).all()


# ── Leakage prevention & splitting ──────────────────────────────────────────

def test_leakage_prevention_and_splitting(raw_df):
    preprocessor = RootIQPreprocessor(include_engineered_features=True)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.clean_and_split_data(raw_df)

    for col in IDENTIFIER_COLS + FAILURE_MODE_COLS:
        assert col not in X_train.columns, f"Leaked column {col} found in X_train!"
        assert col not in X_val.columns,   f"Leaked column {col} found in X_val!"
        assert col not in X_test.columns,  f"Leaked column {col} found in X_test!"

    assert len(X_train) == 7000
    assert len(X_val)   == 1500
    assert len(X_test)  == 1500
    assert len(y_train) == 7000

    rate_train = y_train.mean()
    rate_val   = y_val.mean()
    rate_test  = y_test.mean()
    assert abs(rate_train - rate_val)  < 0.005
    assert abs(rate_train - rate_test) < 0.005


def test_target_col_not_in_features(raw_df):
    """Target column must never appear as a model input feature."""
    preprocessor = RootIQPreprocessor(include_engineered_features=True)
    X_train, X_val, X_test, _, _, _ = preprocessor.clean_and_split_data(raw_df)
    for split_name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        assert TARGET_COL not in X.columns, \
            f"Target column '{TARGET_COL}' found in {split_name} features — data leakage!"


# ── fit / transform pipeline ────────────────────────────────────────────────

def test_fit_transform_pipeline(raw_df):
    preprocessor = RootIQPreprocessor(include_engineered_features=True)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.clean_and_split_data(raw_df)

    X_train_trans = preprocessor.fit_transform(X_train)
    assert isinstance(X_train_trans, np.ndarray)
    assert X_train_trans.shape[0] == 7000
    assert X_train_trans.shape[1] == 11
    assert len(preprocessor.feature_names_) == 11
    assert not np.isnan(X_train_trans).any()

    X_test_trans = preprocessor.transform(X_test)
    assert X_test_trans.shape == (1500, 11)
    assert not np.isnan(X_test_trans).any()


def test_transform_scales_data(raw_df):
    """Numerical output should be approximately standardised (mean≈0, std≈1)."""
    preprocessor = RootIQPreprocessor(include_engineered_features=True)
    X_train, _, _, _, _, _ = preprocessor.clean_and_split_data(raw_df)
    X_scaled = preprocessor.fit_transform(X_train)

    col_means = X_scaled.mean(axis=0)
    col_stds  = X_scaled.std(axis=0)
    # Numerical columns only (one-hot encoded cols will be 0/1, excluded from check)
    # Check that at least half the columns have mean close to 0
    near_zero_mean = np.abs(col_means) < 0.5
    assert near_zero_mean.sum() >= 6, "Expected most columns to be near zero-mean after scaling"


def test_preprocessor_reproducibility(raw_df):
    """Same random seed must produce identical train/val/test splits."""
    p1 = RootIQPreprocessor(include_engineered_features=True)
    p2 = RootIQPreprocessor(include_engineered_features=True)
    X1, _, _, y1, _, _ = p1.clean_and_split_data(raw_df)
    X2, _, _, y2, _, _ = p2.clean_and_split_data(raw_df)
    pd.testing.assert_frame_equal(X1.reset_index(drop=True), X2.reset_index(drop=True))
    pd.testing.assert_series_equal(y1.reset_index(drop=True), y2.reset_index(drop=True))


def test_single_row_raw_transform(fitted_preprocessor):
    """The dashboard inference path: transform a single raw-format dict row."""
    raw_instance = {
        "Type": "L",
        "Air temperature [K]": 303.8,
        "Process temperature [K]": 313.2,
        "Rotational speed [rpm]": 1280.0,
        "Torque [Nm]": 62.0,
        "Tool wear [min]": 185.0,
    }
    instance_df = pd.DataFrame([raw_instance])
    scaled = fitted_preprocessor.transform(instance_df)
    assert scaled.shape == (1, 11), f"Expected shape (1, 11), got {scaled.shape}"
    assert not np.isnan(scaled).any(), "Scaled single row must contain no NaN values"
