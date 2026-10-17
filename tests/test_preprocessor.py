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


def test_engineer_domain_features(raw_df):
    engineered_df = engineer_domain_features(raw_df)
    assert "Temp difference [K]" in engineered_df.columns
    assert "Power [kW]" in engineered_df.columns
    assert "Wear torque product [min*Nm]" in engineered_df.columns

    # Verify physical calculations for first row
    row0 = engineered_df.iloc[0]
    expected_temp_diff = round(row0["Process temperature [K]"] - row0["Air temperature [K]"], 2)
    assert row0["Temp difference [K]"] == expected_temp_diff

    omega = row0["Rotational speed [rpm]"] * 2.0 * np.pi / 60.0
    expected_power = round((row0["Torque [Nm]"] * omega) / 1000.0, 3)
    assert row0["Power [kW]"] == expected_power


def test_leakage_prevention_and_splitting(raw_df):
    preprocessor = RootIQPreprocessor(include_engineered_features=True)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.clean_and_split_data(raw_df)

    # Verify identifiers and leakage flags are excluded from features
    for col in IDENTIFIER_COLS + FAILURE_MODE_COLS:
        assert col not in X_train.columns, f"Leaked column {col} found in X_train!"
        assert col not in X_val.columns, f"Leaked column {col} found in X_val!"
        assert col not in X_test.columns, f"Leaked column {col} found in X_test!"

    # Verify split sizes (70% train, 15% val, 15% test)
    assert len(X_train) == 7000
    assert len(X_val) == 1500
    assert len(X_test) == 1500
    assert len(y_train) == 7000

    # Verify stratification (failure rate roughly identical across splits)
    rate_train = y_train.mean()
    rate_val = y_val.mean()
    rate_test = y_test.mean()
    assert abs(rate_train - rate_val) < 0.005
    assert abs(rate_train - rate_test) < 0.005


def test_fit_transform_pipeline(raw_df):
    preprocessor = RootIQPreprocessor(include_engineered_features=True)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.clean_and_split_data(raw_df)

    X_train_trans = preprocessor.fit_transform(X_train)
    assert isinstance(X_train_trans, np.ndarray)
    assert X_train_trans.shape[0] == 7000
    # 5 base numerical + 3 engineered + 3 one-hot categories (Type_L, Type_M, Type_H) = 11 features
    assert X_train_trans.shape[1] == 11
    assert len(preprocessor.feature_names_) == 11

    # Verify no NaN values
    assert not np.isnan(X_train_trans).any()

    # Transform test set
    X_test_trans = preprocessor.transform(X_test)
    assert X_test_trans.shape == (1500, 11)
    assert not np.isnan(X_test_trans).any()
