"""Preprocessing and Feature Engineering pipeline for RootIQ."""

import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.config import (
    RANDOM_SEED,
    TARGET_COL,
    IDENTIFIER_COLS,
    FAILURE_MODE_COLS,
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    DATA_PROCESSED_DIR,
    MODELS_DIR
)

logger = logging.getLogger(__name__)


def engineer_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer physically meaningful industrial domain features based on machine dynamics.
    
    1. Temp difference [K] = Process temperature - Air temperature
       (Key indicator for Heat Dissipation Failure - HDF)
    2. Power [kW] = Torque * Rotational speed * 2 * pi / 60 / 1000
       (Key indicator for Power Failure - PWF)
    3. Tool wear torque strain [min*Nm] = Tool wear * Torque
       (Key indicator for Overstrain Failure - OSF and Tool Wear Failure - TWF)
    """
    df = df.copy()
    
    # Temperature difference (Delta T in Kelvin)
    df["Temp difference [K]"] = (
        df["Process temperature [K]"] - df["Air temperature [K]"]
    ).round(2)
    
    # Mechanical Power output in kW: P = tau * omega
    # omega (rad/s) = rpm * 2 * pi / 60
    omega = df["Rotational speed [rpm]"] * 2.0 * np.pi / 60.0
    df["Power [kW]"] = ((df["Torque [Nm]"] * omega) / 1000.0).round(3)
    
    # Wear-Torque product (Strain metric)
    df["Wear torque product [min*Nm]"] = (
        df["Tool wear [min]"] * df["Torque [Nm]"]
    ).round(2)
    
    return df


class RootIQPreprocessor:
    """Preprocessing pipeline for RootIQ: handles scaling, categorical encoding,
    and metadata extraction without data leakage.
    """
    
    def __init__(self, include_engineered_features: bool = True):
        self.include_engineered_features = include_engineered_features
        self.preprocessor: ColumnTransformer = None
        self.feature_names_: List[str] = []
        self.raw_numerical_cols: List[str] = list(NUMERICAL_COLS)
        self.raw_categorical_cols: List[str] = list(CATEGORICAL_COLS)
        
    def _get_feature_lists(self) -> Tuple[List[str], List[str]]:
        num_cols = list(self.raw_numerical_cols)
        if self.include_engineered_features:
            num_cols.extend([
                "Temp difference [K]",
                "Power [kW]",
                "Wear torque product [min*Nm]"
            ])
        cat_cols = list(self.raw_categorical_cols)
        return num_cols, cat_cols

    def clean_and_split_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Drop identifiers and leakage flags, engineer domain features,
        and perform stratified train/val/test splitting.
        """
        # Exclude identifiers and leakage features
        cols_to_drop = [c for c in IDENTIFIER_COLS + FAILURE_MODE_COLS if c in df.columns]
        cleaned_df = df.drop(columns=cols_to_drop)
        
        if self.include_engineered_features:
            cleaned_df = engineer_domain_features(cleaned_df)
            
        X = cleaned_df.drop(columns=[TARGET_COL])
        y = cleaned_df[TARGET_COL]
        
        # Step 1: Split into Train (70%) and Temp (30%)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=(VAL_RATIO + TEST_RATIO),
            stratify=y,
            random_state=RANDOM_SEED
        )
        
        # Step 2: Split Temp into Validation (15% overall) and Test (15% overall)
        relative_test_size = TEST_RATIO / (VAL_RATIO + TEST_RATIO)  # 0.50
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=relative_test_size,
            stratify=y_temp,
            random_state=RANDOM_SEED
        )
        
        logger.info(
            f"Dataset split: Train={X_train.shape[0]} (Pos: {y_train.sum()}), "
            f"Val={X_val.shape[0]} (Pos: {y_val.sum()}), "
            f"Test={X_test.shape[0]} (Pos: {y_test.sum()})"
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def fit(self, X_train: pd.DataFrame) -> "RootIQPreprocessor":
        """Fit ColumnTransformer on training features only."""
        num_cols, cat_cols = self._get_feature_lists()
        
        numeric_transformer = Pipeline(steps=[
            ("scaler", StandardScaler())
        ])
        
        try:
            onehot_encoder = OneHotEncoder(categories=[["L", "M", "H"]], handle_unknown="ignore", sparse=False)
        except TypeError:
            onehot_encoder = OneHotEncoder(categories=[["L", "M", "H"]], handle_unknown="ignore", sparse_output=False)

        categorical_transformer = Pipeline(steps=[
            ("onehot", onehot_encoder)
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_cols),
                ("cat", categorical_transformer, cat_cols)
            ]
        )
        
        self.preprocessor.fit(X_train)
        
        # Determine output feature names
        cat_encoder = self.preprocessor.named_transformers_["cat"].named_steps["onehot"]
        encoded_cat_names = [f"Type_{cat}" for cat in cat_encoder.categories_[0]]
        self.feature_names_ = num_cols + encoded_cat_names
        
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform raw or engineered features using fitted pipeline."""
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")
        X_df = X.copy()
        if self.include_engineered_features:
            missing_eng = [col for col in ["Temp difference [K]", "Power [kW]", "Wear torque product [min*Nm]"] if col not in X_df.columns]
            if missing_eng:
                X_df = engineer_domain_features(X_df)
        return self.preprocessor.transform(X_df)

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """Fit and transform training features."""
        return self.fit(X_train).transform(X_train)

    def transform_to_df(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform input DataFrame to DataFrame with named columns."""
        transformed_arr = self.transform(X)
        return pd.DataFrame(transformed_arr, columns=self.feature_names_, index=X.index)

    def save(self, filepath: Path = MODELS_DIR / "preprocessor.joblib"):
        """Save fitted preprocessor to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Preprocessor saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path = MODELS_DIR / "preprocessor.joblib") -> "RootIQPreprocessor":
        """Load fitted preprocessor from disk."""
        return joblib.load(filepath)
