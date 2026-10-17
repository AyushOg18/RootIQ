"""Machine learning models and evaluation module for RootIQ."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score
)

from src.config import RANDOM_SEED, MODELS_DIR, REPORTS_DIR

logger = logging.getLogger(__name__)


def build_candidate_models(random_state: int = RANDOM_SEED) -> Dict[str, Any]:
    """Instantiate candidate predictive maintenance classification models."""
    models = {
        "Random Forest (Balanced)": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.85,
            random_state=random_state
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.08,
            max_depth=6,
            random_state=random_state
        ),
        "Logistic Regression (Balanced)": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state
        )
    }
    return models


def evaluate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics including class-imbalance metrics."""
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        # Scale decision function to [0, 1] via sigmoid
        scores = model.decision_function(X)
        y_prob = 1.0 / (1.0 + np.exp(-scores))
    else:
        y_prob = model.predict(X).astype(float)
        
    y_pred = (y_prob >= threshold).astype(int)
    
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Specificity = TN / (TN + FP)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    # False Positive Rate = FP / (FP + TN)
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    
    # False Negative Rate (Missed failures) = FN / (FN + TP)
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    roc_auc = float(roc_auc_score(y, y_prob)) if len(np.unique(y)) > 1 else 0.0
    pr_auc = float(average_precision_score(y, y_prob)) if len(np.unique(y)) > 1 else 0.0

    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "support": {
            "total": int(len(y)),
            "positive_failures": int(np.sum(y)),
            "negative_normal": int(len(y) - np.sum(y))
        }
    }
    return metrics


class ModelTrainer:
    """Trainer and model selection manager based on validation metrics."""

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.models: Dict[str, Any] = build_candidate_models(random_state)
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[Any] = None
        self.validation_results: Dict[str, Dict[str, Any]] = {}
        self.test_results: Dict[str, Dict[str, Any]] = {}

    def train_and_validate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        selection_metric: str = "f1"
    ) -> str:
        """Train all candidate models on train split, evaluate on validation split,
        and select best model using strictly validation criteria.
        """
        logger.info("Training candidate models...")
        best_score = -1.0
        
        for name, model in self.models.items():
            logger.info(f"Fitting {name}...")
            model.fit(X_train, y_train)
            
            val_metrics = evaluate_model(model, X_val, y_val)
            self.validation_results[name] = val_metrics
            score = val_metrics.get(selection_metric, 0.0)
            
            logger.info(f"{name} -> Validation {selection_metric.upper()}: {score:.4f}, PR-AUC: {val_metrics['pr_auc']:.4f}")
            
            # Select model with highest validation score (tiebreaker: PR-AUC)
            if score > best_score or (np.isclose(score, best_score) and val_metrics["pr_auc"] > self.validation_results.get(self.best_model_name, {}).get("pr_auc", -1)):
                best_score = score
                self.best_model_name = name
                self.best_model = model

        logger.info(f"Selected best model: {self.best_model_name} with Validation {selection_metric.upper()} = {best_score:.4f}")
        return self.best_model_name

    def evaluate_test_set(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate all trained models on unseen test split."""
        for name, model in self.models.items():
            test_metrics = evaluate_model(model, X_test, y_test)
            self.test_results[name] = test_metrics
            logger.info(
                f"Test Results [{name}] -> Precision: {test_metrics['precision']:.4f}, "
                f"Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}, "
                f"ROC-AUC: {test_metrics['roc_auc']:.4f}, PR-AUC: {test_metrics['pr_auc']:.4f}"
            )
        return self.test_results

    def save_best_model(self, filepath: Path = MODELS_DIR / "best_model.joblib"):
        """Save the chosen best model to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model_name": self.best_model_name,
            "model": self.best_model,
            "validation_results": self.validation_results,
            "test_results": self.test_results,
            "random_seed": self.random_state
        }
        joblib.dump(artifact, filepath)
        logger.info(f"Saved best model ({self.best_model_name}) to {filepath}")

    @classmethod
    def load_model_artifact(cls, filepath: Path = MODELS_DIR / "best_model.joblib") -> Dict[str, Any]:
        """Load trained model artifact dictionary from disk."""
        return joblib.load(filepath)
