"""Unit tests for model training and evaluation metrics."""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.models import build_candidate_models, evaluate_model, ModelTrainer


@pytest.fixture
def mock_data():
    np.random.seed(42)
    X = np.random.randn(200, 10)
    # Severe class imbalance: positive samples distributed across dataset
    y = np.zeros(200, dtype=int)
    y[::15] = 1
    return X, y


def test_build_candidate_models():
    models = build_candidate_models(random_state=42)
    assert "Random Forest (Balanced)" in models
    assert "Gradient Boosting" in models
    assert "Logistic Regression (Balanced)" in models


def test_evaluate_model(mock_data):
    X, y = mock_data
    rf = RandomForestClassifier(n_estimators=10, random_state=42, class_weight="balanced")
    rf.fit(X, y)

    metrics = evaluate_model(rf, X, y, threshold=0.50)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["confusion_matrix"]["tp"] + metrics["confusion_matrix"]["fn"] == int(np.sum(y))
    assert metrics["confusion_matrix"]["tn"] + metrics["confusion_matrix"]["fp"] == int(len(y) - np.sum(y))


def test_model_trainer_validation_selection(mock_data):
    X, y = mock_data
    X_train, y_train = X[:140], y[:140]
    X_val, y_val = X[140:], y[140:]

    trainer = ModelTrainer(random_state=42)
    best_name = trainer.train_and_validate(X_train, y_train, X_val, y_val, selection_metric="f1")
    assert best_name is not None
    assert trainer.best_model is not None
    assert best_name in trainer.validation_results
