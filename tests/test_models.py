"""Unit tests for model training and evaluation metrics."""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from src.models import build_candidate_models, evaluate_model, ModelTrainer


@pytest.fixture
def mock_data():
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = np.zeros(200, dtype=int)
    y[::15] = 1
    return X, y


@pytest.fixture
def trained_rf(mock_data):
    X, y = mock_data
    rf = RandomForestClassifier(n_estimators=10, random_state=42, class_weight="balanced")
    rf.fit(X, y)
    return rf, X, y


# ── Candidate model building ─────────────────────────────────────────────────

def test_build_candidate_models():
    models = build_candidate_models(random_state=42)
    assert "Random Forest (Balanced)" in models
    assert "Gradient Boosting" in models
    assert "Logistic Regression (Balanced)" in models


def test_all_candidate_models_are_fitted_after_training(mock_data):
    """Every candidate model must expose predict_proba after fit."""
    X, y = mock_data
    models = build_candidate_models(random_state=42)
    for name, model in models.items():
        model.fit(X, y)
        assert hasattr(model, "predict_proba"), f"{name} missing predict_proba"
        proba = model.predict_proba(X[:5])
        assert proba.shape == (5, 2), f"{name} predict_proba shape wrong: {proba.shape}"


# ── evaluate_model ───────────────────────────────────────────────────────────

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


def test_predict_proba_output_shape_and_range(trained_rf):
    """predict_proba must return shape (n, 2) with values in [0, 1]."""
    rf, X, y = trained_rf
    proba = rf.predict_proba(X)
    assert proba.shape == (len(X), 2), f"Expected ({len(X)}, 2), got {proba.shape}"
    assert (proba >= 0).all() and (proba <= 1).all(), "Probabilities outside [0, 1]"
    assert np.allclose(proba.sum(axis=1), 1.0), "Class probabilities must sum to 1.0"


def test_threshold_sensitivity(trained_rf):
    """High threshold → fewer positives; low threshold → more positives."""
    rf, X, y = trained_rf
    metrics_strict = evaluate_model(rf, X, y, threshold=0.99)
    metrics_loose  = evaluate_model(rf, X, y, threshold=0.01)

    tp_strict = metrics_strict["confusion_matrix"]["tp"]
    tp_loose  = metrics_loose["confusion_matrix"]["tp"]
    assert tp_loose >= tp_strict, "Lower threshold should catch at least as many true positives"


def test_model_metric_values_in_valid_range(trained_rf):
    """All reported metrics must be proper fractions in [0, 1]."""
    rf, X, y = trained_rf
    metrics = evaluate_model(rf, X, y, threshold=0.50)
    for key in ["precision", "recall", "f1", "roc_auc", "pr_auc", "balanced_accuracy"]:
        val = metrics[key]
        assert 0.0 <= val <= 1.0, f"Metric '{key}' out of range: {val}"


# ── ModelTrainer ─────────────────────────────────────────────────────────────

def test_model_trainer_validation_selection(mock_data):
    X, y = mock_data
    X_train, y_train = X[:140], y[:140]
    X_val,   y_val   = X[140:], y[140:]

    trainer = ModelTrainer(random_state=42)
    best_name = trainer.train_and_validate(X_train, y_train, X_val, y_val, selection_metric="f1")
    assert best_name is not None
    assert trainer.best_model is not None
    assert best_name in trainer.validation_results


def test_model_persistence_roundtrip(mock_data):
    """Save and reload model artifact — predictions must be identical."""
    X, y = mock_data
    trainer = ModelTrainer(random_state=42)
    trainer.train_and_validate(X[:140], y[:140], X[140:], y[140:], selection_metric="f1")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_model.joblib"
        trainer.save_best_model(save_path)           # no test_results kwarg

        artifact     = ModelTrainer.load_model_artifact(save_path)
        loaded_model = artifact["model"]

    # Predictions from reloaded model must match original
    original_proba = trainer.best_model.predict_proba(X[140:])
    loaded_proba   = loaded_model.predict_proba(X[140:])
    np.testing.assert_array_almost_equal(original_proba, loaded_proba, decimal=6)
