"""Integration test: end-to-end pipeline from raw data through
model inference and explanation generation.

This is the single most important test — it exercises the full stack
exactly as the Streamlit dashboard does, without any mocks.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import RAW_DATA_PATH, RANDOM_SEED
from src.data_loader import load_raw_data
from src.preprocessor import RootIQPreprocessor
from src.models import ModelTrainer, evaluate_model
from src.explainers import (
    RootIQShapExplainer,
    RootIQLimeExplainer,
    compare_shap_and_lime_explanations,
)
from src.stability import apply_controlled_perturbation, compute_explanation_stability_metrics


# ── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipeline():
    """Run the full data → model → explain pipeline once per test module."""
    df = load_raw_data(RAW_DATA_PATH)

    preprocessor = RootIQPreprocessor(include_engineered_features=True)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.clean_and_split_data(df)

    X_train_s = preprocessor.fit_transform(X_train)
    X_val_s   = preprocessor.transform(X_val)
    X_test_s  = preprocessor.transform(X_test)

    trainer = ModelTrainer(random_state=RANDOM_SEED)
    trainer.train_and_validate(X_train_s, y_train, X_val_s, y_val, selection_metric="f1")

    return {
        "preprocessor": preprocessor,
        "trainer":       trainer,
        "model":         trainer.best_model,
        "feature_names": preprocessor.feature_names_,
        "X_train_s":     X_train_s,
        "X_test_s":      X_test_s,
        "y_test":        y_test,
    }


# ── End-to-end pipeline integrity ────────────────────────────────────────────

def test_pipeline_produces_valid_model(pipeline):
    """Best model must exist and support predict_proba."""
    model = pipeline["model"]
    assert model is not None
    assert hasattr(model, "predict_proba")


def test_pipeline_test_set_metrics_are_reasonable(pipeline):
    """Test-set F1 must be > 0.5 — sanity-check that the model actually learned."""
    model        = pipeline["model"]
    X_test_s     = pipeline["X_test_s"]
    y_test       = pipeline["y_test"]

    metrics = evaluate_model(model, X_test_s, y_test, threshold=0.50)
    assert metrics["f1"] > 0.50, (
        f"Test F1 too low ({metrics['f1']:.3f}) — model may not have learned anything useful"
    )
    assert metrics["roc_auc"] > 0.80, (
        f"ROC-AUC too low ({metrics['roc_auc']:.3f}) — model worse than expected"
    )


def test_pipeline_confusion_matrix_totals(pipeline):
    """TP+FN+TN+FP must equal the full test set size."""
    model    = pipeline["model"]
    X_test_s = pipeline["X_test_s"]
    y_test   = pipeline["y_test"]

    metrics = evaluate_model(model, X_test_s, y_test, threshold=0.50)
    cm = metrics["confusion_matrix"]
    total = cm["tp"] + cm["fn"] + cm["tn"] + cm["fp"]
    assert total == len(y_test), f"Confusion matrix totals {total} != test set size {len(y_test)}"


# ── Single-row inference (dashboard path) ────────────────────────────────────

def test_single_row_inference_failure_scenario(pipeline):
    """HDF scenario (high torque + high wear) should yield elevated failure probability."""
    preprocessor = pipeline["preprocessor"]
    model        = pipeline["model"]

    raw = {
        "Type": "L",
        "Air temperature [K]": 303.8,
        "Process temperature [K]": 313.2,
        "Rotational speed [rpm]": 1280.0,
        "Torque [Nm]": 62.0,
        "Tool wear [min]": 185.0,
    }
    scaled = preprocessor.transform(pd.DataFrame([raw]))
    prob   = float(model.predict_proba(scaled)[0, 1])
    # HDF scenario should have non-trivial failure probability
    assert prob > 0.05, f"HDF scenario probability surprisingly low: {prob:.4f}"


def test_single_row_inference_safe_scenario(pipeline):
    """Nominal safe scenario should yield very low failure probability."""
    preprocessor = pipeline["preprocessor"]
    model        = pipeline["model"]

    raw = {
        "Type": "M",
        "Air temperature [K]": 298.5,
        "Process temperature [K]": 308.7,
        "Rotational speed [rpm]": 1550.0,
        "Torque [Nm]": 38.0,
        "Tool wear [min]": 25.0,
    }
    scaled = preprocessor.transform(pd.DataFrame([raw]))
    prob   = float(model.predict_proba(scaled)[0, 1])
    assert prob < 0.50, f"Safe nominal scenario unexpectedly high failure prob: {prob:.4f}"


# ── Explainability on real model ─────────────────────────────────────────────

def test_shap_on_real_model_returns_correct_shape(pipeline):
    """SHAP explanation vector length must match number of features."""
    model        = pipeline["model"]
    X_train_s    = pipeline["X_train_s"]
    feature_names = pipeline["feature_names"]
    X_test_s     = pipeline["X_test_s"]

    shap_exp = RootIQShapExplainer(model, X_train_s, feature_names)
    exp = shap_exp.explain_instance(X_test_s[0])

    assert exp["raw_vector"].shape == (len(feature_names),)
    assert len(exp["feature_attributions"]) == len(feature_names)


def test_lime_on_real_model_returns_correct_shape(pipeline):
    """LIME explanation must cover all features."""
    model        = pipeline["model"]
    X_train_s    = pipeline["X_train_s"]
    feature_names = pipeline["feature_names"]
    X_test_s     = pipeline["X_test_s"]
    cat_indices  = [i for i, n in enumerate(feature_names) if n.startswith("Type_")]

    lime_exp = RootIQLimeExplainer(model, X_train_s, feature_names, cat_indices)
    exp = lime_exp.explain_instance(X_test_s[0], num_samples=500, random_seed=RANDOM_SEED)

    assert exp["raw_vector"].shape == (len(feature_names),)
    assert len(exp["feature_attributions"]) == len(feature_names)


def test_shap_lime_comparison_metrics_bounded(pipeline):
    """Spearman ρ ∈ [-1, 1], Jaccard ∈ [0, 1], Cosine ∈ [-1, 1]."""
    model        = pipeline["model"]
    X_train_s    = pipeline["X_train_s"]
    feature_names = pipeline["feature_names"]
    X_test_s     = pipeline["X_test_s"]
    cat_indices  = [i for i, n in enumerate(feature_names) if n.startswith("Type_")]

    shap_exp = RootIQShapExplainer(model, X_train_s, feature_names)
    lime_exp = RootIQLimeExplainer(model, X_train_s, feature_names, cat_indices)

    s_res = shap_exp.explain_instance(X_test_s[0])
    l_res = lime_exp.explain_instance(X_test_s[0], num_samples=500, random_seed=RANDOM_SEED)
    comp  = compare_shap_and_lime_explanations(s_res, l_res, top_k=3)

    assert -1.0 <= comp["spearman_rank_correlation"] <= 1.0
    assert  0.0 <= comp["top_3_jaccard_similarity"]  <= 1.0
    assert -1.0 <= comp["cosine_similarity"]          <= 1.0


# ── Perturbation stability on real model ─────────────────────────────────────

def test_shap_stable_under_small_perturbation(pipeline):
    """SHAP Spearman ρ must stay > 0.70 under 1 % perturbation."""
    model        = pipeline["model"]
    X_train_s    = pipeline["X_train_s"]
    feature_names = pipeline["feature_names"]
    preprocessor = pipeline["preprocessor"]

    raw = {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100.0,
    }
    pert_raw, _ = apply_controlled_perturbation(raw, epsilon=0.01, random_state=RANDOM_SEED)
    pert_raw["Type"] = raw["Type"]  # keep categorical unchanged

    orig_scaled = preprocessor.transform(pd.DataFrame([raw]))[0]
    pert_scaled = preprocessor.transform(pd.DataFrame([pert_raw]))[0]

    shap_exp   = RootIQShapExplainer(model, X_train_s, feature_names)
    orig_v     = shap_exp.explain_instance(orig_scaled)["raw_vector"]
    pert_v     = shap_exp.explain_instance(pert_scaled)["raw_vector"]
    metrics    = compute_explanation_stability_metrics(orig_v, pert_v, feature_names, top_k=3)

    assert metrics["spearman_correlation"] > 0.70, (
        f"SHAP unstable under 1% perturbation: ρ = {metrics['spearman_correlation']:.3f}"
    )
