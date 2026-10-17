"""Unit tests for SHAP, LIME, and XAI comparison."""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.explainers import (
    RootIQShapExplainer,
    RootIQLimeExplainer,
    compare_shap_and_lime_explanations
)


@pytest.fixture
def trained_model_and_data():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    feature_names = ["feat_A", "feat_B", "feat_C", "feat_D", "feat_E"]

    rf = RandomForestClassifier(n_estimators=15, max_depth=4, random_state=42)
    rf.fit(X, y)
    return rf, X, feature_names


def test_shap_explainer(trained_model_and_data):
    rf, X, feature_names = trained_model_and_data
    shap_exp = RootIQShapExplainer(rf, X, feature_names)

    instance = X[0]
    exp = shap_exp.explain_instance(instance)
    assert exp["method"] == "SHAP"
    assert len(exp["feature_attributions"]) == 5
    assert len(exp["sorted_attributions"]) == 5
    assert "raw_vector" in exp
    assert exp["raw_vector"].shape == (5,)


def test_lime_explainer(trained_model_and_data):
    rf, X, feature_names = trained_model_and_data
    lime_exp = RootIQLimeExplainer(rf, X, feature_names)

    instance = X[0]
    exp = lime_exp.explain_instance(instance, num_samples=100, random_seed=42)
    assert exp["method"] == "LIME"
    assert len(exp["feature_attributions"]) == 5
    assert len(exp["sorted_attributions"]) == 5
    assert "raw_vector" in exp
    assert exp["raw_vector"].shape == (5,)


def test_compare_shap_and_lime(trained_model_and_data):
    rf, X, feature_names = trained_model_and_data
    shap_exp = RootIQShapExplainer(rf, X, feature_names)
    lime_exp = RootIQLimeExplainer(rf, X, feature_names)

    instance = X[0]
    s_res = shap_exp.explain_instance(instance)
    l_res = lime_exp.explain_instance(instance, num_samples=100, random_seed=42)

    comp = compare_shap_and_lime_explanations(s_res, l_res, top_k=3)
    assert "spearman_rank_correlation" in comp
    assert "top_3_jaccard_similarity" in comp
    assert "cosine_similarity" in comp
    assert "comparison_table" in comp
    assert len(comp["comparison_table"]) == 5
