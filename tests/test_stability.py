"""Unit tests for perturbation generation and explanation stability metrics."""

import pytest
import numpy as np

from src.config import FEATURE_CONSTRAINTS
from src.stability import (
    apply_controlled_perturbation,
    compute_explanation_stability_metrics
)


def test_controlled_perturbation_bounds():
    instance = {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100.0
    }

    # Test perturbation at 10%
    pert_dict, deltas = apply_controlled_perturbation(instance, epsilon=0.10, random_state=42)

    for feat in ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]:
        val = pert_dict[feat]
        c = FEATURE_CONSTRAINTS[feat]
        assert c["min"] <= val <= c["max"], f"Feature {feat} exceeded physical bounds: {val}"


def test_perturbation_clipping_extremes():
    # Set instance right on the maximum border
    instance = {
        "Air temperature [K]": 310.0,  # Max allowed
        "Process temperature [K]": 320.0,  # Max allowed
        "Rotational speed [rpm]": 2900.0,  # Max allowed
        "Torque [Nm]": 80.0,  # Max allowed
        "Tool wear [min]": 260.0  # Max allowed
    }

    pert_dict, _ = apply_controlled_perturbation(instance, epsilon=0.20, random_state=42)
    # Ensure clipping kept it within maximum bound
    for feat, val in pert_dict.items():
        assert val <= FEATURE_CONSTRAINTS[feat]["max"]
        assert val >= FEATURE_CONSTRAINTS[feat]["min"]


def test_stability_metrics_identical_vectors():
    v1 = np.array([0.5, 0.3, -0.2, 0.1, 0.05])
    v2 = np.copy(v1)
    feature_names = ["f1", "f2", "f3", "f4", "f5"]

    metrics = compute_explanation_stability_metrics(v1, v2, feature_names, top_k=3)
    assert np.isclose(metrics["spearman_correlation"], 1.0)
    assert np.isclose(metrics["top_3_jaccard"], 1.0)
    assert np.isclose(metrics["cosine_similarity"], 1.0)
    assert np.isclose(metrics["attribution_l1_shift"], 0.0)


def test_stability_metrics_inverted_vectors():
    v1 = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    v2 = np.array([0.2, 0.4, 0.6, 0.8, 1.0])  # Completely inverted ranking
    feature_names = ["f1", "f2", "f3", "f4", "f5"]

    metrics = compute_explanation_stability_metrics(v1, v2, feature_names, top_k=2)
    assert metrics["spearman_correlation"] < 0.0
    assert metrics["top_2_jaccard"] == 0.0
