"""Visualization and reporting utility helpers for RootIQ."""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR, REPORTS_DIR

logger = logging.getLogger(__name__)

# Set clean scientific plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def plot_confusion_matrix(cm_dict: Dict[str, int], model_name: str, output_path: Path = None) -> str:
    """Render and save annotated confusion matrix."""
    if output_path is None:
        output_path = FIGURES_DIR / f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png"

    cm = np.array([
        [cm_dict["tn"], cm_dict["fp"]],
        [cm_dict["fn"], cm_dict["tp"]]
    ])

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)
    fig.colorbar(cax)

    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            label = f"{count:,}\n" + (
                "(True Normal)" if (i == 0 and j == 0) else
                "(False Failure / False Alarm)" if (i == 0 and j == 1) else
                "(Missed Failure)" if (i == 1 and j == 0) else
                "(Detected Failure)"
            )
            color = "white" if count > cm.max() / 2 else "black"
            ax.text(j, i, label, ha="center", va="center", color=color, fontweight="bold", fontsize=10)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal (0)", "Failure (1)"], fontsize=11)
    ax.set_yticklabels(["Normal (0)", "Failure (1)"], fontsize=11)
    ax.set_xlabel("Predicted Operational State", fontweight="bold", fontsize=12)
    ax.set_ylabel("True Operational State", fontweight="bold", fontsize=12)
    ax.set_title(f"Confusion Matrix: {model_name}", pad=20, fontweight="bold", fontsize=13)
    plt.tight_layout()

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_model_comparison_bar(test_results: Dict[str, Dict[str, Any]], output_path: Path = None) -> str:
    """Plot comparative benchmark metrics across candidate models."""
    if output_path is None:
        output_path = FIGURES_DIR / "model_benchmark_comparison.png"

    models = list(test_results.keys())
    metrics_to_plot = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    metric_labels = ["Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]

    x = np.arange(len(models))
    width = 0.15

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    for i, (m, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        vals = [test_results[model].get(m, 0.0) for model in models]
        ax.bar(x + (i - 2) * width, vals, width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=10, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score (0.0 to 1.0)", fontweight="bold", fontsize=11)
    ax.set_title("RootIQ Predictive Maintenance Model Benchmark Comparison (Test Set)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_explanation_stability_curves(stability_summary: Dict[str, Any], output_path: Path = None) -> str:
    """Plot stability degradation curves (Spearman rho and Top-3 Jaccard) vs. perturbation epsilon."""
    if output_path is None:
        output_path = FIGURES_DIR / "explanation_stability_curves.png"

    epsilons = [float(e) for e in stability_summary.keys()]
    eps_labels = [f"{int(e * 100)}%" for e in epsilons]

    shap_spearman = [stability_summary[str(e)]["SHAP"]["mean_spearman"] for e in epsilons]
    shap_spearman_std = [stability_summary[str(e)]["SHAP"]["std_spearman"] for e in epsilons]
    lime_spearman = [stability_summary[str(e)]["LIME"]["mean_spearman"] for e in epsilons]
    lime_spearman_std = [stability_summary[str(e)]["LIME"]["std_spearman"] for e in epsilons]

    shap_jaccard = [stability_summary[str(e)]["SHAP"]["mean_top3_jaccard"] for e in epsilons]
    lime_jaccard = [stability_summary[str(e)]["LIME"]["mean_top3_jaccard"] for e in epsilons]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # Subplot 1: Spearman Rank Correlation
    ax1.errorbar(epsilons, shap_spearman, yerr=shap_spearman_std, fmt="-o", color="#1f77b4", label="SHAP (TreeExplainer)", capsize=4, lw=2)
    ax1.errorbar(epsilons, lime_spearman, yerr=lime_spearman_std, fmt="-s", color="#ff7f0e", label="LIME (Tabular)", capsize=4, lw=2)
    ax1.set_xlabel("Input Perturbation Magnitude ($\epsilon$ relative to range)", fontweight="bold")
    ax1.set_ylabel("Spearman Rank Correlation ($\\rho$)", fontweight="bold")
    ax1.set_title("Explanation Ranking Stability Under Perturbation", fontweight="bold")
    ax1.set_xticks(epsilons)
    ax1.set_xticklabels(eps_labels)
    ax1.set_ylim(-0.1, 1.05)
    ax1.legend(loc="lower left", frameon=True)

    # Subplot 2: Top-3 Feature Overlap (Jaccard)
    ax2.plot(epsilons, shap_jaccard, "-o", color="#1f77b4", label="SHAP Top-3 Jaccard", lw=2)
    ax2.plot(epsilons, lime_jaccard, "-s", color="#ff7f0e", label="LIME Top-3 Jaccard", lw=2)
    ax2.set_xlabel("Input Perturbation Magnitude ($\epsilon$ relative to range)", fontweight="bold")
    ax2.set_ylabel("Top-3 Feature Jaccard Overlap", fontweight="bold")
    ax2.set_title("Top-3 Attributed Feature Consistency", fontweight="bold")
    ax2.set_xticks(epsilons)
    ax2.set_xticklabels(eps_labels)
    ax2.set_ylim(-0.1, 1.05)
    ax2.legend(loc="lower left", frameon=True)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_side_by_side_local_explanation(
    comparison_table: List[Dict[str, Any]],
    prediction_prob: float,
    predicted_label: int,
    output_path: Path = None
) -> str:
    """Generate high-clarity dual-XAI attribution horizontal bar chart."""
    if output_path is None:
        output_path = FIGURES_DIR / "local_explanation_comparison.png"

    # Take top 8 contributing features
    items = comparison_table[:8]
    features = [item["feature"] for item in items][::-1]
    shap_vals = [item["shap_norm"] for item in items][::-1]
    lime_vals = [item["lime_norm"] for item in items][::-1]

    y = np.arange(len(features))
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.barh(y - height / 2, shap_vals, height, label="SHAP Normalized Contribution", color="#1f77b4", alpha=0.85)
    ax.barh(y + height / 2, lime_vals, height, label="LIME Normalized Contribution", color="#ff7f0e", alpha=0.85)

    ax.axvline(0, color="gray", lw=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(features, fontweight="bold", fontsize=10)
    ax.set_xlabel("Normalized Attribution Weight ($\Delta$ Risk Direction)", fontweight="bold")
    state_str = "FAILURE IMMINENT (High Risk)" if predicted_label == 1 else "NORMAL OPERATION"
    ax.set_title(
        f"Side-by-Side Root Cause Explanation\nPredicted State: {state_str} (Failure Probability: {prediction_prob:.1%})",
        fontweight="bold",
        fontsize=12
    )
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)
