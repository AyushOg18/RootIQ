"""Tests for generated reports and output artefacts.

Verifies that the research pipeline has produced all expected
report files, that they are non-empty, and that key content
sections are present in each document.
"""

import json
import pytest
from pathlib import Path

from src.config import REPORTS_DIR, FIGURES_DIR, MODELS_DIR


# ── Report file existence ────────────────────────────────────────────────────

EXPECTED_REPORTS = [
    "data_quality_report.md",
    "model_comparison_report.md",
    "xai_comparison_report.md",
    "stability_study_report.md",
    "research_findings.md",
    "viva_preparation.md",
    "feature_importance_report.md",
    "metrics_summary.json",
]

@pytest.mark.parametrize("filename", EXPECTED_REPORTS)
def test_report_file_exists(filename):
    """Every expected report must be present on disk."""
    path = REPORTS_DIR / filename
    assert path.exists(), f"Missing report: {path}"


@pytest.mark.parametrize("filename", EXPECTED_REPORTS)
def test_report_file_not_empty(filename):
    """Every report must contain at least some content."""
    path = REPORTS_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} not yet generated")
    assert path.stat().st_size > 100, f"Report appears empty or near-empty: {path}"


# ── Figure existence ─────────────────────────────────────────────────────────

EXPECTED_FIGURES = [
    "confusion_matrix_gradient_boosting.png",
    "model_benchmark_comparison.png",
    "explanation_stability_curves.png",
    "local_explanation_comparison.png",
]

@pytest.mark.parametrize("figname", EXPECTED_FIGURES)
def test_figure_file_exists(figname):
    """Every expected figure must be present on disk."""
    path = FIGURES_DIR / figname
    assert path.exists(), f"Missing figure: {path}"


@pytest.mark.parametrize("figname", EXPECTED_FIGURES)
def test_figure_file_is_valid_png(figname):
    """Each figure file must start with the PNG magic bytes."""
    path = FIGURES_DIR / figname
    if not path.exists():
        pytest.skip(f"{figname} not yet generated")
    with open(path, "rb") as f:
        header = f.read(8)
    PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
    assert header == PNG_MAGIC, f"{figname} does not appear to be a valid PNG"


# ── Model artefacts ──────────────────────────────────────────────────────────

def test_best_model_artefact_exists():
    assert (MODELS_DIR / "best_model.joblib").exists(), "best_model.joblib not found"


def test_preprocessor_artefact_exists():
    assert (MODELS_DIR / "preprocessor.joblib").exists(), "preprocessor.joblib not found"


# ── metrics_summary.json content ────────────────────────────────────────────

@pytest.fixture
def metrics():
    path = REPORTS_DIR / "metrics_summary.json"
    if not path.exists():
        pytest.skip("metrics_summary.json not yet generated")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_metrics_summary_has_model_results(metrics):
    """JSON must contain a model results section with at least one entry."""
    # Actual key is 'all_models_test_results'
    assert "all_models_test_results" in metrics, \
        f"metrics_summary.json missing 'all_models_test_results'. Keys: {list(metrics.keys())}"
    assert len(metrics["all_models_test_results"]) >= 1


def test_metrics_summary_has_best_model_key(metrics):
    """JSON must record which model was selected as best."""
    # Actual key is 'best_model_name'
    assert "best_model_name" in metrics, \
        f"metrics_summary.json missing 'best_model_name'. Keys: {list(metrics.keys())}"
    assert isinstance(metrics["best_model_name"], str)
    assert len(metrics["best_model_name"]) > 0


def test_metrics_summary_has_global_shap(metrics):
    """JSON must contain global_shap_importance with at least one feature."""
    assert "global_shap_importance" in metrics
    assert len(metrics["global_shap_importance"]) >= 1


def test_metrics_summary_has_stability_data(metrics):
    """JSON must contain perturbation stability results."""
    assert "perturbation_stability_summary" in metrics
    stab = metrics["perturbation_stability_summary"]
    assert len(stab) >= 1  # At least one epsilon level


def test_metrics_best_model_f1_is_reasonable(metrics):
    """Best model's recorded test F1 must be above 0.5."""
    best    = metrics["best_model_name"]          # correct key
    results = metrics["all_models_test_results"]  # correct key
    assert best in results, f"Best model '{best}' not in all_models_test_results"
    f1 = results[best].get("f1", 0.0)
    assert f1 > 0.50, f"Recorded test F1 too low: {f1:.4f}"


# ── Markdown report content checks ──────────────────────────────────────────

def _read_report(filename):
    path = REPORTS_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} not yet generated")
    return path.read_text(encoding="utf-8")


def test_data_quality_report_mentions_10000_records():
    content = _read_report("data_quality_report.md")
    assert "10,000" in content or "10000" in content, \
        "Data quality report should mention the 10,000 record dataset size"


def test_model_comparison_report_mentions_gradient_boosting():
    content = _read_report("model_comparison_report.md")
    assert "Gradient Boosting" in content, \
        "Model comparison report should mention Gradient Boosting"


def test_xai_report_mentions_both_methods():
    content = _read_report("xai_comparison_report.md")
    assert "SHAP" in content, "XAI report must mention SHAP"
    assert "LIME" in content, "XAI report must mention LIME"


def test_stability_report_mentions_spearman():
    content = _read_report("stability_study_report.md")
    assert "Spearman" in content or "spearman" in content, \
        "Stability report should mention Spearman rank correlation"


def test_viva_report_has_multiple_questions():
    content = _read_report("viva_preparation.md")
    # Questions are formatted as "### Q1:", "### Q2:", etc.
    import re
    q_count = len(re.findall(r"###\s+Q\d+", content))
    assert q_count >= 5, f"Viva prep should have at least 5 Q&A pairs, found {q_count}"
