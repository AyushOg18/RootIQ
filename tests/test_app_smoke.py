"""Smoke test for Streamlit app modules and UI components."""

import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Mock streamlit before importing components that call st.*
import sys
import types
_st = types.ModuleType("streamlit")
for _attr in [
    "markdown", "metric", "set_page_config", "sidebar", "session_state",
    "cache_data", "cache_resource", "experimental_rerun", "spinner",
    "success", "warning", "info", "error", "button", "expander",
    "columns", "tabs", "dataframe", "pyplot", "image", "caption",
    "download_button", "selectbox", "slider", "radio", "text_input",
    "bar_chart", "line_chart", "stop",
]:
    setattr(_st, _attr, lambda *a, **kw: None)
_st.session_state = {}
# sidebar needs sub-attributes
_st.sidebar = types.SimpleNamespace(
    title=lambda *a, **kw: None,
    markdown=lambda *a, **kw: None,
    radio=lambda *a, **kw: None,
    info=lambda *a, **kw: None,
)
sys.modules.setdefault("streamlit", _st)

from app.components import (
    render_risk_gauge,
    plot_horizontal_attributions,
    apply_custom_css,
    render_project_badge,
)


# ── plot_horizontal_attributions ────────────────────────────────────────────

def test_components_plot_horizontal_attributions():
    """Standard attribution list renders a Figure."""
    mock_attributions = [
        ("Wear torque product", 0.45),
        ("Rotational speed", -0.30),
        ("Torque", 0.25),
        ("Air temperature", -0.10),
    ]
    fig = plot_horizontal_attributions(mock_attributions, "Test Title", color="#1f77b4")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_horizontal_attributions_single_item():
    """Single-feature attribution must not crash."""
    fig = plot_horizontal_attributions([("Tool wear [min]", 0.9)], "Single Feature")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_horizontal_attributions_all_negative():
    """All-negative attribution values (risk-reducing features) must render."""
    attrs = [("f1", -0.5), ("f2", -0.3), ("f3", -0.1)]
    fig = plot_horizontal_attributions(attrs, "All Negative")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ── render_risk_gauge ────────────────────────────────────────────────────────

def test_risk_gauge_normal_zone():
    """Probability below 25 % — no exception raised."""
    render_risk_gauge(0.10, threshold=0.50)


def test_risk_gauge_elevated_zone():
    """Probability between 25 % and 50 % — elevated warning zone."""
    render_risk_gauge(0.35, threshold=0.50)


def test_risk_gauge_failure_zone():
    """Probability at exactly the threshold — danger zone triggered."""
    render_risk_gauge(0.50, threshold=0.50)


def test_risk_gauge_certainty():
    """Probability = 1.0 (100 % certain failure) — must not crash."""
    render_risk_gauge(1.0, threshold=0.50)


def test_risk_gauge_zero():
    """Probability = 0.0 (perfectly safe) — must not crash."""
    render_risk_gauge(0.0, threshold=0.50)


# ── apply_custom_css / render_project_badge ──────────────────────────────────

def test_apply_custom_css_does_not_raise():
    """CSS injector must not raise even outside a live Streamlit server."""
    apply_custom_css()   # returns None, just checks no exception


def test_render_project_badge_does_not_raise():
    """Badge renderer must not raise outside a live Streamlit server."""
    render_project_badge()


# ── app module structure ─────────────────────────────────────────────────────

def test_app_imports_cleanly():
    import app.app
    assert hasattr(app.app, "main")


def test_app_has_cached_loaders():
    """Dashboard must expose both cached loader functions."""
    import app.app
    assert hasattr(app.app, "get_cached_raw_data")
    assert hasattr(app.app, "get_cached_model_and_preprocessor")
