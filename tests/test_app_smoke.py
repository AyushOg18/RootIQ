"""Smoke test for Streamlit app modules and UI components."""

import pytest
import matplotlib.pyplot as plt
from app.components import render_risk_gauge, plot_horizontal_attributions


def test_components_plot_horizontal_attributions():
    mock_attributions = [
        ("Wear torque product", 0.45),
        ("Rotational speed", -0.30),
        ("Torque", 0.25),
        ("Air temperature", -0.10)
    ]
    fig = plot_horizontal_attributions(mock_attributions, "Test Title", color="#1f77b4")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_app_imports_cleanly():
    import app.app
    assert hasattr(app.app, "main")
