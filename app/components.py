"""UI components and styling helpers for RootIQ Streamlit Dashboard."""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def apply_custom_css():
    """Inject custom modern CSS for industrial dashboard styling."""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #0E1117;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #555;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background: #F8F9FA;
            border-left: 5px solid #0066CC;
            padding: 1.1rem;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            margin-bottom: 1rem;
        }
        .alert-card-warning {
            background: #FFF8E6;
            border-left: 5px solid #FF9900;
            padding: 1rem;
            border-radius: 6px;
            margin-bottom: 1rem;
        }
        .alert-card-danger {
            background: #FDE8E8;
            border-left: 5px solid #E02424;
            padding: 1rem;
            border-radius: 6px;
            margin-bottom: 1rem;
        }
        .alert-card-success {
            background: #EDFDF5;
            border-left: 5px solid #0E9F6E;
            padding: 1rem;
            border-radius: 6px;
            margin-bottom: 1rem;
        }
        .stButton>button {
            border-radius: 6px;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)


def render_project_badge():
    """Render project identity header badge."""
    st.markdown("""
    <div style="background-color:#EBF5FB; padding: 12px 18px; border-radius: 8px; border: 1px solid #D4E6F1; margin-bottom: 20px;">
        <span style="font-weight: bold; color: #1B4F72;">RootIQ: Explainable AI for Root Cause Analysis in Smart Manufacturing</span><br>
        <span style="font-size: 0.9em; color: #2C3E50;">
            <b>Team Diamonds</b> | Worklet #22 (Tier 3 Advanced) | Course: Applied AI for Hyper Personalized Production Systems
        </span>
    </div>
    """, unsafe_allow_html=True)


def render_risk_gauge(probability: float, threshold: float = 0.50):
    """Render a risk indicator visual for failure probability."""
    pct = probability * 100
    if probability >= threshold:
        color = "#E02424"
        badge_text = "HIGH RISK: MACHINE FAILURE PREDICTED"
        card_class = "alert-card-danger"
        recommendation = "Recommended Action: Immediate inspection of spindle bearings, lubrication system, and cutter wear."
    elif probability >= 0.25:
        color = "#FF9900"
        badge_text = "ELEVATED RISK: PRECAUTIONARY MAINTENANCE REQUIRED"
        card_class = "alert-card-warning"
        recommendation = "Recommended Action: Plan maintenance window during next scheduled batch transition."
    else:
        color = "#0E9F6E"
        badge_text = "NORMAL OPERATION: STABLE TELEMETRY"
        card_class = "alert-card-success"
        recommendation = "Recommended Action: Continue production under standard operating parameters."

    st.markdown(f"""
    <div class="{card_class}">
        <h3 style="margin-top:0; color:{color}; font-weight:bold;">{badge_text}</h3>
        <p style="font-size:1.4rem; margin:0.3rem 0; font-weight:bold;">
            Failure Probability: <span style="color:{color};">{pct:.1f}%</span> 
            <span style="font-size:0.9rem; font-weight:normal; color:#666;">(Threshold: {threshold*100:.0f}%)</span>
        </p>
        <p style="margin-bottom:0; font-size:0.95rem; color:#333;">{recommendation}</p>
    </div>
    """, unsafe_allow_html=True)


def plot_horizontal_attributions(attributions: list, title: str, color: str = "#1f77b4", max_features: int = 8):
    """Plot horizontal bar chart of feature contributions."""
    items = attributions[:max_features][::-1]
    features = [k for k, v in items]
    vals = [v for k, v in items]
    
    fig, ax = plt.subplots(figsize=(7, max(3.5, len(features) * 0.45)), dpi=120)
    bar_colors = [color if v >= 0 else "#2ca02c" for v in vals]
    
    y = np.arange(len(features))
    ax.barh(y, vals, color=bar_colors, alpha=0.85, height=0.55)
    ax.axvline(0, color="gray", lw=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(features, fontsize=9, fontweight="bold")
    ax.set_xlabel("Attribution Weight (Positive = Increases Failure Risk)", fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=10, fontweight="bold")
    plt.tight_layout()
    return fig
