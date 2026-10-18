"""RootIQ: Explainable AI for Root Cause Analysis in Smart Manufacturing
Streamlit Interactive Dashboard Application.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ensure project root and app/ dir are in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(APP_DIR))

from src.config import (
    RAW_DATA_PATH,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    FEATURE_CONSTRAINTS,
    TARGET_COL,
    RANDOM_SEED
)
from src.data_loader import load_raw_data, audit_data_quality
from src.preprocessor import RootIQPreprocessor, engineer_domain_features
from src.models import ModelTrainer, evaluate_model
from src.explainers import RootIQShapExplainer, RootIQLimeExplainer, compare_shap_and_lime_explanations
from src.stability import apply_controlled_perturbation, compute_explanation_stability_metrics
from components import (
    apply_custom_css,
    render_project_badge,
    render_risk_gauge,
    plot_horizontal_attributions
)

st.set_page_config(
    page_title="RootIQ — Explainable AI in Smart Manufacturing",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()


@st.cache_data
def get_cached_raw_data():
    """Load and cache raw dataset."""
    if not RAW_DATA_PATH.exists():
        return None
    return load_raw_data(RAW_DATA_PATH)


def _train_pipeline_from_scratch():
    """Train the full pipeline from raw data. Used when joblib artifacts
    cannot be loaded (e.g. cross-Python-version pickle incompatibility on cloud)."""
    from src.models import ModelTrainer, evaluate_model
    import numpy as np

    df = load_raw_data(RAW_DATA_PATH)
    preprocessor = RootIQPreprocessor(include_engineered_features=True)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.clean_and_split_data(df)

    X_train_s = preprocessor.fit_transform(X_train)
    X_val_s   = preprocessor.transform(X_val)
    X_test_s  = preprocessor.transform(X_test)

    trainer = ModelTrainer(random_state=RANDOM_SEED)
    best_name = trainer.train_and_validate(X_train_s, y_train, X_val_s, y_val, selection_metric="f1")
    test_results = trainer.evaluate_test_set(X_test_s, y_test.values if hasattr(y_test, "values") else y_test)

    trainer.save_best_model()
    preprocessor.save()

    # Save processed splits
    np.save(DATA_PROCESSED_DIR / "X_train_scaled.npy", X_train_s)
    np.save(DATA_PROCESSED_DIR / "X_val_scaled.npy",   X_val_s)
    np.save(DATA_PROCESSED_DIR / "X_test_scaled.npy",  X_test_s)

    model_artifact = ModelTrainer.load_model_artifact(MODELS_DIR / "best_model.joblib")
    return model_artifact, preprocessor


@st.cache_resource
def get_cached_model_and_preprocessor():
    """Load model artifact and fitted preprocessor.
    Falls back to retraining if joblib files are missing or
    incompatible (e.g. pickled on a different Python version).
    """
    model_path = MODELS_DIR / "best_model.joblib"
    prep_path  = MODELS_DIR / "preprocessor.joblib"

    if model_path.exists() and prep_path.exists():
        try:
            model_artifact = ModelTrainer.load_model_artifact(model_path)
            preprocessor   = RootIQPreprocessor.load(prep_path)
            return model_artifact, preprocessor
        except Exception:
            # Cross-Python-version pickle error — retrain from scratch
            pass

    # Artifacts missing or unloadable — train now
    return _train_pipeline_from_scratch()


def main():
    render_project_badge()

    # Sidebar Navigation
    st.sidebar.title("RootIQ Platform")
    st.sidebar.markdown("**Explainable Root Cause Analysis**")

    pages = [
        "1. Executive Overview",
        "2. Dataset Exploration & Quality",
        "3. Model Performance & Matrix",
        "4. Machine Operational Simulator",
        "5. Failure Probability & Decision",
        "6. SHAP Local & Global Explanations",
        "7. LIME Local Explanation",
        "8. Side-by-Side Dual-XAI Comparison",
        "9. Explanation-Stability Lab",
        "10. Research Findings & Reports"
    ]

    selected_page = st.sidebar.radio("Navigate Module:", pages)

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**System Notice:** Machine predictions and explanations highlight diagnostic correlations "
        "and risk trends. They should be corroborated with physical sensor inspection."
    )

    df_raw = get_cached_raw_data()

    if df_raw is None:
        st.error("Dataset not found at `data/raw/ai4i2020.csv`. Please ensure the dataset is committed to the repository.")
        return

    with st.spinner("Loading model (first run may take ~60 seconds to train)..."):
        model_artifact, preprocessor = get_cached_model_and_preprocessor()

    if model_artifact is None:
        st.error("Model could not be loaded or trained. Check the logs for details.")
        return

    best_model = model_artifact["model"]
    model_name = model_artifact["model_name"]
    test_results = model_artifact.get("test_results", {})
    feature_names = preprocessor.feature_names_

    # ── Global Persistent Machine State Initialization ──────────────────────
    if "sim_type" not in st.session_state:
        st.session_state["sim_type"] = "M"
        st.session_state["sim_air_temp"] = 300.0
        st.session_state["sim_proc_temp"] = 310.0
        st.session_state["sim_speed"] = 1500.0
        st.session_state["sim_torque"] = 40.0
        st.session_state["sim_wear"] = 50.0

    if "current_raw_instance" not in st.session_state:
        init_raw = {
            "Type": st.session_state["sim_type"],
            "Air temperature [K]": float(st.session_state["sim_air_temp"]),
            "Process temperature [K]": float(st.session_state["sim_proc_temp"]),
            "Rotational speed [rpm]": float(st.session_state["sim_speed"]),
            "Torque [Nm]": float(st.session_state["sim_torque"]),
            "Tool wear [min]": float(st.session_state["sim_wear"])
        }
        init_scaled = preprocessor.transform(pd.DataFrame([init_raw]))
        st.session_state["current_raw_instance"] = init_raw
        st.session_state["current_scaled_instance"] = init_scaled[0]
        st.session_state["current_prob"] = float(best_model.predict_proba(init_scaled)[0, 1])

    # -------------------------------------------------------------------------
    # PAGE 1: EXECUTIVE OVERVIEW
    # -------------------------------------------------------------------------
    if selected_page == "1. Executive Overview":
        st.markdown("<h2 class='main-header'>RootIQ: Transparent Predictive Maintenance</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p class='sub-header'>Explainable AI (XAI) for real-time root cause analysis and diagnosis of CNC milling failures.</p>",
            unsafe_allow_html=True
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Production Fleet Dataset", f"{len(df_raw):,} records")
        col2.metric("Failure Incidence Rate", f"{(df_raw[TARGET_COL].mean()*100):.2f}%")
        col3.metric("Selected Model", model_name.split()[0])
        col4.metric("Test Set F1-Score", f"{test_results.get(model_name, {}).get('f1', 0.0):.3f}")

        st.markdown("### The Problem: The Industrial Trust Deficit")
        st.write(
            "In modern Industry 4.0 production lines, black-box machine learning models accurately predict equipment "
            "failures, but process engineers resist acting on uninterpretable alarms. Shutting down a high-throughput milling "
            "cell based on an unexplained alert costs thousands of dollars per hour. RootIQ bridges this trust gap by coupling "
            "predictive machine failure classification with rigorous, dual-method Explainable AI (SHAP and LIME) and evaluating "
            "their stability under operational noise."
        )

        st.markdown("### Research Architecture & Methodology")
        st.markdown(r"""
        1. **Verified Telemetry Dataset:** High-fidelity milling machine telemetry from UCI AI4I 2020 Predictive Maintenance dataset.
        2. **Physics-Informed Preprocessing:** Target leakage eliminated by removing deterministic failure flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) and identifiers (`UDI`, `Product ID`). Thermodynamic and mechanical power features engineered from first principles.
        3. **Class-Aware Predictive Modeling:** Candidate models (Random Forest, Gradient Boosting, HistGradientBoosting, Logistic Regression) optimized under severe class imbalance (3.4% failure rate).
        4. **Dual-Method Explainability:** Simultaneous evaluation of **SHAP** (exact Shapley cooperative game values) and **LIME** (sparse local surrogate linear models).
        5. **Empirical Stability Analysis:** Controlled physical input perturbation testing ($\epsilon \in [1\%, 5\%, 10\%, 20\%]$) and LIME stochastic seed variance quantification.
        """)

    # -------------------------------------------------------------------------
    # PAGE 2: DATASET EXPLORATION & QUALITY
    # -------------------------------------------------------------------------
    elif selected_page == "2. Dataset Exploration & Quality":
        st.markdown("<h2 class='main-header'>Dataset Exploration & Data Quality Audit</h2>", unsafe_allow_html=True)
        st.markdown("<p class='sub-header'>Inspection of schema, distributions, and class imbalance.</p>", unsafe_allow_html=True)

        audit = audit_data_quality(df_raw)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Telemetry Rows", f"{audit['total_records']:,}")
        col2.metric("Normal Operating States", f"{audit['target_distribution']['counts'].get('0', 0):,}")
        col3.metric("Catastrophic Failures", f"{audit['target_distribution']['counts'].get('1', 0):,}")
        col4.metric("Imbalance Ratio", f"{audit['class_imbalance_ratio']}:1")

        tab_data, tab_stats, tab_failures = st.tabs(["Raw Telemetry (All 10,000 Rows)", "Numerical Distributions", "Failure Mode Breakdown"])
        
        with tab_data:
            st.caption(f"Showing all {len(df_raw):,} telemetry records — scroll to explore the full dataset.")
            st.dataframe(df_raw, use_container_width=True, height=500)

        with tab_stats:
            num_cols = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
            st.dataframe(df_raw[num_cols].describe(), use_container_width=True)

        with tab_failures:
            st.markdown("#### Distribution of Physical Failure Modes (Post-Hoc Modes in Raw Data)")
            breakdown = audit["failure_modes_breakdown"]
            f_df = pd.DataFrame(list(breakdown.items()), columns=["Failure Mode Code", "Occurrence Count"])
            f_df["Description"] = [
                "Tool Wear Failure (TWF)",
                "Heat Dissipation Failure (HDF)",
                "Power Failure (PWF)",
                "Overstrain Failure (OSF)",
                "Random Failure (RNF)"
            ]
            st.dataframe(f_df, use_container_width=True)

    # -------------------------------------------------------------------------
    # PAGE 3: MODEL PERFORMANCE & BENCHMARK MATRIX
    # -------------------------------------------------------------------------
    elif selected_page == "3. Model Performance & Matrix":
        st.markdown("<h2 class='main-header'>Predictive Model Benchmark Matrix</h2>", unsafe_allow_html=True)
        st.markdown("<p class='sub-header'>Evaluation across class-imbalance metrics on the held-out test set (15% split).</p>", unsafe_allow_html=True)

        # Performance table
        if not test_results or model_name not in test_results:
            # Fallback evaluate on test set if missing
            X_test_scaled_path = DATA_PROCESSED_DIR / "X_test_scaled.npy"
            y_test_path = DATA_PROCESSED_DIR / "y_test.csv"
            if X_test_scaled_path.exists() and y_test_path.exists():
                X_t = np.load(X_test_scaled_path)
                y_t = pd.read_csv(y_test_path).iloc[:, 0].values
                test_results = {model_name: evaluate_model(best_model, X_t, y_t)}
            else:
                test_results = {}

        rows = []
        for name, metrics in test_results.items():
            rows.append({
                "Model Architecture": ("⭐ " + name) if name == model_name else name,
                "Precision": f"{metrics['precision']:.4f}",
                "Recall": f"{metrics['recall']:.4f}",
                "F1-Score": f"{metrics['f1']:.4f}",
                "PR-AUC": f"{metrics['pr_auc']:.4f}",
                "ROC-AUC": f"{metrics['roc_auc']:.4f}",
                "Balanced Accuracy": f"{metrics['balanced_accuracy']:.4f}"
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        col1, col2 = st.columns(2)
        target_metrics = test_results.get(model_name)
        if target_metrics and "confusion_matrix" in target_metrics:
            cm_dict = target_metrics["confusion_matrix"]
            with col1:
                st.markdown("#### Selected Model Confusion Matrix")
                fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
                cm = np.array([[cm_dict["tn"], cm_dict["fp"]], [cm_dict["fn"], cm_dict["tp"]]])
                cax = ax.matshow(cm, cmap="Blues", alpha=0.85)
                fig.colorbar(cax)
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black", fontweight="bold")
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(["Normal (0)", "Failure (1)"])
                ax.set_yticklabels(["Normal (0)", "Failure (1)"])
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True Ground Truth")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with col2:
                st.markdown("#### Operational Trade-Off Analysis")
                fnr = target_metrics.get("fnr", 0.0)
                fpr = target_metrics.get("fpr", 0.0)
                st.markdown(f"""
                - **Missed Failure Rate (FNR):** `{fnr:.2%}`  
                  *(Percentage of true machine failures that slipped past the model)*
                - **False Alarm Rate (FPR):** `{fpr:.2%}`  
                  *(Percentage of normal cycles that triggered an unnecessary alert)*
                - **Selection Justification:** `{model_name}` achieved optimal F1 balance on the validation set, preventing excessive false shutdowns while catching critical mechanical failure risks.
                """)

    # -------------------------------------------------------------------------
    # PAGE 4: MACHINE OPERATIONAL SIMULATOR (input only)
    # -------------------------------------------------------------------------
    elif selected_page == "4. Machine Operational Simulator":
        st.markdown("<h2 class='main-header'>Machine Operating Parameter Simulator</h2>", unsafe_allow_html=True)
        st.markdown("<p class='sub-header'>Configure real-time sensor parameters below. Your settings are saved — navigate to Page 5 to see the failure prediction.</p>", unsafe_allow_html=True)

        # Initialize persistent simulator state if not present
        if "sim_type" not in st.session_state:
            st.session_state["sim_type"] = "M"
            st.session_state["sim_air_temp"] = 300.0
            st.session_state["sim_proc_temp"] = 310.0
            st.session_state["sim_speed"] = 1500.0
            st.session_state["sim_torque"] = 40.0
            st.session_state["sim_wear"] = 50.0

        preset_options = [
            "— Select Preset (Optional) —",
            "Scenario A: Stable Nominal Milling",
            "Scenario B: Heat Dissipation Overload (HDF)",
            "Scenario C: Critical Tool Wear & Overstrain (OSF)",
            "Scenario D: High-Power Spindle Jam (PWF)"
        ]

        preset_configs = {
            "Scenario A: Stable Nominal Milling": {
                "Type": "M", "Air temperature [K]": 298.5, "Process temperature [K]": 308.7,
                "Rotational speed [rpm]": 1550.0, "Torque [Nm]": 38.0, "Tool wear [min]": 25.0
            },
            "Scenario B: Heat Dissipation Overload (HDF)": {
                "Type": "L", "Air temperature [K]": 303.8, "Process temperature [K]": 313.2,
                "Rotational speed [rpm]": 1280.0, "Torque [Nm]": 62.0, "Tool wear [min]": 185.0
            },
            "Scenario C: Critical Tool Wear & Overstrain (OSF)": {
                "Type": "L", "Air temperature [K]": 299.5, "Process temperature [K]": 309.2,
                "Rotational speed [rpm]": 1360.0, "Torque [Nm]": 65.0, "Tool wear [min]": 218.0
            },
            "Scenario D: High-Power Spindle Jam (PWF)": {
                "Type": "H", "Air temperature [K]": 300.0, "Process temperature [K]": 310.5,
                "Rotational speed [rpm]": 2750.0, "Torque [Nm]": 68.0, "Tool wear [min]": 110.0
            }
        }

        col_presets, col_reset = st.columns([4, 1])
        with col_presets:
            selected_preset = st.selectbox(
                "Load Industrial Scenario Preset:",
                preset_options,
                index=0
            )
            # If user picks an active preset from dropdown, apply its values
            if selected_preset in preset_configs:
                cfg = preset_configs[selected_preset]
                st.session_state["sim_type"] = cfg["Type"]
                st.session_state["sim_air_temp"] = float(cfg["Air temperature [K]"])
                st.session_state["sim_proc_temp"] = float(cfg["Process temperature [K]"])
                st.session_state["sim_speed"] = float(cfg["Rotational speed [rpm]"])
                st.session_state["sim_torque"] = float(cfg["Torque [Nm]"])
                st.session_state["sim_wear"] = float(cfg["Tool wear [min]"])

        with col_reset:
            if st.button("Reset Defaults"):
                st.session_state["sim_type"] = "M"
                st.session_state["sim_air_temp"] = 300.0
                st.session_state["sim_proc_temp"] = 310.0
                st.session_state["sim_speed"] = 1500.0
                st.session_state["sim_torque"] = 40.0
                st.session_state["sim_wear"] = 50.0
                st.experimental_rerun()

        st.markdown("### ⚙️ Operating Parameter Controls")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Thermal Parameters**")
            type_options = ["L", "M", "H"]
            cur_type = st.session_state.get("sim_type", "M")
            type_idx = type_options.index(cur_type) if cur_type in type_options else 1
            val_type = st.selectbox(
                "Product Quality Type",
                type_options,
                index=type_idx
            )
            st.session_state["sim_type"] = val_type

            val_air_temp = st.slider(
                "Air Temperature [K]",
                290.0, 310.0,
                value=float(st.session_state.get("sim_air_temp", 300.0)),
                step=0.1
            )
            st.session_state["sim_air_temp"] = val_air_temp

            val_proc_temp = st.slider(
                "Process Temperature [K]",
                300.0, 320.0,
                value=float(st.session_state.get("sim_proc_temp", 310.0)),
                step=0.1
            )
            st.session_state["sim_proc_temp"] = val_proc_temp

        with c2:
            st.markdown("**Mechanical Parameters**")
            val_speed = st.slider(
                "Rotational Speed [rpm]",
                1100.0, 2900.0,
                value=float(st.session_state.get("sim_speed", 1500.0)),
                step=10.0
            )
            st.session_state["sim_speed"] = val_speed

            val_torque = st.slider(
                "Torque [Nm]",
                3.0, 80.0,
                value=float(st.session_state.get("sim_torque", 40.0)),
                step=0.5
            )
            st.session_state["sim_torque"] = val_torque

            val_wear = st.slider(
                "Tool Wear [min]",
                0.0, 260.0,
                value=float(st.session_state.get("sim_wear", 50.0)),
                step=1.0
            )
            st.session_state["sim_wear"] = val_wear

        with c3:
            st.markdown("**📐 Derived Physical Indicators**")
            temp_diff = round(val_proc_temp - val_air_temp, 2)
            omega = val_speed * 2.0 * np.pi / 60.0
            power = round((val_torque * omega) / 1000.0, 2)
            strain = round(val_wear * val_torque, 1)

            st.metric("Temperature Difference (ΔT)", f"{temp_diff} K",
                      delta="High" if temp_diff > 12 else "Normal",
                      delta_color="inverse" if temp_diff > 12 else "normal")
            st.metric("Mechanical Power (P)", f"{power} kW",
                      delta="High" if power > 5.0 else "Normal",
                      delta_color="inverse" if power > 5.0 else "normal")
            st.metric("Wear-Torque Strain Index", f"{strain} min·Nm",
                      delta="Critical" if strain > 10000 else "Normal",
                      delta_color="inverse" if strain > 10000 else "normal")

        # Save to session state for explanation & inference pages
        raw_instance = {
            "Type": val_type,
            "Air temperature [K]": float(val_air_temp),
            "Process temperature [K]": float(val_proc_temp),
            "Rotational speed [rpm]": float(val_speed),
            "Torque [Nm]": float(val_torque),
            "Tool wear [min]": float(val_wear)
        }
        instance_df = pd.DataFrame([raw_instance])
        instance_scaled = preprocessor.transform(instance_df)
        prob = float(best_model.predict_proba(instance_scaled)[0, 1])

        st.session_state["current_raw_instance"] = raw_instance
        st.session_state["current_scaled_instance"] = instance_scaled[0]
        st.session_state["current_prob"] = prob

        st.markdown("---")
        st.info("✅ Parameters saved in real-time. Navigate to **Page 5 → Failure Probability & Decision** or **Page 6/7/8** for SHAP & LIME explanations.")

        st.markdown("### 📋 Current Parameter Summary")
        summary_df = pd.DataFrame([{
            "Parameter": k, "Value": f"{v}" if isinstance(v, str) else f"{v:.2f}"
        } for k, v in raw_instance.items()])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # PAGE 5: FAILURE PROBABILITY & DECISION (output only)
    # -------------------------------------------------------------------------
    elif selected_page == "5. Failure Probability & Decision":
        st.markdown("<h2 class='main-header'>Failure Probability & Operational Decision</h2>", unsafe_allow_html=True)
        st.markdown("<p class='sub-header'>Model inference result for the machine state configured on Page 4.</p>", unsafe_allow_html=True)

        raw_instance = st.session_state.get("current_raw_instance")
        scaled_inst  = st.session_state.get("current_scaled_instance")
        prob         = st.session_state.get("current_prob")

        if raw_instance is None or prob is None:
            st.warning("⚠️ No machine state configured yet. Go to **Page 4 → Machine Operational Simulator** first and set the parameters.")
            return

        # Risk gauge
        render_risk_gauge(prob, threshold=0.50)

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔬 Active Machine State")
            state_df = pd.DataFrame([{
                "Parameter": k,
                "Value": f"{v}" if isinstance(v, str) else f"{v:.2f}"
            } for k, v in raw_instance.items()])
            st.dataframe(state_df, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("### 📊 Model Confidence Breakdown")
            normal_prob = 1.0 - prob
            fig_p, ax_p = plt.subplots(figsize=(4, 3), dpi=120)
            bars = ax_p.barh(
                ["Normal Operation", "Machine Failure"],
                [normal_prob * 100, prob * 100],
                color=["#0E9F6E", "#E02424"],
                alpha=0.85,
                height=0.5
            )
            ax_p.set_xlim(0, 100)
            ax_p.set_xlabel("Probability (%)", fontsize=9)
            ax_p.set_title("Predicted Class Probabilities", fontsize=10, fontweight="bold")
            for bar, val in zip(bars, [normal_prob * 100, prob * 100]):
                ax_p.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                          f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_p)
            plt.close(fig_p)

        st.markdown("---")
        st.markdown("### 🛠️ Recommended Actions")
        if prob >= 0.50:
            st.error("""
**IMMEDIATE ACTION REQUIRED**
- Halt production and initiate emergency inspection
- Check spindle bearings, lubrication levels, and cutter condition
- Review tool wear against replacement schedule
- Log incident in maintenance management system (MMS)
            """)
        elif prob >= 0.25:
            st.warning("""
**PRECAUTIONARY MAINTENANCE ADVISED**
- Schedule inspection during next batch transition window
- Monitor temperature differential and torque trends closely
- Consider proactive tool replacement if wear > 180 min
            """)
        else:
            st.success("""
**NORMAL OPERATION — NO ACTION REQUIRED**
- All parameters within safe operating bounds
- Continue standard production cycle
- Next scheduled maintenance per normal interval
            """)

        st.markdown("---")
        st.caption("⚠️ *Attribution ≠ Causation. Model predictions are diagnostic aids and must be corroborated with physical sensor inspection by qualified engineers.*")

    # -------------------------------------------------------------------------
    # PAGE 6: SHAP EXPLANATION & GLOBAL IMPORTANCE
    # -------------------------------------------------------------------------
    elif selected_page == "6. SHAP Local & Global Explanations":
        st.markdown("<h2 class='main-header'>SHAP (Shapley Additive exPlanations)</h2>", unsafe_allow_html=True)
        st.markdown("<p class='sub-header'>Game-theoretic marginal feature attributions for root-cause identification.</p>", unsafe_allow_html=True)

        scaled_inst = st.session_state.get("current_scaled_instance")
        if scaled_inst is None:
            st.info("Navigate to tab 4 to set up a machine state first.")
            return

        shap_exp = RootIQShapExplainer(best_model, np.load(DATA_PROCESSED_DIR / "X_train_scaled.npy"), feature_names)
        local_shap = shap_exp.explain_instance(scaled_inst)

        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("#### Local SHAP Attribution (Current Machine State)")
            fig = plot_horizontal_attributions(local_shap["sorted_attributions"], "SHAP Marginal Feature Contribution", color="#1f77b4")
            st.pyplot(fig)

        with col2:
            st.markdown("#### Ranked Local Impact Table")
            df_shap = pd.DataFrame(local_shap["sorted_attributions"], columns=["Feature", "SHAP Attribution Value"])
            df_shap["Direction"] = df_shap["SHAP Attribution Value"].apply(lambda v: "🔴 Increases Failure Risk" if v > 0 else "🟢 Mitigates Risk (Normalizing)")
            st.dataframe(df_shap, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Fleet-Level Global Feature Importance (Mean |SHAP|)")
        if (REPORTS_DIR / "metrics_summary.json").exists():
            with open(REPORTS_DIR / "metrics_summary.json", "r") as f:
                metrics_data = json.load(f)
            global_shap = metrics_data.get("global_shap_importance", {})
            # Use matplotlib instead of st.bar_chart to avoid Arrow/complex256 dtype crash
            # in streamlit 1.18.0 with numpy float arrays
            shap_series = pd.Series(global_shap).head(8).sort_values(ascending=True)
            fig_g, ax_g = plt.subplots(figsize=(8, max(3, len(shap_series) * 0.5)), dpi=120)
            ax_g.barh(range(len(shap_series)), shap_series.values.astype(float),
                      color="#1f77b4", alpha=0.85)
            ax_g.set_yticks(range(len(shap_series)))
            ax_g.set_yticklabels(shap_series.index.tolist(), fontsize=9, fontweight="bold")
            ax_g.set_xlabel("Mean |SHAP| Value", fontsize=9, fontweight="bold")
            ax_g.set_title("Global Feature Importance (Top 8 Features)", fontsize=10, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_g)
            plt.close(fig_g)

    # -------------------------------------------------------------------------
    # PAGE 7: LIME LOCAL EXPLANATION
    # -------------------------------------------------------------------------
    elif selected_page == "7. LIME Local Explanation":
        st.markdown("<h2 class='main-header'>LIME (Local Interpretable Model-agnostic Explanations)</h2>", unsafe_allow_html=True)
        st.markdown("<p class='sub-header'>Local surrogate linear regression within the perturbation neighborhood.</p>", unsafe_allow_html=True)

        scaled_inst = st.session_state.get("current_scaled_instance")
        if scaled_inst is None:
            st.info("Navigate to tab 4 to set up a machine state first.")
            return

        cat_indices = [i for i, name in enumerate(feature_names) if name.startswith("Type_")]
        lime_exp = RootIQLimeExplainer(best_model, np.load(DATA_PROCESSED_DIR / "X_train_scaled.npy"), feature_names, cat_indices)
        
        lime_seed = st.slider("LIME Random Seed (Test Stochasticity):", 1, 1000, 42)
        local_lime = lime_exp.explain_instance(scaled_inst, random_seed=lime_seed)

        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(f"#### LIME Local Feature Attributions (Seed = {lime_seed})")
            fig = plot_horizontal_attributions(local_lime["sorted_attributions"], "LIME Local Surrogate Weights", color="#ff7f0e")
            st.pyplot(fig)

        with col2:
            st.markdown("#### LIME Decision Rules & Local Fit")
            st.metric("Local Surrogate R² Fit", f"{local_lime.get('r2_score', 0.0):.3f}" if local_lime.get('r2_score') is not None else "N/A")
            rules = [{"Feature": k, "Decision Rule Range": local_lime["feature_conditions"].get(k, "—"), "LIME Weight": round(v, 4)} for k, v in local_lime["sorted_attributions"][:8]]
            st.dataframe(pd.DataFrame(rules), use_container_width=True)

    # -------------------------------------------------------------------------
    # PAGE 8: SIDE-BY-SIDE DUAL-XAI COMPARISON
    # -------------------------------------------------------------------------
    elif selected_page == "8. Side-by-Side Dual-XAI Comparison":
        st.markdown("<h2 class='main-header'>Side-by-Side Root Cause Comparison: SHAP vs. LIME</h2>", unsafe_allow_html=True)
        st.markdown("<p class='sub-header'>Direct alignment of game-theoretic vs. surrogate explanations on the identical operating point.</p>", unsafe_allow_html=True)

        scaled_inst = st.session_state.get("current_scaled_instance")
        if scaled_inst is None:
            st.info("Navigate to tab 4 to set up a machine state first.")
            return

        cat_indices = [i for i, name in enumerate(feature_names) if name.startswith("Type_")]
        shap_exp = RootIQShapExplainer(best_model, np.load(DATA_PROCESSED_DIR / "X_train_scaled.npy"), feature_names)
        lime_exp = RootIQLimeExplainer(best_model, np.load(DATA_PROCESSED_DIR / "X_train_scaled.npy"), feature_names, cat_indices)

        s_exp = shap_exp.explain_instance(scaled_inst)
        l_exp = lime_exp.explain_instance(scaled_inst, random_seed=42)
        comp = compare_shap_and_lime_explanations(s_exp, l_exp, top_k=5)

        c1, c2, c3 = st.columns(3)
        c1.metric("Spearman Rank Correlation ($\\rho$)", f"{comp['spearman_rank_correlation']:.3f}")
        c2.metric("Top-5 Feature Jaccard Overlap", f"{comp['top_5_jaccard_similarity']:.1%}")
        c3.metric("Cosine Vector Similarity", f"{comp['cosine_similarity']:.3f}")

        st.markdown("#### Comparative Normalized Attribution Chart")
        fig = plt.figure(figsize=(9, 5), dpi=130)
        items = comp["comparison_table"][:8]
        features = [item["feature"] for item in items][::-1]
        shap_vals = [item["shap_norm"] for item in items][::-1]
        lime_vals = [item["lime_norm"] for item in items][::-1]
        y = np.arange(len(features))
        h = 0.35
        plt.barh(y - h/2, shap_vals, h, label="SHAP (Normalized)", color="#1f77b4")
        plt.barh(y + h/2, lime_vals, h, label="LIME (Normalized)", color="#ff7f0e")
        plt.axvline(0, color="gray", lw=0.8, linestyle="--")
        plt.yticks(y, features, fontweight="bold")
        plt.xlabel("Normalized Importance Weight")
        plt.legend(loc="lower right")
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("#### Detailed Attribution Alignment Table")
        st.dataframe(pd.DataFrame(comp["comparison_table"]), use_container_width=True)

    # -------------------------------------------------------------------------
    # PAGE 9: EXPLANATION-STABILITY LAB
    # -------------------------------------------------------------------------
    elif selected_page == "9. Explanation-Stability Lab":
        st.markdown("<h2 class='main-header'>Empirical Explanation Stability & Perturbation Lab</h2>", unsafe_allow_html=True)
        st.markdown("<p class='sub-header'>Controlled measurement of explanation robustness under input sensor noise.</p>", unsafe_allow_html=True)

        if (REPORTS_DIR / "metrics_summary.json").exists():
            with open(REPORTS_DIR / "metrics_summary.json", "r") as f:
                metrics_data = json.load(f)
            stab_summary = metrics_data.get("perturbation_stability_summary", {})
            stochastic_summary = metrics_data.get("lime_seed_stochasticity", {})

            st.markdown("### Stability Summary Under Controlled Perturbations")
            stab_rows = []
            for eps, methods in stab_summary.items():
                stab_rows.append({
                    "Perturbation Level": f"{int(float(eps)*100)}%",
                    "SHAP Mean Spearman $\\rho$": f"{methods['SHAP']['mean_spearman']:.3f} (±{methods['SHAP']['std_spearman']:.3f})",
                    "LIME Mean Spearman $\\rho$": f"{methods['LIME']['mean_spearman']:.3f} (±{methods['LIME']['std_spearman']:.3f})",
                    "SHAP Top-3 Jaccard": f"{methods['SHAP']['mean_top3_jaccard']:.3f}",
                    "LIME Top-3 Jaccard": f"{methods['LIME']['mean_top3_jaccard']:.3f}",
                    "SHAP Cosine Sim": f"{methods['SHAP']['mean_cosine_sim']:.3f}",
                    "LIME Cosine Sim": f"{methods['LIME']['mean_cosine_sim']:.3f}"
                })
            st.dataframe(pd.DataFrame(stab_rows), use_container_width=True)

            if (FIGURES_DIR / "explanation_stability_curves.png").exists():
                st.image(str(FIGURES_DIR / "explanation_stability_curves.png"), caption="Explanation Stability Degradation Curves (SHAP vs. LIME)")

            st.markdown("### LIME Seed Stochastic Sensitivity")
            st.markdown(f"""
            - **Mean Cross-Seed Spearman Rank Correlation:** `{stochastic_summary.get('overall_mean_correlation', 0.0):.4f}`
            - **Cross-Seed Standard Deviation:** `{stochastic_summary.get('overall_std_correlation', 0.0):.4f}`
            - **Takeaway:** LIME exhibits slight ranking variance across random seeds for the *identical* telemetry record, while SHAP is completely deterministic for Tree models.
            """)

    # -------------------------------------------------------------------------
    # PAGE 10: RESEARCH FINDINGS & DOWNLOADS
    # -------------------------------------------------------------------------
    elif selected_page == "10. Research Findings & Reports":
        st.markdown("<h2 class='main-header'>Research Findings & Comprehensive Reports</h2>", unsafe_allow_html=True)
        st.markdown("<p class='sub-header'>Downloadable technical reports, viva preparation guide, and key takeaways.</p>", unsafe_allow_html=True)

        report_files = {
            "Data Quality & Schema Audit Report": REPORTS_DIR / "data_quality_report.md",
            "Predictive Model Comparison Report": REPORTS_DIR / "model_comparison_report.md",
            "SHAP vs. LIME Comparison Study": REPORTS_DIR / "xai_comparison_report.md",
            "Explanation Stability & Perturbation Report": REPORTS_DIR / "stability_study_report.md",
            "Viva Preparation Guide": REPORTS_DIR / "viva_preparation.md",
            "Research Findings Summary": REPORTS_DIR / "research_findings.md"
        }

        for title, filepath in report_files.items():
            if filepath.exists():
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                st.download_button(
                    label=f"📥 Download {title} (.md)",
                    data=content,
                    file_name=filepath.name,
                    mime="text/markdown"
                )
                with st.expander(f"View {title}"):
                    st.markdown(content)


# Entry point: call main() when run directly or under Streamlit server.
# Guard prevents execution during pytest import (test_app_smoke.py uses import app.app).
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    _in_streamlit = get_script_run_ctx() is not None
except ImportError:
    _in_streamlit = False

if _in_streamlit or __name__ == "__main__":
    main()
