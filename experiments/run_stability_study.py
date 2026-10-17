"""Standalone, configurable experiment script for explanation stability research.

Evaluates:
- Perturbation response across epsilon values: 1%, 5%, 10%, 20%
- Metric outputs: Spearman rho, Top-k Jaccard overlap, Cosine vector similarity, L1 shift
- LIME stochasticity across distinct seeds
- Generates dedicated research figures and markdown tables
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    RANDOM_SEED,
    PERTURBATION_EPSILONS
)
from src.preprocessor import RootIQPreprocessor
from src.models import ModelTrainer
from src.explainers import RootIQShapExplainer, RootIQLimeExplainer
from src.stability import ExplanationStabilityStudy
from src.utils import plot_explanation_stability_curves

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_standalone_stability_study(sample_size: int = 30):
    logger.info("Initializing Standalone Explanation Stability Research Study...")
    
    # Load model and preprocessor
    model_artifact = ModelTrainer.load_model_artifact(MODELS_DIR / "best_model.joblib")
    model = model_artifact["model"]
    preprocessor = RootIQPreprocessor.load(MODELS_DIR / "preprocessor.joblib")
    feature_names = preprocessor.feature_names_

    # Load test split data
    X_test_df = pd.read_csv(DATA_PROCESSED_DIR / "X_test_raw.csv")
    y_test = pd.read_csv(DATA_PROCESSED_DIR / "y_test.csv").iloc[:, 0]
    X_train_scaled = np.load(DATA_PROCESSED_DIR / "X_train_scaled.npy")

    # Initialize explainers
    shap_explainer = RootIQShapExplainer(model, X_train_scaled, feature_names)
    cat_indices = [i for i, name in enumerate(feature_names) if name.startswith("Type_")]
    lime_explainer = RootIQLimeExplainer(model, X_train_scaled, feature_names, categorical_features=cat_indices)

    stability_engine = ExplanationStabilityStudy(
        shap_explainer=shap_explainer,
        lime_explainer=lime_explainer,
        preprocessor=preprocessor,
        feature_names=feature_names
    )

    # Sample balanced set
    pos_idx = np.where(y_test.values == 1)[0]
    neg_idx = np.where(y_test.values == 0)[0]
    
    n_pos = min(sample_size // 2, len(pos_idx))
    n_neg = min(sample_size - n_pos, len(neg_idx))

    samples_pos = X_test_df.iloc[pos_idx[:n_pos]]
    samples_neg = X_test_df.iloc[neg_idx[:n_neg]]
    sample_df = pd.concat([samples_pos, samples_neg]).reset_index(drop=True)

    # Run perturbation analysis
    logger.info(f"Perturbation Study on {len(sample_df)} instances across epsilons: {PERTURBATION_EPSILONS}")
    pert_results = stability_engine.run_perturbation_study(
        sample_df,
        epsilons=PERTURBATION_EPSILONS,
        random_seed=RANDOM_SEED
    )

    # Run LIME seed sensitivity analysis
    logger.info("LIME Seed Sensitivity Experiment across 5 random seeds...")
    seed_results = stability_engine.run_lime_seed_sensitivity_study(
        sample_df.iloc[:min(10, len(sample_df))],
        seeds=[10, 42, 99, 123, 777]
    )

    plot_explanation_stability_curves(pert_results["summary"])
    logger.info("Stability study completed. Summary:")
    print(json.dumps(pert_results["summary"], indent=2))
    return pert_results, seed_results


if __name__ == "__main__":
    run_standalone_stability_study()
