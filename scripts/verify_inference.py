"""Verification script: passing brand new inputs through the complete inference & XAI pipeline."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import MODELS_DIR
from src.preprocessor import RootIQPreprocessor
from src.models import ModelTrainer
from src.explainers import RootIQShapExplainer, RootIQLimeExplainer, compare_shap_and_lime_explanations

def main():
    print("Testing brand-new operational machine input...")
    preprocessor = RootIQPreprocessor.load(MODELS_DIR / "preprocessor.joblib")
    model_artifact = ModelTrainer.load_model_artifact(MODELS_DIR / "best_model.joblib")
    model = model_artifact["model"]

    # Simulating a high-stress CNC operational state
    sample_input = {
        "Type": "L",
        "Air temperature [K]": 302.5,
        "Process temperature [K]": 312.1,
        "Rotational speed [rpm]": 1320.0,
        "Torque [Nm]": 64.5,
        "Tool wear [min]": 215.0
    }
    
    input_df = pd.DataFrame([sample_input])
    print(f"Input features:\n{input_df}")

    # Transform through pipeline
    X_scaled = preprocessor.transform(input_df)
    print(f"Scaled feature shape: {X_scaled.shape}")

    # Predict
    prob = float(model.predict_proba(X_scaled)[0, 1])
    pred = int(prob >= 0.50)
    print(f"Prediction: {'FAILURE RISK (Class 1)' if pred == 1 else 'NORMAL (Class 0)'}")
    print(f"Predicted Failure Probability: {prob:.4f} ({prob*100:.1f}%)")

    # Explain with SHAP
    X_train_scaled = np.load("data/processed/X_train_scaled.npy")
    shap_exp = RootIQShapExplainer(model, X_train_scaled, preprocessor.feature_names_)
    shap_res = shap_exp.explain_instance(X_scaled[0])
    print("\nTop 3 SHAP Risk Contributors:")
    for feat, weight in shap_res["sorted_attributions"][:3]:
        print(f"  - {feat}: {weight:+.4f}")

    # Explain with LIME
    cat_indices = [i for i, name in enumerate(preprocessor.feature_names_) if name.startswith("Type_")]
    lime_exp = RootIQLimeExplainer(model, X_train_scaled, preprocessor.feature_names_, categorical_features=cat_indices)
    lime_res = lime_exp.explain_instance(X_scaled[0], num_samples=500, random_seed=42)
    print("\nTop 3 LIME Risk Contributors:")
    for feat, weight in lime_res["sorted_attributions"][:3]:
        print(f"  - {feat}: {weight:+.4f}")

    # Compare
    comp = compare_shap_and_lime_explanations(shap_res, lime_res, top_k=3)
    print(f"\nDual-XAI Spearman Rank Correlation: {comp['spearman_rank_correlation']:.4f}")
    print(f"Dual-XAI Top-3 Jaccard Overlap: {comp['top_3_jaccard_similarity']:.4f}")
    print("\nINFERENCE & DUAL-XAI VERIFICATION: SUCCESS!")

if __name__ == "__main__":
    main()
