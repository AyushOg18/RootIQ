"""Explainable AI (XAI) module integrating SHAP and LIME for RootIQ."""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import shap
import lime
import lime.lime_tabular

from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)


class RootIQShapExplainer:
    """SHAP Explainer wrapper providing TreeExplainer and KernelExplainer capabilities."""

    def __init__(self, model: Any, X_train: np.ndarray, feature_names: List[str]):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.explainer = None
        self._init_explainer()

    def _init_explainer(self):
        """Initialize appropriate SHAP explainer."""
        try:
            # TreeExplainer is fast, exact, and optimal for Tree ensembles
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("Initialized shap.TreeExplainer")
        except Exception as e:
            logger.info(f"TreeExplainer not directly supported ({e}), initializing KernelExplainer background sample.")
            # Sample 50 background rows for KernelExplainer
            background_sample = shap.sample(self.X_train, min(50, len(self.X_train)), random_state=RANDOM_SEED)
            self.explainer = shap.KernelExplainer(self.model.predict_proba, background_sample)

    def explain_instance(self, instance: np.ndarray) -> Dict[str, Any]:
        """Generate local SHAP explanation for a single 1D feature array for Class 1 (Failure)."""
        instance_2d = instance.reshape(1, -1)
        raw_shap_values = self.explainer.shap_values(instance_2d)

        # Handle different output formats of shap_values across shap versions
        if isinstance(raw_shap_values, list):
            # Binary classification list: [class_0_values, class_1_values]
            sv = raw_shap_values[1][0]
            expected_val = (
                self.explainer.expected_value[1]
                if isinstance(self.explainer.expected_value, (list, np.ndarray))
                else self.explainer.expected_value
            )
        elif isinstance(raw_shap_values, np.ndarray) and raw_shap_values.ndim == 3:
            # Shape (1, n_features, n_classes)
            sv = raw_shap_values[0, :, 1]
            expected_val = (
                self.explainer.expected_value[1]
                if isinstance(self.explainer.expected_value, (list, np.ndarray))
                else self.explainer.expected_value
            )
        else:
            # 2D array: (1, n_features)
            sv = raw_shap_values[0]
            expected_val = self.explainer.expected_value

        attribution_dict = {name: float(val) for name, val in zip(self.feature_names, sv)}
        
        # Sort by absolute attribution magnitude
        sorted_attributions = sorted(
            attribution_dict.items(),
            key=lambda item: abs(item[1]),
            reverse=True
        )

        return {
            "method": "SHAP",
            "feature_attributions": attribution_dict,
            "sorted_attributions": sorted_attributions,
            "base_value": float(expected_val) if expected_val is not None else 0.0,
            "feature_names": self.feature_names,
            "raw_vector": sv
        }

    def explain_dataset(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Compute global SHAP values across dataset sample and return global feature importance."""
        raw_shap = self.explainer.shap_values(X)
        if isinstance(raw_shap, list):
            sv = raw_shap[1]
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
            sv = raw_shap[:, :, 1]
        else:
            sv = raw_shap
            
        mean_abs_shap = np.mean(np.abs(sv), axis=0)
        global_importance = {
            name: float(val) for name, val in zip(self.feature_names, mean_abs_shap)
        }
        sorted_global = dict(
            sorted(global_importance.items(), key=lambda x: x[1], reverse=True)
        )
        return sv, sorted_global


class RootIQLimeExplainer:
    """LIME Tabular Explainer wrapper."""

    def __init__(
        self,
        model: Any,
        X_train: np.ndarray,
        feature_names: List[str],
        categorical_features: Optional[List[int]] = None
    ):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.X_train,
            feature_names=self.feature_names,
            class_names=["Normal", "Failure"],
            categorical_features=self.categorical_features,
            mode="classification",
            random_state=RANDOM_SEED
        )

    def explain_instance(
        self,
        instance: np.ndarray,
        num_features: int = None,
        num_samples: int = 2000,
        random_seed: int = RANDOM_SEED
    ) -> Dict[str, Any]:
        """Generate local LIME explanation for a single 1D feature array for Class 1."""
        if num_features is None:
            num_features = len(self.feature_names)

        # Temporarily set seed on explainer random_state to test stochasticity
        self.explainer.random_state = np.random.RandomState(random_seed)
        
        # Predict probability function
        predict_fn = self.model.predict_proba

        exp = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=predict_fn,
            labels=(1,),
            num_features=num_features,
            num_samples=num_samples
        )

        # LIME returns list of (feature_rule_str, weight)
        raw_list = exp.as_list(label=1)
        
        # Map back to canonical feature names
        attribution_dict = {name: 0.0 for name in self.feature_names}
        feature_conditions = {}
        for rule_str, weight in raw_list:
            matched_name = None
            # Find which feature name matches the rule string
            for name in self.feature_names:
                if name in rule_str:
                    matched_name = name
                    break
            if matched_name:
                attribution_dict[matched_name] = float(weight)
                feature_conditions[matched_name] = rule_str
            else:
                attribution_dict[rule_str] = float(weight)

        sorted_attributions = sorted(
            attribution_dict.items(),
            key=lambda item: abs(item[1]),
            reverse=True
        )

        vector = np.array([attribution_dict[name] for name in self.feature_names])

        return {
            "method": "LIME",
            "feature_attributions": attribution_dict,
            "sorted_attributions": sorted_attributions,
            "feature_conditions": feature_conditions,
            "feature_names": self.feature_names,
            "raw_vector": vector,
            "intercept": float(exp.intercept[1]) if hasattr(exp, "intercept") and isinstance(exp.intercept, dict) and 1 in exp.intercept else 0.0,
            "local_pred": float(exp.local_pred[0]) if hasattr(exp, "local_pred") and isinstance(exp.local_pred, (list, np.ndarray)) else (float(exp.local_pred) if hasattr(exp, "local_pred") else None),
            "r2_score": float(exp.score) if hasattr(exp, "score") and exp.score is not None else None
        }


def compare_shap_and_lime_explanations(
    shap_exp: Dict[str, Any],
    lime_exp: Dict[str, Any],
    top_k: int = 5
) -> Dict[str, Any]:
    """Compare local SHAP and LIME explanations for the identical observation.
    
    Computes:
    1. Normalized feature weights: w_norm = w / sum(|w|)
    2. Rank correlation (Spearman rho)
    3. Top-k feature overlap (Jaccard similarity)
    4. Sign agreement on dominant features
    """
    feature_names = shap_exp["feature_names"]
    
    # Extract raw vectors
    v_shap = np.array([shap_exp["feature_attributions"].get(f, 0.0) for f in feature_names])
    v_lime = np.array([lime_exp["feature_attributions"].get(f, 0.0) for f in feature_names])
    
    # Normalized weights
    sum_abs_shap = np.sum(np.abs(v_shap))
    sum_abs_lime = np.sum(np.abs(v_lime))
    
    norm_shap = (v_shap / sum_abs_shap) if sum_abs_shap > 1e-12 else np.zeros_like(v_shap)
    norm_lime = (v_lime / sum_abs_lime) if sum_abs_lime > 1e-12 else np.zeros_like(v_lime)
    
    # Rankings (1 = highest magnitude)
    abs_shap = np.abs(v_shap)
    abs_lime = np.abs(v_lime)
    
    rank_shap = len(abs_shap) - np.argsort(np.argsort(abs_shap))
    rank_lime = len(abs_lime) - np.argsort(np.argsort(abs_lime))
    
    # Spearman rank correlation
    if np.all(v_shap == 0) or np.all(v_lime == 0):
        spearman_corr = 0.0
        p_val = 1.0
    else:
        corr_res = spearmanr(abs_shap, abs_lime)
        spearman_corr = float(corr_res.correlation) if not np.isnan(corr_res.correlation) else 0.0
        p_val = float(corr_res.pvalue) if not np.isnan(corr_res.pvalue) else 1.0

    # Top-K sets
    sorted_shap_features = [item[0] for item in shap_exp["sorted_attributions"][:top_k]]
    sorted_lime_features = [item[0] for item in lime_exp["sorted_attributions"][:top_k]]
    
    set_shap_k = set(sorted_shap_features)
    set_lime_k = set(sorted_lime_features)
    
    intersection_k = set_shap_k.intersection(set_lime_k)
    union_k = set_shap_k.union(set_lime_k)
    jaccard_k = float(len(intersection_k) / len(union_k)) if len(union_k) > 0 else 0.0
    
    # Cosine similarity of normalized vectors
    denom = (np.linalg.norm(norm_shap) * np.linalg.norm(norm_lime))
    cosine_sim = float(np.dot(norm_shap, norm_lime) / denom) if denom > 1e-12 else 0.0
    
    comparison_table = []
    for f in feature_names:
        w_s = shap_exp["feature_attributions"].get(f, 0.0)
        w_l = lime_exp["feature_attributions"].get(f, 0.0)
        nw_s = float(w_s / sum_abs_shap) if sum_abs_shap > 1e-12 else 0.0
        nw_l = float(w_l / sum_abs_lime) if sum_abs_lime > 1e-12 else 0.0
        same_sign = bool((w_s * w_l) > 0 or (abs(w_s) < 1e-6 and abs(w_l) < 1e-6))
        
        comparison_table.append({
            "feature": f,
            "shap_raw": float(w_s),
            "shap_norm": round(nw_s, 4),
            "lime_raw": float(w_l),
            "lime_norm": round(nw_l, 4),
            "sign_agreement": same_sign
        })

    # Sort table by average absolute normalized contribution
    comparison_table = sorted(
        comparison_table,
        key=lambda x: (abs(x["shap_norm"]) + abs(x["lime_norm"])),
        reverse=True
    )

    return {
        "spearman_rank_correlation": round(spearman_corr, 4),
        "spearman_pvalue": round(p_val, 4),
        f"top_{top_k}_overlap_count": len(intersection_k),
        f"top_{top_k}_jaccard_similarity": round(jaccard_k, 4),
        f"top_{top_k}_common_features": list(intersection_k),
        "cosine_similarity": round(cosine_sim, 4),
        "comparison_table": comparison_table
    }
