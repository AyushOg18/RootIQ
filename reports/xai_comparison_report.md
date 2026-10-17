# Explainable AI: SHAP vs. LIME Comparative Study

## 1. Purpose & Theoretical Underpinnings
Industrial engineers require transparent explanations before scheduling expensive machine shutdowns.
This investigation compares two foundational post-hoc explainability paradigms on identical machine states:
1. **SHAP (Shapley Additive exPlanations):** Grounded in cooperative game theory (Lloyd Shapley, 1953). Computes exact marginal contributions of each operating parameter across all possible feature subsets. Guaranteed to satisfy efficiency, symmetry, dummy, and additivity axioms.
2. **LIME (Local Interpretable Model-agnostic Explanations):** Grounded in sparse local surrogate approximations. Generates randomized perturbations around the operating point and fits an interpretable linear model weighted by distance.

## 2. Global Feature Importance (SHAP Aggregate)
The top influential operating parameters driving predictive maintenance risk across the industrial fleet:

| Rank | Feature | Mean Absolute SHAP Value | Physical Interpretation |
| :--- | :--- | :--- | :--- |
| 1 | `Power [kW]` | 12655.5947 | Dominant fleet-level risk factor |
| 2 | `Process temperature [K]` | 12077.2697 | Dominant fleet-level risk factor |
| 3 | `Tool wear [min]` | 0.6266 | Dominant fleet-level risk factor |
| 4 | `Rotational speed [rpm]` | 0.5192 | Dominant fleet-level risk factor |
| 5 | `Temp difference [K]` | 0.3658 | Dominant fleet-level risk factor |
| 6 | `Wear torque product [min*Nm]` | 0.2349 | Dominant fleet-level risk factor |
| 7 | `Torque [Nm]` | 0.1803 | Dominant fleet-level risk factor |
| 8 | `Air temperature [K]` | 0.1434 | Dominant fleet-level risk factor |

## 3. Local Explanation Agreement for High-Risk Machine (Predicted Failure Risk: 99.2%)
- **Spearman Rank Correlation ($\rho$):** `0.2091`
- **Top-5 Feature Jaccard Overlap:** `42.86%`
- **Normalized Vector Cosine Similarity:** `0.2387`
- **Common Top-5 Risk Drivers:** `Temp difference [K]`, `Rotational speed [rpm]`, `Power [kW]`

## 4. Crucial Distinction: Attribution vs. Physical Causality
> [!WARNING]
> Machine learning feature attributions reflect mathematical correlations and decision boundaries learned by the model. While high `Wear torque product` correlates strongly with machine failure, the model explanation does **not** prove mechanical causation without sensor calibration and metallurgical verification.
