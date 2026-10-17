# RootIQ: Empirical Research Findings & Scientific Summary

**Worklet #22:** Tier 3 Advanced — RootIQ: Explainable AI for Root Cause Analysis in Smart Manufacturing  
**Research Team:** Team Diamonds  
**Academic Module:** Applied AI for Hyper Personalized Production Systems  

---

## 1. Executive Research Summary
This research investigation addresses the industrial trust barrier in AI-driven predictive maintenance. Using the verified **UCI AI4I 2020 Predictive Maintenance Dataset** (10,000 continuous CNC milling observations), we built a leak-free predictive model, integrated dual post-hoc explainability engines (**SHAP** and **LIME**), and conducted controlled input perturbation and seed stochasticity experiments to quantify explanation stability under operational industrial sensor noise.

---

## 2. Experimental Outcomes: Model Benchmarking
Models were trained with stratified 70/15/15 splitting and evaluated on the held-out test set ($N=1,500$ records, including 51 true machine failures).

| Model Architecture | Precision | Recall | F1-Score | PR-AUC | ROC-AUC | Balanced Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** *(Selected)* | **1.0000** | **0.8431** | **0.9149** | **0.9160** | **0.9728** | **0.9215** |
| **Random Forest (Balanced)** | 0.9524 | 0.7843 | 0.8602 | 0.9031 | 0.9769 | 0.8911 |
| **HistGradientBoosting** | 0.9111 | 0.8039 | 0.8542 | 0.8878 | 0.9770 | 0.9004 |
| **Logistic Regression (Balanced)** | 0.1803 | 0.8627 | 0.2983 | 0.4431 | 0.9372 | 0.8967 |

### Key Modeling Insights:
1. **Tree Ensembles Outperform Linear Baselines:** Gradient Boosting achieved the highest F1-score (0.9149) with perfect precision (zero false alarms on the test set) and strong recall ($84.3\%$), while Logistic Regression suffered from high false alarm rates ($81.9\%$ false discovery rate) due to inability to model nonlinear thermodynamic thresholds.
2. **First-Principles Feature Engineering Impact:** Engineering domain features—`Temp difference [K]`, `Power [kW]`, and `Wear torque product [min*Nm]`—proved decisive. They mapped directly onto the underlying physical failure modes without relying on post-hoc labels.

---

## 3. Explainability Findings: SHAP vs. LIME
Both explainers were evaluated on identical machine states and predictions from the selected Gradient Boosting model.

### Fleet-Level Feature Importance (Global SHAP):
1. **`Torque [Nm]` & `Power [kW]`**: Primary drivers of power-overload and mechanical resistance failures.
2. **`Wear torque product [min*Nm]`**: Crucial combined stress metric separating normal tool wear from sudden cutter fracture.
3. **`Temp difference [K]`**: Clear discriminator for Heat Dissipation Failure (HDF), triggering when thermal exchange drops below critical cooling thresholds.
4. **`Rotational speed [rpm]`**: Interacts inversely with torque; extreme low rpm under high torque causes immediate stalls.

### Local Alignment (Instance-Level Dual-XAI):
- **Sign Agreement:** On dominant high-magnitude features, SHAP and LIME exhibited high sign agreement ($>85\%$), consistently flagging the same primary risk drivers (e.g. elevated torque and tool wear).
- **Ranking Divergence:** On mid- and low-importance features, LIME and SHAP exhibited rank divergence. LIME's local linear surrogate sometimes assigned non-zero weight to neutral features due to local sampling noise, whereas SHAP's dummy axiom guaranteed zero marginal contribution for inactive features.

---

## 4. Explanation Stability & Perturbation Analysis
Controlled relative input perturbations $\epsilon \in [1\%, 5\%, 10\%, 20\%]$ were injected into continuous sensor readings across 40 held-out test instances, strictly clipped to physical operating boundaries:

### Empirical Stability Degradation:
- **SHAP Stability:** Maintained high Spearman rank correlation ($\rho > 0.88$) and top-3 feature overlap ($>80\%$) at $\epsilon = 1\%$ and $5\%$, demonstrating graceful decay up to $\epsilon = 20\%$.
- **LIME Stability:** Exhibited steeper degradation ($\rho$ dropping toward $0.65$ at $\epsilon = 10\%$). Because LIME fits a linear hyperplane to a randomly sampled Gaussian ball, slight shifts in the sampling center alter surrogate regression slopes.
- **LIME Seed Stochasticity:** Evaluating LIME with 5 distinct random seeds on static machine states revealed cross-seed Spearman correlations of $\approx 0.85 \pm 0.08$. This confirms that LIME explanations possess intrinsic stochastic jitter unless random seeds are fixed.

---

## 5. Documented Failed Approaches & Justified Modifications

1. **Attempted Use of Raw Failure Modes in Feature Set:**  
   *Initial Exploration:* Using `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` produced an artificial $100\%$ test accuracy.  
   *Justification & Fix:* Diagnosed as catastrophic target leakage. In production, these flags are diagnostic post-failure outcomes, not pre-failure telemetry. They were strictly excluded from feature inputs.
2. **Scikit-Learn Version Compatibility Adjustments:**  
   *Issue:* Scikit-learn 1.0.2 in the runtime environment does not support `sparse_output=False` in `OneHotEncoder` (requires `sparse=False`) nor `class_weight` directly in `HistGradientBoostingClassifier`.  
   *Justification & Fix:* Implemented dynamic version-resilient try/except wrappers and verified backward compatibility.
3. **LIME Explainer Random State Typing:**  
   *Issue:* Passing a raw integer to `explainer.random_state` caused an `AttributeError` during data inversion sampling.  
   *Justification & Fix:* Initialized `self.explainer.random_state = np.random.RandomState(random_seed)`, guaranteeing deterministic sampling reproducibility.

---

## 6. Recommendations for Industrial Deployment
1. **Prefer SHAP for Safety-Critical Shutdowns:** Due to deterministic guarantees, cooperative game axioms, and robustness against sensor noise.
2. **Use LIME for Fast "What-If" Counterfactual Exploration:** LIME's decision rules (e.g. `Torque > 52.4 Nm`) are intuitive for shop-floor operators.
3. **Always Calibrate Sensor Noise Envelopes:** Enforce threshold checks ($\epsilon \le 5\%$) before trusting automated root-cause recommendations.
