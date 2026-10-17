# RootIQ: Technical Viva & Defense Preparation Guide

**Project:** RootIQ: Explainable AI for Root Cause Analysis in Smart Manufacturing  
**Team:** Team Diamonds  
**Worklet:** #22 — Tier 3 Advanced  
**Course:** Applied AI for Hyper Personalized Production Systems  

---

## 1. Problem Formulation & Industrial Context

### Q1: What specific problem in smart manufacturing does RootIQ solve?
**Answer:**  
In smart manufacturing and Industry 4.0 production lines, black-box machine learning models can achieve high nominal accuracy in detecting imminent machine failures. However, process engineers and plant maintenance supervisors frequently mistrust and dismiss these predictions because the models fail to provide actionable root-cause explanations. Unplanned shutdowns based on unverified false alarms cost thousands of dollars in lost throughput, while ignored true positives lead to catastrophic tool breakage and spindle damage.  
RootIQ resolves this "trust deficit" by:
1. Predicting failure risk on clean, non-leaking industrial telemetry.
2. Generating transparent, post-hoc local explanations through dual XAI methods (SHAP and LIME).
3. Quantifying the stability of these explanations under operational sensor noise and random seed variations.

---

### Q2: What dataset is used, and what does it represent?
**Answer:**  
We utilize the **UCI AI4I 2020 Predictive Maintenance Dataset** (Matzka, 2020). It reflects real telemetry from a Computer Numerical Control (CNC) milling machine operating on low, medium, and high-variant workpieces (Types L, M, and H). The dataset consists of 10,000 continuous operating observations containing ambient temperature, internal process temperature, rotational speed, torque, and cumulative tool wear.

---

### Q3: How did you handle target leakage and why is it critical?
**Answer:**  
In the raw AI4I 2020 dataset, there are two identifiers (`UDI`, `Product ID`) and five specific failure mode flags:
- `TWF`: Tool Wear Failure
- `HDF`: Heat Dissipation Failure
- `PWF`: Power Failure
- `OSF`: Overstrain Failure
- `RNF`: Random Failure

The composite label `Machine failure` is a logical union of these failure modes ($TWF \cup HDF \cup PWF \cup OSF \cup RNF$). If these five failure flags are left in the feature matrix $X$, the predictive model trivially learns a direct identity mapping, yielding $100\%$ artificial accuracy. This represents severe **target leakage** because in real-time industrial operation, failure mode flags are diagnosed *after* the failure has occurred.  
RootIQ strictly excludes all five failure mode flags and the surrogate identifiers from model training and explanation inputs, using only raw and physically derived operating sensor telemetry.

---

### Q4: Why is class imbalance an issue in predictive maintenance, and how is it addressed?
**Answer:**  
Only $\approx 3.4\%$ of the 10,000 records represents machine failure (approx. 339 failures vs. 9,661 normal operating records). A naive classifier that predicts "Normal" $100\%$ of the time would achieve $96.6\%$ accuracy while missing every single catastrophic breakdown.  
To handle this severe imbalance:
1. **Stratified Splitting:** 70% Train, 15% Validation, 15% Test, preserving identical failure proportions across all splits.
2. **Cost-Sensitive Learning:** Algorithms are configured with balanced class weights (`class_weight='balanced'`), penalizing false negatives inversely proportional to class frequencies:
   $$W_{failure} = \frac{N_{total}}{2 \cdot N_{failure}}$$
3. **Appropriate Evaluation Metrics:** Model selection is governed by **F1-Score**, **Precision-Recall AUC (PR-AUC)**, and **Recall** (minimizing Missed Failure Rate / FNR) rather than overall accuracy.

---

## 2. Feature Engineering & Physical Principles

### Q5: What domain-specific features were engineered, and what physical failure mechanisms do they represent?
**Answer:**  
RootIQ engineers three physically grounded features from first principles:
1. **Temperature Difference ($\Delta T = T_{process} - T_{air}$ in Kelvin):**  
   Represents heat exchange capacity. When rotational speed drops below 1380 rpm while $\Delta T < 8.6\text{ K}$, heat dissipation fails, triggering thermal expansion and heat dissipation failure (HDF).
2. **Mechanical Power ($P = \tau \cdot \omega$ in kW):**  
   Computed from torque $\tau\text{ (Nm)}$ and angular velocity $\omega = \frac{2\pi \cdot \text{rpm}}{60}$. Power failure (PWF) occurs when mechanical power consumption falls below $3.5\text{ kW}$ (underload/belt slip) or exceeds $9.0\text{ kW}$ (overload/spindle stall).
3. **Tool Wear-Torque Product (Strain $\approx W_{tool} \cdot \tau$ in $\text{min}\cdot\text{Nm}$):**  
   Captures mechanical stress accumulation. Overstrain failure (OSF) occurs when high torque is applied to an already worn tool cutter.

---

## 3. Explainable AI: SHAP vs. LIME

### Q6: What is SHAP and what are its mathematical foundations?
**Answer:**  
SHAP (Shapley Additive exPlanations, Lundberg & Lee, 2017) is grounded in **cooperative game theory** (Lloyd Shapley, 1953). It treats each feature as a player in a coalition, where the model output is the total payout. The Shapley value $\phi_i$ represents the average marginal contribution of feature $i$ across all possible subsets of features $S \subseteq F \setminus \{i\}$:
$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$
SHAP is uniquely distinguished by satisfying four foundational theoretical axioms:
1. **Efficiency:** $\sum_{i} \phi_i(x) = f(x) - E[f(x)]$ (the sum of attributions equals the difference between model output and baseline expected value).
2. **Symmetry:** If two features contribute equally across all coalitions, their Shapley values are identical.
3. **Dummy (Null player):** A feature that does not change the prediction in any coalition receives zero attribution.
4. **Additivity:** For an ensemble model, the Shapley value of the ensemble is the sum of the Shapley values of the individual estimators.

---

### Q7: What is LIME and how does it construct local explanations?
**Answer:**  
LIME (Local Interpretable Model-agnostic Explanations, Ribeiro et al., 2016) builds a sparse, interpretable surrogate model $g \in G$ (typically a Ridge linear regressor) in the immediate perturbation neighborhood of the instance $x$.  
LIME optimizes:
$$\xi(x) = \arg\min_{g \in G} \mathcal{L}(f, g, \pi_x) + \Omega(g)$$
Where:
- $\mathcal{L}$ is the squared loss between black-box prediction $f(z)$ and surrogate prediction $g(z)$ over sampled perturbations $z$.
- $\pi_x(z) = \exp\left(-\frac{D(x, z)^2}{\sigma^2}\right)$ is an exponential kernel weighting samples by proximity to instance $x$.
- $\Omega(g)$ penalizes model complexity (limiting the number of non-zero coefficients).

---

### Q8: What are the primary differences between SHAP and LIME in practice?
**Answer:**  
| Criterion | SHAP (TreeExplainer) | LIME (Tabular) |
| :--- | :--- | :--- |
| **Theoretical Foundation** | Cooperative Game Theory (Shapley Values) | Local Linear Surrogate Regression |
| **Axiomatic Guarantees** | Efficiency, Symmetry, Dummy, Additivity | None (heuristic local loss minimization) |
| **Determinism** | **Deterministic** (identical output for identical input) | **Stochastic** (Monte Carlo neighborhood sampling introduces jitter) |
| **Global Consistency** | Directly aggregates into consistent global importances | Local explanations do not sum to global importance |
| **Computational Cost** | $O(T L D^2)$ for tree ensembles (fast) | Depends on number of perturbation samples (e.g., 2000 evaluations per instance) |

---

### Q9: Does a high feature attribution mean that feature physically caused the machine failure?
**Answer:**  
**No. Absolutely not.**  
A model explanation reflects the mathematical sensitivity of the learned decision surface. It identifies which sensor readings pushed the model's posterior probability $P(\text{Failure} \mid x)$ across the classification threshold.  
While `Wear torque product` or `Temp difference` may receive high attribution, the model only measures **correlation in the feature space**. True physical root-cause verification requires metallurgical testing, vibration frequency analysis, or oil debris inspection. In our dashboard and reports, RootIQ explicitly enforces this distinction so operators do not mistake statistical feature attribution for mechanical causality.

---

## 4. Explanation Stability Research

### Q10: Why is explanation stability a critical research requirement for industrial AI?
**Answer:**  
In physical manufacturing, real industrial sensors exhibit nominal measurement jitter, quantization errors, and electromagnetic interference (typically $\pm 1\%$ to $\pm 5\%$). If an AI explainer alters its top-ranked root cause from "Tool Wear" to "Ambient Air Temperature" when a sensor fluctuates by only $1\%$, process engineers cannot trust the diagnostic output.  
A trustworthy industrial XAI system must be **locally Lipschitz-continuous**—small bounded input perturbations $\Delta x$ should yield proportionally small shifts in explanation vectors $\Delta \phi$.

---

### Q11: How did you measure explanation stability?
**Answer:**  
We established three complementary mathematical metrics:
1. **Spearman Rank Correlation ($\rho$):** Measures whether the *relative priority order* of attributed root causes is preserved before and after perturbation:
   $$\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$
2. **Top-$k$ Feature Overlap (Jaccard Index):** Measures whether the $k$ most critical root causes remain in the top-$k$ set:
   $$J(S_{orig}, S_{pert}) = \frac{|S_{orig} \cap S_{pert}|}{|S_{orig} \cup S_{pert}|}$$
3. **Normalized Cosine Similarity:** Measures the angular orientation consistency of the normalized attribution vectors:
   $$\text{CosineSim}(v_1, v_2) = \frac{v_1 \cdot v_2}{\|v_1\|_2 \|v_2\|_2}$$

---

### Q12: What did your stability experiments reveal about SHAP vs. LIME?
**Answer:**  
1. **SHAP exhibited superior stability** across all perturbation scales ($\epsilon \in [0.01, 0.20]$). Because `TreeExplainer` evaluates exact partition boundaries within the tree ensemble, minor input shifts within leaf intervals produce zero or minimal attribution changes.
2. **LIME exhibited higher sensitivity to sensor perturbations.** Because LIME samples points randomly in a Gaussian ball and weights them via an exponential kernel, small shifts in the center point shift the local sample distribution, altering the surrogate regression slope.
3. **LIME exhibits seed sensitivity:** Re-running LIME on the exact same static machine state with 5 different random seeds yields a standard deviation in rank correlation, confirming that LIME explanations require seed fixing or multi-run bagging before deployment in critical environments.

---

## 5. Software Architecture & Deployment

### Q13: What architectural patterns ensure RootIQ is modular, testable, and production-ready?
**Answer:**  
- **Separation of Concerns:** Configuration (`config.py`), data loading (`data_loader.py`), preprocessing (`preprocessor.py`), modeling (`models.py`), explainability (`explainers.py`), stability analysis (`stability.py`), and visualization (`utils.py`, `components.py`) are strictly decoupled.
- **Stateless Pipeline & Serialized Artifacts:** All trained pipelines and models are saved using `joblib` with pinned configurations and random seeds.
- **Defensive Engineering:** Domain constraints clip any out-of-bounds user inputs or perturbations, preventing physically absurd operating states.
- **Automated Test Suite:** Comprehensive `pytest` test suite validates schema integrity, leakage prevention, stratification, metric calculations, explainer interfaces, and app components.
