# Explanation Stability Research Report: Perturbation & Stochastic Sensitivity

**Project:** RootIQ: Explainable AI for Root Cause Analysis in Smart Manufacturing  
**Team:** Team Diamonds | Worklet #22 (Tier 3 Advanced)  
**Evaluated Model:** Gradient Boosting Classifier (Production Best)  

---

## 1. Research Hypothesis & Problem Statement
In Industry 4.0 smart manufacturing, real physical sensors inherently experience minor analog-to-digital conversion noise, thermal drift, and operational fluctuations (typically between $\pm 1\%$ and $\pm 5\%$).
If an explainability system is to be deployed in safety-critical automated manufacturing:
> **Hypothesis:** Bounded input perturbations representing realistic sensor noise should not trigger erratic, discontinuous permutations in the attributed root-cause feature rankings.

This study empirically evaluates whether **SHAP (Shapley Additive exPlanations)** and **LIME (Local Interpretable Model-agnostic Explanations)** satisfy this operational robustness requirement.

---

## 2. Experimental Design & Methodology

### A. Test Population & Sampling
- **Held-Out Test Sample ($N=40$):** Selected from the unseen test split, balanced across actual catastrophic failure states ($y=1$) and stable nominal machine cycles ($y=0$).
- **No Ground-Truth Leakage:** Ground truth labels and post-failure flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) were strictly excluded from explanation feature sets.

### B. Controlled Perturbation Mechanism
For each continuous operational telemetry feature $x_i$, perturbations were generated at four relative scales:
$$\epsilon \in \{0.01, 0.05, 0.10, 0.20\} \quad (1\%, 5\%, 10\%, 20\%)$$
where:
$$x_{i,\text{pert}} = x_i + \delta_i, \quad \delta_i \sim \text{Uniform}(-\epsilon \cdot \text{span}_i, +\epsilon \cdot \text{span}_i)$$
$$\text{span}_i = x_{i,\max} - x_{i,\min}$$

### C. Physical Domain Boundary Constraints
To prevent physically absurd operating states (such as negative rotational speeds or temperatures below absolute zero), all perturbed values were rigorously clipped to the physical domain envelope:
- Air Temperature: $[290.0, 310.0]\text{ K}$
- Process Temperature: $[300.0, 320.0]\text{ K}$
- Rotational Speed: $[1100.0, 2900.0]\text{ rpm}$
- Torque: $[3.0, 80.0]\text{ Nm}$
- Tool Wear: $[0.0, 260.0]\text{ min}$

### D. Mathematical Stability Metrics
1. **Spearman Rank Correlation ($\rho$):** Preserved ranking order of feature importance magnitudes between base explanation $v$ and perturbed explanation $v'$.
2. **Top-3 Feature Overlap (Jaccard Index):** Overlap ratio $J = \frac{|S_3 \cap S_3'|}{|S_3 \cup S_3'|}$ measuring whether the three primary root causes remain identical.
3. **Normalized Cosine Vector Similarity:** Vector angular orientation similarity $\frac{v \cdot v'}{\|v\|_2 \|v'\|_2}$.
4. **Mean Attribution Shift (L1 Distance):** Shift in normalized feature weights $\frac{1}{d} \sum |w_{\text{norm}, i} - w'_{\text{norm}, i}|$.

---

## 3. Actual Empirical Experimental Findings

### Table 1: Explanation Stability Across Perturbation Magnitudes ($\epsilon$)
*Data obtained from execution on $N=40$ held-out test observations:*

| Perturbation Level ($\epsilon$) | Method | Mean Spearman $\rho$ ($\pm$ std) | Mean Top-3 Jaccard | Mean Cosine Sim | Mean L1 Shift |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1%** Nominal Noise | **SHAP** | **0.9848** $\pm$ 0.0393 | **0.9625** | **0.9777** | **0.0002** |
| | LIME | 0.3230 $\pm$ 0.2897 | 0.3925 | 0.5811 | 0.0562 |
| **5%** Moderate Noise | **SHAP** | **0.9598** $\pm$ 0.0514 | **0.9125** | **0.9107** | **0.0010** |
| | LIME | 0.3620 $\pm$ 0.3260 | 0.4075 | 0.5784 | 0.0547 |
| **10%** Sensor Drift | **SHAP** | **0.9291** $\pm$ 0.0859 | **0.8125** | **0.8438** | **0.0017** |
| | LIME | 0.2484 $\pm$ 0.2813 | 0.3725 | 0.5374 | 0.0583 |
| **20%** Operational Spike | **SHAP** | **0.9120** $\pm$ 0.0761 | **0.8000** | **0.7098** | **0.0032** |
| | LIME | 0.1809 $\pm$ 0.3559 | 0.3175 | 0.4299 | 0.0649 |

---

## 4. LIME Stochastic Seed Sensitivity Study

Because LIME samples instances from a continuous normal perturbation distribution to fit its local Ridge surrogate, we evaluated whether LIME produces identical explanations when run with different random seeds on the **exact same machine state**.

### Experimental Setup:
- Fixed test instances evaluated across 5 distinct seeds: `[10, 42, 99, 123, 777]`.
- Pairwise Spearman rank correlation computed between explanations generated across seeds.

### Empirical Findings:
- **Overall Mean Cross-Seed Spearman Correlation:** `0.4041`
- **Cross-Seed Correlation Standard Deviation:** `0.1716`
- **Observation:** Even without any input perturbation (input values identical to 6 decimal places), LIME's ranking of root causes fluctuates with a mean rank correlation of only $0.4041$. In contrast, TreeSHAP is strictly deterministic ($\rho = 1.0000$ across runs).

---

## 5. Scientific Discussion & Root Causes of Instability

### Why is SHAP significantly more stable?
1. **Deterministic Partition Traversal:** TreeSHAP calculates exact conditional expectations by traversing decision tree leaf partitions directly. As long as input perturbations do not cross a tree split threshold, the Shapley value remains identical.
2. **Game-Theoretic Axioms:** The efficiency and additivity axioms guarantee that feature contributions sum to the model margin, preventing arbitrary rescaling.

### Why is LIME prone to perturbation and seed instability?
1. **Curse of Dimensionality in Neighborhood Sampling:** In an 11-dimensional feature space, sampling 500 or even 2000 random points leaves vast empty volumes. The Ridge surrogate fits a hyperplane to sparse samples, making the resulting slope highly sensitive to random draws.
2. **Exponential Kernel Bandwidth Sensitivity:** LIME weights samples using an exponential kernel $\pi(z) = \exp(-D^2 / \sigma^2)$. Small shifts in the query instance displace the high-weight kernel center, radically altering sample weights.

---

## 6. Industrial Engineering Guidelines
1. **Mandate SHAP for Autonomous Shutdown Logic:** When automated SCADA systems trigger machine interlocks based on root cause attributions, SHAP must be used to prevent spurious shutdowns triggered by sensor jitter.
2. **Seed-Fixing Protocol for LIME:** If LIME is deployed for operator-facing dashboards due to its intuitive decision rules, maintenance software **must pin the random seed** or average explanations across an ensemble of runs (Bagged LIME).
3. **Sensor Quality Thresholds:** Real-time sensor error must be calibrated within $\le 5\%$ relative error to guarantee $>90\%$ root-cause ranking fidelity.
