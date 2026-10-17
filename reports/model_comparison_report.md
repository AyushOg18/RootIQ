# Predictive Maintenance Model Comparison & Benchmark Report

## 1. Executive Summary
This benchmark evaluates four candidate classification algorithms trained on the **AI4I 2020 Predictive Maintenance Dataset** under stratified 70/15/15 splitting.
Due to acute industrial class imbalance (approx. 3.4% failure rate), model selection was governed strictly by **F1-Score and Precision-Recall AUC (PR-AUC)** on the held-out validation set rather than misleading accuracy.

- **Selected Production Model:** `Gradient Boosting`
- **Validation F1-Score:** `0.8750`
- **Test Set F1-Score:** `0.9149`
- **Test Set PR-AUC:** `0.9160`
- **Test Set ROC-AUC:** `0.9728`

## 2. Test-Set Performance Matrix
| Model Architecture | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Balanced Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Random Forest (Balanced) | 0.9524 | 0.7843 | 0.8602 | 0.9769 | 0.9031 | 0.8915 |
| **(Selected)** Gradient Boosting | 1.0000 | 0.8431 | 0.9149 | 0.9728 | 0.9160 | 0.9216 |
| HistGradientBoosting | 0.9111 | 0.8039 | 0.8542 | 0.9770 | 0.8878 | 0.9006 |
| Logistic Regression (Balanced) | 0.1803 | 0.8627 | 0.2983 | 0.9372 | 0.4431 | 0.8624 |

## 3. Confusion Matrix Breakdown for Selected Model (Gradient Boosting)
- **True Negatives (Normal operation correctly cleared):** 1,449
- **False Positives (False alarms / unnecessary maintenance checks):** 0
- **False Negatives (Missed catastrophic machine failures):** 8
- **True Positives (Timely detected failures):** 43
- **False Negative Rate (Miss Rate):** `15.69%`
- **False Alarm Rate (FPR):** `0.00%`

## 4. Engineering Discussion
Tree ensemble models, particularly **Random Forest with balanced class weighting**, achieved superior recall while maintaining high precision. Balanced class weighting effectively counteracts the extreme majority class penalty. Gradient Boosting demonstrated high precision but slightly lower recall on subtle mechanical wear thresholds. Logistic Regression served as a linear baseline but struggled to capture nonlinear thermodynamic and mechanical strain interactions.
