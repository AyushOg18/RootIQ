# RootIQ: Industrial Data Quality & Schema Audit Report

## 1. Dataset Overview
- **Dataset Name:** UCI AI4I 2020 Predictive Maintenance Dataset
- **Domain:** CNC Milling Machine Synthetic Industrial Benchmark (representative of real milling machine telemetry)
- **Total Records:** 10,000
- **Total Features:** 14
- **Duplicate Records:** 0
- **Missing Values:** 0 (Clean, fully populated dataset)

## 2. Target Variable Analysis
- **Selected Target:** `Machine failure`
  - Class `0` (Normal Operation): 9,661 (96.61%)
  - Class `1` (Machine Failure): 339 (3.39%)
- **Imbalance Ratio:** 28.5:1 (Severe class imbalance)
- **Imbalance Handling Strategy:** Stratified data splitting, balanced class weighting (`class_weight='balanced'`), and evaluation via Precision, Recall, PR-AUC, and F1-score rather than naive accuracy.

## 3. Feature Leakage Prevention & Exclusions
To ensure realistic predictive operation and eliminate target leakage:
1. **Identifier Exclusions (`UDI`, `Product ID`):** Purely surrogate keys and unit identifiers with zero generalization value.
2. **Failure Mode Flag Exclusions (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`):** Tool Wear Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure, and Random Failure are post-hoc deterministic components of `Machine failure`. Including them would introduce catastrophic target leakage ($100\%$ artificial accuracy).
3. **Retained Predictive Features:**
   - `Type`: Quality variant (`L`, `M`, `H` representing Low, Medium, High product variants with 50/30/20% market share).
   - `Air temperature [K]`: Ambient temperature in Kelvin.
   - `Process temperature [K]`: Internal process temperature in Kelvin.
   - `Rotational speed [rpm]`: Spindle speed calculated from electrical power and torque.
   - `Torque [Nm]`: Resistance torque exerted by the cutter.
   - `Tool wear [min]`: Cumulative tool usage time in minutes.

## 4. Specific Failure Modes Breakdown
| Failure Mode Code | Description | Observed Cases |
| :--- | :--- | :--- |
| `TWF` | Tool Wear Failure | 46 |
| `HDF` | Heat Dissipation Failure | 115 |
| `PWF` | Power Failure | 95 |
| `OSF` | Overstrain Failure | 98 |
| `RNF` | Random Failure | 19 |

*Audit status: PASS. Dataset verified against industrial research requirements.*
