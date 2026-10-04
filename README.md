# Continual Learning for Industrial Monitoring

Comparing static and incremental machine learning models under concept drift using the NASA C-MAPSS turbofan engine degradation dataset.

---

## Overview

Industrial equipment degrades over time, causing sensor readings to drift away from their calibrated baselines. Machine learning models trained on historical data become progressively inaccurate as this drift accumulates, yet most industrial AI systems are deployed once and never updated.

This project investigates whether an incrementally-updated model can maintain accuracy under concept drift where a static model fails. Three models are compared across two drift levels, with all experiments tracked using MLflow.

---

## Key Result

At high concept drift, a streaming Hoeffding Tree achieved **18.8% lower RMSE** than a static Random Forest. At low drift, all three models performed equivalently.

| Drift Level | Static RF | SGD Incremental | Hoeffding Incremental |
|-------------|-----------|-----------------|----------------------|
| Low         | 56.81     | 56.53           | 56.99                |
| High        | 82.04     | 346.90          | **66.65**            |

The SGD Regressor's catastrophic failure at high drift demonstrates that not all incremental methods are robust to concept drift — algorithm choice matters as much as the decision to update incrementally.

---

## Dataset

**NASA C-MAPSS FD001** — Turbofan Engine Degradation Simulation

- 100 turbofan engines
- 21 sensor measurements + 3 operational settings per cycle
- Each engine runs from healthy state to failure
- 17 active sensors after removing constant columns

The dataset is not committed to this repository. Download `train_FD001.txt` and place it in `data/`.

Source: [Hugging Face — SoyVitou/NASA-C-MAPSS-Turbofan-Engine](https://huggingface.co/datasets/SoyVitou/NASA-C-MAPSS-Turbofan-Engine)

---

## Methodology

### 1. Data Preparation
- Compute Remaining Useful Life (RUL) target
- Drop constant sensors
- Split 70 engines for training / 30 for testing (by engine, not by row)

### 2. Concept Drift Simulation
Gaussian noise is added to the test set, scaled by `cycle^1.5`. This produces negligible drift at early cycles and substantial drift at late cycles.

Two drift levels:
- **Low:** noise_rate = 0.001
- **High:** noise_rate = 0.010

### 3. Models Compared

| Model | Library | Type |
|---|---|---|
| Random Forest | scikit-learn | Static (trained once) |
| SGD Regressor | scikit-learn | Incremental (`partial_fit`) |
| Hoeffding Tree | River ML | Streaming (per-sample) |

### 4. Evaluation
RMSE on drifted test data. Sensitivity analysis across drift levels.

### 5. MLOps
All parameters, metrics, and artifacts tracked in MLflow (SQLite backend).

---

## Repository Structure
