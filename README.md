# Structured SDEs for Realized GARCH: Continuous-Time Volatility Modelling and Component-Level Analysis

This repository contains the code and empirical results for a continuous-time realized-volatility forecasting framework based on a structured stochastic differential equation (SDE). It also includes four benchmark models, component-level ablation studies, and statistical forecast evaluation.

## Repository Structure

```text
realized-garch-sde/

├── data/
├── evaluation/
├── experiments/
├── models/
├── results/
├── SDE_Garch_utils/
├── README.md
└── requirements.txt
```

## Data

The `data/` folder contains the dataset used in the forecasting experiments.

The experiment scripts load the data directly from this directory.

## Models

The `models/` folder contains the implementations of the proposed model and five benchmark models:

* Proposed Realized-GARCH-SDE
* GARCH-Itô RV-QMLE
* GARCH(1,1)
* HEAVY
* Realized GARCH
* HAR-QML


The proposed SDE implementation also supports the ablation specifications through Boolean configuration switches, so separate model implementations are not required for each ablation.

## Experiments

The `experiments/` folder contains the scripts used to run the forecasting experiments for:

* Proposed Realized-GARCH-SDE
* GARCH-Itô RV-QMLE
* GARCH(1,1)
* HEAVY
* Realized GARCH
* HAR-QML

The proposed SDE experiment can also be run under the ablation settings used in the paper by changing the corresponding Boolean switches.

The main ablation specifications are:

1. **No realized-measure feedback:** the feedback coefficient is fixed at $\gamma = 0$.

2. **Independent Brownian motions:** the return-state Brownian dependence is removed by setting the corresponding correlation parameters to zero.

When the ablation switches are disabled, the experiment corresponds to the full proposed SDE specification.

## SDE Utilities

The `SDE_Garch_utils/` folder contains shared utility functions used by the proposed SDE experiments.

```text
SDE_Garch_utils/
├── __init__.py
└── utils1.py
```

## Evaluation

The `evaluation/` folder contains the statistical evaluation scripts:

```text
evaluation/

├── DM_MCS.py
├── gamma_ablation.py
├── indepbm_ablation.py
└── significance_benchmark.py
```

### `DM_MCS.py`

Performs the Diebold-Mariano (DM) tests and Model Confidence Set (MCS) analysis used to evaluate predictive performance.

### `significance_ablation.py`

Evaluates the statistical significance of forecast-loss differences between the full proposed SDE and its restricted ablation specifications.

### `significance_benchmark.py`

Evaluates the statistical significance of forecast-loss differences between the proposed SDE and the benchmark models.

## Results

The `results/` folder contains the final forecasting and statistical outputs used in the workshop paper.

```text
results/

├── ablation_gamma0/
├── ablation_indep_bm/
├── GARCH_ito_RV_QMLE_rolling_50_50/
├── GARCH11_rolling_50_50/
├── HEAVY_rolling_50_50/
├── HAR_RV_QML/
├── proposed_SDE/
├── realized_garch_rolling_50_50/
├── statistical_test_ablation_gamma/
├── statistical_test_ablation_indepbm/
└── statistical_tests/
```

The folders correspond to:

* `proposed_SDE/` — results for the full proposed SDE.
* `mean_reverting_z_gamma0/` — ablation results with realized-measure feedback removed by fixing $\gamma = 0$.
* `mean_reverting_z_independent_BM_naive/` — results for the independent-Brownian specification.
* `GARCH_ito_RV_QMLE_rolling_50_50/` — GARCH-Itô RV-QMLE benchmark results.
* `GARCH11_rolling_50_50/` — GARCH(1,1) benchmark results.
* `HEAVY_rolling_50_50/` — HEAVY benchmark results.
* `HAR_RV_QML/` — HAR-RV-QML benchmark results.
* `realized_garch_rolling_50_50/` — Realized GARCH benchmark results.
* `statistical_tests/` — statistical comparison results, including DM and MCS outputs.

## Forecast Evaluation

The models are evaluated under a common rolling out-of-sample forecasting framework.

The initial estimation window uses the first half of each time series, after which one-step-ahead forecasts are generated over the remaining observations.

For the proposed SDE, the one-day-ahead forecast target is integrated variance,

$$
H_{n+1}
=
\int_n^{n+1} s_t\,dt,
$$

where $s_t$ denotes the latent instantaneous variance process.

Forecast performance is evaluated using:

* QLIKE
* Variance MSE
* Variance MAE
* Return negative log-likelihood (Return NLL)

## Statistical Tests

Predictive accuracy is evaluated using two complementary procedures.

### Diebold-Mariano Test

The Diebold-Mariano test is used for pairwise comparisons of predictive accuracy between competing models.

It is used for both:

* proposed SDE versus benchmark comparisons; and
* full SDE versus ablation comparisons.

### Model Confidence Set

The Model Confidence Set procedure is used to identify the set of statistically superior forecasting models across the competing specifications.

The reported analysis uses a 90% Model Confidence Set.

## Running the Experiments

Run all scripts from the repository root.

```bash
python experiments/run_proposedSDE.py
python experiments/run_GARCH.py
python experiments/run_GARCH_ito.py
python experiments/run_HEAVY.py
python experiments/run_RG.py
python experiments/run_HARRV.py
```

For the proposed SDE, the full model and ablation specifications are selected using the Boolean switches defined in `run_proposedSDE.py`.

After generating the forecasts, run the statistical evaluation scripts with:

```bash
python evaluation/significance_benchmark.py
python evaluation/significance_ablation.py
python evaluation/DM_MCS.py
```

## Reproducibility

The repository separates model implementation, experiment execution, saved results, and statistical evaluation:

```text
models/ + SDE_Garch_utils/
          │
          ▼
     experiments/
          │
          ▼
       results/
          │
          ▼
     evaluation/
```

This structure allows the reported results to be reproduced either by rerunning the forecasting experiments or by applying the evaluation scripts to the included final forecast outputs.
