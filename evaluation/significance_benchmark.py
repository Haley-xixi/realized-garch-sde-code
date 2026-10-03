import sys

from pathlib import Path

import numpy as np

import pandas as pd

# ============================================================

# Project setup

# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(

    0,

    str(PROJECT_ROOT),

)

RESULTS_DIR = (

    PROJECT_ROOT

    / "results"

)

# ============================================================

# SDE experiment used as reference

# ============================================================

SDE_RESULTS_SUBDIR = (

    "proposed_SDE"

)

SDE_RESULTS_DIR = (

    RESULTS_DIR

    / SDE_RESULTS_SUBDIR

)

# ============================================================

# Import DM / MCS

# ============================================================

from DM_MCS import (

    ModelConfidenceSet,

    pairwise_dm_table,

)

# ============================================================

# Configuration

# ============================================================

SEED = 1313

QLIKE_EPS = 1e-8

# ------------------------------------------------------------

# Metrics

# ------------------------------------------------------------

METRICS = [

    "QLIKE",

    "MSE_var",

    "MAE_var",

]

# ------------------------------------------------------------

# DM

# ------------------------------------------------------------

DM_BENCHMARK = "SDE"

DM_TEST_TYPE = "DM"

# Two-sided:

# tests whether predictive accuracy differs in either direction.

DM_ALTERNATIVE = "two-sided"

FORECAST_HORIZON = 1

# DM significance level.

# For the two-sided DM test, reject equal predictive accuracy when

# p_value <= DM_ALPHA.

DM_ALPHA = 0.10

# ------------------------------------------------------------

# MCS

# ------------------------------------------------------------

MCS_ALPHA = 0.10

MCS_BOOTSTRAPS = 5000

MCS_ALGORITHM = "R"

# ============================================================

# Tickers

# ============================================================

TICKERS = [

    "SPX",

    "SSMI",

    "STI",

    "OMXSPI",

    "KS11",

    "OMXHPI",

    "SSEC",

    "KSE",

    "MXX",

    "IXIC",

    "OMXC20",

    "STOXX50E",

    "SMSI",

    "N225",

    "OSEAX",

    "RUT",

    "NSEI",

]

# ============================================================

# Output directories

# ============================================================

STAT_TEST_DIR = (

    RESULTS_DIR

    / "statistical_tests"

)

LOSS_DIR = (

    STAT_TEST_DIR

    / "losses"

)

QLIKE_DIR = (

    LOSS_DIR

    / "qlike"

)

MSE_VAR_DIR = (

    LOSS_DIR

    / "mse_var"

)

MAE_VAR_DIR = (

    LOSS_DIR

    / "mae_var"

)

DM_DIR = (

    STAT_TEST_DIR

    / "dm"

)

MCS_DIR = (

    STAT_TEST_DIR

    / "mcs"

)

ALIGNMENT_DIR = (

    STAT_TEST_DIR

    / "alignment"

)

for directory in [

    STAT_TEST_DIR,

    LOSS_DIR,

    QLIKE_DIR,

    MSE_VAR_DIR,

    MAE_VAR_DIR,

    DM_DIR,

    MCS_DIR,

    ALIGNMENT_DIR,

]:

    directory.mkdir(

        parents=True,

        exist_ok=True,

    )

# ============================================================

# Generic utilities

# ============================================================

def ensure_1d_float(

    values,

    *,

    name="array",

):

    x = np.asarray(

        values,

        dtype=np.float64,

    ).reshape(-1)

    if len(x) == 0:

        raise ValueError(

            f"{name} is empty."

        )

    if not np.all(

        np.isfinite(x)

    ):

        raise ValueError(

            f"{name} contains non-finite values."

        )

    return x

def load_array(

    path,

):

    path = Path(

        path

    )

    if not path.exists():

        raise FileNotFoundError(

            f"File not found: {path}"

        )

    return ensure_1d_float(

        np.load(path),

        name=str(path),

    )

# ============================================================

# Loss functions

# ============================================================

def qlike_loss(

    realized_var,

    forecast_var,

    eps=QLIKE_EPS,

):

    """

    Daily QLIKE loss:

        log(h_t) + RV_t / h_t

    where:

        RV_t = realized variance target

        h_t  = predicted variance

    Lower is better.

    """

    rv = ensure_1d_float(

        realized_var,

        name="realized_var",

    )

    h = ensure_1d_float(

        forecast_var,

        name="forecast_var",

    )

    if len(rv) != len(h):

        raise ValueError(

            "realized_var and forecast_var "

            "must have equal length."

        )

    rv = np.maximum(

        rv,

        eps,

    )

    h = np.maximum(

        h,

        eps,

    )

    return (

        np.log(h)

        + rv / h

    )

def mse_var_loss(

    realized_var,

    forecast_var,

):

    """

    Daily variance MSE loss:

        (RV_t - h_t)^2

    Lower is better.

    """

    rv = ensure_1d_float(

        realized_var,

        name="realized_var",

    )

    h = ensure_1d_float(

        forecast_var,

        name="forecast_var",

    )

    if len(rv) != len(h):

        raise ValueError(

            "realized_var and forecast_var "

            "must have equal length."

        )

    return (

        rv - h

    ) ** 2

def mae_var_loss(

    realized_var,

    forecast_var,

):

    """

    Daily variance MAE loss:

        |RV_t - h_t|

    Lower is better.

    """

    rv = ensure_1d_float(

        realized_var,

        name="realized_var",

    )

    h = ensure_1d_float(

        forecast_var,

        name="forecast_var",

    )

    if len(rv) != len(h):

        raise ValueError(

            "realized_var and forecast_var "

            "must have equal length."

        )

    return np.abs(

        rv - h

    )

# ============================================================

# Model loaders

# ============================================================

def load_sde(

    ticker,

):

    """

    Load SDE forecasts.

    The SDE supplies:

        - predicted integrated variance

        - realized-variance target

    The SDE realized-variance target is used as the common

    evaluation target after common-end alignment.

    """

    model_dir = (

        SDE_RESULTS_DIR

    )

    return {

        "pred_var":

            load_array(

                model_dir

                / (

                    f"{ticker}_rolling_"

                    "variance_pred.npy"

                )

            ),

        "rv_true":

            load_array(

                model_dir

                / (

                    f"{ticker}_rolling_"

                    "rv_target.npy"

                )

            ),

    }

def load_garch11(

    ticker,

):

    model_dir = (

        RESULTS_DIR

        / "GARCH11_rolling_50_50"

        / "predictions"

    )

    return {

        "pred_var":

            load_array(

                model_dir

                / (

                    f"GARCH11_rolling_"

                    f"{ticker}_pred_var.npy"

                )

            ),

        "rv_true":

            load_array(

                model_dir

                / (

                    f"GARCH11_rolling_"

                    f"{ticker}_true_rv.npy"

                )

            ),

    }

def load_realized_garch(

    ticker,

):

    model_dir = (

        RESULTS_DIR

        / "realized_garch_rolling_50_50"

    )

    return {

        "pred_var":

            load_array(

                model_dir

                / (

                    f"{ticker}_rg_"

                    "variance_pred.npy"

                )

            ),

        "rv_true":

            load_array(

                model_dir

                / (

                    f"{ticker}_rg_"

                    "rv_target.npy"

                )

            ),

    }

def load_heavy(

    ticker,

):

    model_dir = (

        RESULTS_DIR

        / "HEAVY_rolling_50_50"

        / "predictions"

    )

    return {

        "pred_var":

            load_array(

                model_dir

                / (

                    f"HEAVY_rolling_"

                    f"{ticker}_pred_var.npy"

                )

            ),

        "rv_true":

            load_array(

                model_dir

                / (

                    f"HEAVY_rolling_"

                    f"{ticker}_true_rv.npy"

                )

            ),

    }

def load_garch_ito(

    ticker,

):

    model_dir = (

        RESULTS_DIR

        / "GARCH_ito_RV_QMLE_rolling_50_50"

        / "predictions"

    )

    return {

        "pred_var":

            load_array(

                model_dir

                / (

                    f"GARCH_ito_rolling_"

                    f"{ticker}_pred_var.npy"

                )

            ),

        "rv_true":

            load_array(

                model_dir

                / (

                    f"GARCH_ito_rolling_"

                    f"{ticker}_true_rv.npy"

                )

            ),

    }

def load_har_qml(

    ticker,

):

    model_dir = (

        RESULTS_DIR

        / "HAR_RV_QML"

    )

    return {

        "pred_var":

            load_array(

                model_dir

                / (

                    f"{ticker}_HAR_QML_"

                    "h_raw_forecast.npy"

                )

            ),

        "rv_true":

            load_array(

                model_dir

                / (

                    f"{ticker}_HAR_QML_"

                    "RV_target.npy"

                )

            ),

    }

# ============================================================

# Load all models

# ============================================================

def load_models(

    ticker,

):

    return {

        "SDE":

            load_sde(

                ticker

            ),

        "GARCH-Ito":

            load_garch_ito(

                ticker

            ),

        "GARCH":

            load_garch11(

                ticker

            ),

        "Realized-GARCH":

            load_realized_garch(

                ticker

            ),

        "HEAVY":

            load_heavy(

                ticker

            ),

        "HAR-QML":

            load_har_qml(

                ticker

            ),

    }

# ============================================================

# Common-end alignment

# ============================================================

def align_models_by_length(

    ticker,

    original_models,

):

    """

    Length-based common-end alignment.

    Principle

    ---------

    The SDE remains the reference source for the realized-variance

    evaluation target.

    Procedure

    ---------

    1. Load the full saved forecast arrays for every model.

    2. Determine:

           common_n = minimum forecast length across models

    3. Keep the LAST common_n observations of every model.

    4. Use the LAST common_n observations of the SDE realized-variance

       target as the common evaluation target.

    Assumption

    ----------

    All saved rolling forecast sequences share the same ending date.

    This function performs no:

        - RV equality matching

        - automatic date-shift search

        - realized-variance rescaling

        - forecast rescaling

    """

    if "SDE" not in original_models:

        raise ValueError(

            f"{ticker}: SDE missing."

        )

    # ========================================================

    # SDE reference arrays

    # ========================================================

    sde_pred = ensure_1d_float(

        original_models[

            "SDE"

        ][

            "pred_var"

        ],

        name=(

            f"{ticker} "

            "SDE prediction"

        ),

    )

    sde_rv = ensure_1d_float(

        original_models[

            "SDE"

        ][

            "rv_true"

        ],

        name=(

            f"{ticker} "

            "SDE RV"

        ),

    )

    if len(sde_pred) != len(sde_rv):

        raise ValueError(

            f"{ticker}: SDE prediction/RV "

            "lengths are inconsistent: "

            f"{len(sde_pred)} vs "

            f"{len(sde_rv)}."

        )

    # ========================================================

    # Load every model and record original lengths

    # ========================================================

    loaded_models = {}

    model_lengths = {}

    for (

        model_name,

        model_data,

    ) in original_models.items():

        pred_var = ensure_1d_float(

            model_data[

                "pred_var"

            ],

            name=(

                f"{ticker} "

                f"{model_name} prediction"

            ),

        )

        rv_true = ensure_1d_float(

            model_data[

                "rv_true"

            ],

            name=(

                f"{ticker} "

                f"{model_name} RV"

            ),

        )

        if len(pred_var) != len(rv_true):

            raise ValueError(

                f"{ticker}: {model_name} "

                "prediction/RV length mismatch: "

                f"{len(pred_var)} vs "

                f"{len(rv_true)}."

            )

        loaded_models[

            model_name

        ] = {

            "pred_var":

                pred_var,

            "rv_true":

                rv_true,

        }

        model_lengths[

            model_name

        ] = len(

            pred_var

        )

    # ========================================================

    # Common sample size

    # ========================================================

    common_n = min(

        model_lengths.values()

    )

    if common_n < 2:

        raise ValueError(

            f"{ticker}: common sample is too short: "

            f"{common_n}."

        )

    # ========================================================

    # Keep LAST common_n observations for every model

    # ========================================================

    aligned_models = {}

    alignment_rows = []

    for (

        model_name,

        model_data,

    ) in loaded_models.items():

        original_n = len(

            model_data[

                "pred_var"

            ]

        )

        left_trim = (

            original_n

            - common_n

        )

        pred_aligned = (

            model_data[

                "pred_var"

            ][

                -common_n:

            ]

        )

        rv_aligned = (

            model_data[

                "rv_true"

            ][

                -common_n:

            ]

        )

        if not (

            len(pred_aligned)

            == len(rv_aligned)

            == common_n

        ):

            raise RuntimeError(

                f"{ticker}: {model_name} "

                "final alignment failed."

            )

        aligned_models[

            model_name

        ] = {

            "pred_var":

                pred_aligned,

            # Retained only for diagnostics.

            # Final evaluation uses common SDE RV.

            "rv_true":

                rv_aligned,

        }

        if left_trim == 0:

            alignment_action = (

                "original"

            )

        else:

            alignment_action = (

                f"drop first {left_trim}"

            )

        alignment_rows.append(

            {

                "ticker":

                    ticker,

                "model":

                    model_name,

                "original_n":

                    original_n,

                "left_trim":

                    left_trim,

                "alignment_action":

                    alignment_action,

                "final_n":

                    common_n,

            }

        )

    # ========================================================

    # Common evaluation RV from SDE

    # ========================================================

    common_rv = ensure_1d_float(

        sde_rv[

            -common_n:

        ],

        name=(

            f"{ticker} "

            "common SDE RV"

        ),

    )

    if len(common_rv) != common_n:

        raise RuntimeError(

            f"{ticker}: common SDE RV "

            "is not aligned."

        )

    # ========================================================

    # Alignment diagnostics

    # ========================================================

    alignment_df = pd.DataFrame(

        alignment_rows

    )

    alignment_df[

        "evaluation_rv"

    ] = (

        "SDE_common_RV"

    )

    alignment_df[

        "alignment_rule"

    ] = (

        "keep last common_n observations"

    )

    alignment_df[

        "rescaled"

    ] = False

    print(

        "\n"

        + "=" * 90

    )

    print(

        f"{ticker}: LENGTH-BASED "

        "COMMON-END ALIGNMENT"

    )

    print(

        "=" * 90

    )

    print(

        f"Common observations : "

        f"{common_n}"

    )

    print(

        f"Mean common SDE RV  : "

        f"{np.mean(common_rv):.8f}"

    )

    print(

        "\nAlignment:"

    )

    for _, row in (

        alignment_df.iterrows()

    ):

        print(

            f"{row['model']:<18} | "

            f"original="

            f"{int(row['original_n']):<6} | "

            f"left_trim="

            f"{int(row['left_trim']):<3} | "

            f"final="

            f"{int(row['final_n'])}"

        )

    return (

        aligned_models,

        common_rv,

        alignment_df,

    )

# ============================================================

# Construct common-date loss matrices

# ============================================================

def construct_loss_dfs(

    ticker,

):

    """

    Construct daily loss matrices for DM and MCS.

    Metrics:

        - QLIKE

        - variance MSE

        - variance MAE

    Every model is evaluated against the same common SDE

    realized-variance sequence.

    The common sample length is the minimum saved forecast length

    across models. The last common_n observations of every model

    are retained.

    """

    # ========================================================

    # Load original model outputs

    # ========================================================

    original_models = (

        load_models(

            ticker

        )

    )

    # ========================================================

    # SDE pre-alignment diagnostics

    # ========================================================

    sde_pred_full = ensure_1d_float(

        original_models[

            "SDE"

        ][

            "pred_var"

        ],

        name=(

            f"{ticker} "

            "SDE prediction"

        ),

    )

    sde_rv_full = ensure_1d_float(

        original_models[

            "SDE"

        ][

            "rv_true"

        ],

        name=(

            f"{ticker} "

            "SDE RV"

        ),

    )

    if len(sde_pred_full) != len(sde_rv_full):

        raise ValueError(

            f"{ticker}: SDE saved prediction/RV "

            "arrays have inconsistent lengths."

        )

    print(

        "\n"

        + "=" * 90

    )

    print(

        f"{ticker}: "

        "SDE SOURCE / PRE-ALIGNMENT CHECK"

    )

    print(

        "=" * 90

    )

    print(

        f"SDE directory      : "

        f"{SDE_RESULTS_DIR}"

    )

    print(

        f"SDE forecast count : "

        f"{len(sde_pred_full)}"

    )

    print(

        f"Mean SDE variance  : "

        f"{np.mean(sde_pred_full):.8f}"

    )

    print(

        f"Mean SDE raw RV    : "

        f"{np.mean(sde_rv_full):.8f}"

    )

    # ========================================================

    # Common-end length alignment

    # ========================================================

    (

        aligned_models,

        common_rv,

        alignment_df,

    ) = align_models_by_length(

        ticker=ticker,

        original_models=original_models,

    )

    common_n = len(

        common_rv

    )

    # ========================================================

    # Build daily loss matrices

    # ========================================================

    qlike_data = {}

    mse_var_data = {}

    mae_var_data = {}

    for (

        model_name,

        model_data,

    ) in aligned_models.items():

        pred_var = ensure_1d_float(

            model_data[

                "pred_var"

            ],

            name=(

                f"{ticker} "

                f"{model_name} "

                "final prediction"

            ),

        )

        if len(pred_var) != common_n:

            raise RuntimeError(

                f"{ticker}: {model_name} "

                f"has {len(pred_var)} predictions "

                f"after alignment; expected {common_n}."

            )

        # ----------------------------------------------------

        # Every model uses the SAME realized-variance target

        # ----------------------------------------------------

        qlike_data[

            model_name

        ] = qlike_loss(

            realized_var=common_rv,

            forecast_var=pred_var,

        )

        mse_var_data[

            model_name

        ] = mse_var_loss(

            realized_var=common_rv,

            forecast_var=pred_var,

        )

        mae_var_data[

            model_name

        ] = mae_var_loss(

            realized_var=common_rv,

            forecast_var=pred_var,

        )

    # ========================================================

    # Convert to DataFrames

    # ========================================================

    qlike_df = pd.DataFrame(

        qlike_data

    )

    mse_var_df = pd.DataFrame(

        mse_var_data

    )

    mae_var_df = pd.DataFrame(

        mae_var_data

    )

    loss_dfs = {

        "QLIKE":

            qlike_df,

        "MSE_var":

            mse_var_df,

        "MAE_var":

            mae_var_df,

    }

    # ========================================================

    # Loss matrix checks

    # ========================================================

    expected_columns = list(

        aligned_models.keys()

    )

    for (

        metric_name,

        loss_df,

    ) in loss_dfs.items():

        if list(

            loss_df.columns

        ) != expected_columns:

            raise RuntimeError(

                f"{ticker}: {metric_name} "

                "model columns are inconsistent."

            )

        if len(

            loss_df

        ) != common_n:

            raise RuntimeError(

                f"{ticker}: {metric_name} has "

                f"{len(loss_df)} observations; "

                f"expected {common_n}."

            )

        loss_array = loss_df.to_numpy(

            dtype=np.float64

        )

        if not np.all(

            np.isfinite(

                loss_array

            )

        ):

            raise RuntimeError(

                f"{ticker}: {metric_name} "

                "contains non-finite losses."

            )

    # ========================================================

    # Final diagnostics

    # ========================================================

    print(

        "\n"

        + "=" * 90

    )

    print(

        f"{ticker}: "

        "FINAL COMMON-DATE EVALUATION"

    )

    print(

        "=" * 90

    )

    print(

        f"Final common observations : "

        f"{common_n}"

    )

    print(

        f"Mean common raw RV        : "

        f"{np.mean(common_rv):.8f}"

    )

    for (

        metric_name,

        loss_df,

    ) in loss_dfs.items():

        print(

            f"\nMean {metric_name}:"

        )

        print(

            loss_df.mean()

        )

    # ========================================================

    # Add evaluation information to alignment CSV

    # ========================================================

    alignment_df = (

        alignment_df.copy()

    )

    alignment_df[

        "sde_results_subdir"

    ] = (

        SDE_RESULTS_SUBDIR

    )

    alignment_df[

        "common_n"

    ] = (

        common_n

    )

    alignment_df[

        "mean_common_rv"

    ] = float(

        np.mean(

            common_rv

        )

    )

    return {

        "QLIKE":

            qlike_df,

        "MSE_var":

            mse_var_df,

        "MAE_var":

            mae_var_df,

        "alignment":

            alignment_df,

    }

# ============================================================

# DM result annotation

# ============================================================

def annotate_dm_results(

    dm_df,

):

    """

    Add explicit 10% DM significance decisions and direction.

    All metrics in this script are lower-is-better.

    Therefore:

        - the smaller mean loss determines the better model;

        - statistical significance is determined only by the

          two-sided DM p-value.

    """

    required_columns = {

        "competitor",

        "mean_loss_benchmark",

        "mean_loss_competitor",

        "p_value",

    }

    missing_columns = (

        required_columns

        .difference(

            dm_df.columns

        )

    )

    if missing_columns:

        raise KeyError(

            "DM output is missing required columns: "

            f"{sorted(missing_columns)}"

        )

    result = dm_df.copy()

    # ========================================================

    # Validate p-values

    # ========================================================

    p_values = pd.to_numeric(

        result[

            "p_value"

        ],

        errors="coerce",

    )

    if p_values.isna().any():

        raise ValueError(

            "DM p_value contains missing or "

            "non-numeric values."

        )

    p_array = p_values.to_numpy(

        dtype=np.float64

    )

    if not np.all(

        np.isfinite(

            p_array

        )

    ):

        raise ValueError(

            "DM p_value contains non-finite values."

        )

    if np.any(

        (p_array < 0.0)

        | (p_array > 1.0)

    ):

        raise ValueError(

            "DM p_value must lie in [0, 1]."

        )

    # ========================================================

    # Validate mean losses

    # ========================================================

    benchmark_loss = pd.to_numeric(

        result[

            "mean_loss_benchmark"

        ],

        errors="coerce",

    )

    competitor_loss = pd.to_numeric(

        result[

            "mean_loss_competitor"

        ],

        errors="coerce",

    )

    if (

        benchmark_loss.isna().any()

        or competitor_loss.isna().any()

    ):

        raise ValueError(

            "DM mean-loss columns contain missing "

            "or non-numeric values."

        )

    benchmark_array = benchmark_loss.to_numpy(

        dtype=np.float64

    )

    competitor_array = competitor_loss.to_numpy(

        dtype=np.float64

    )

    if not (

        np.all(

            np.isfinite(

                benchmark_array

            )

        )

        and np.all(

            np.isfinite(

                competitor_array

            )

        )

    ):

        raise ValueError(

            "DM mean-loss columns contain "

            "non-finite values."

        )

    # ========================================================

    # Significance

    # ========================================================

    result[

        "DM_alpha"

    ] = float(

        DM_ALPHA

    )

    result[

        "significant_10pct"

    ] = (

        p_values

        <= DM_ALPHA

    )

    # ========================================================

    # Loss direction

    # ========================================================

    benchmark_better = (

        benchmark_loss

        < competitor_loss

    )

    competitor_better = (

        competitor_loss

        < benchmark_loss

    )

    result[

        "better_model"

    ] = np.where(

        benchmark_better,

        DM_BENCHMARK,

        np.where(

            competitor_better,

            result[

                "competitor"

            ].astype(str),

            "Tie",

        ),

    )

    significant = result[

        "significant_10pct"

    ].to_numpy(

        dtype=bool

    )

    competitor_names = result[

        "competitor"

    ].astype(str)

    result[

        "DM_conclusion_10pct"

    ] = np.where(

        ~significant,

        "No significant difference",

        np.where(

            benchmark_better,

            (

                f"{DM_BENCHMARK} "

                "significantly better"

            ),

            np.where(

                competitor_better,

                (

                    competitor_names

                    + " significantly better"

                ),

                (

                    "Significant DM result; "

                    "equal mean losses"

                ),

            ),

        ),

    )

    return result

# ============================================================

# MCS

# ============================================================

def run_mcs(

    loss_df,

):

    """

    Run the 90% Model Confidence Set.

    MCS_ALPHA = 0.10 corresponds to a 90% MCS.

    """

    mcs = ModelConfidenceSet(

        data=loss_df,

        alpha=MCS_ALPHA,

        B=MCS_BOOTSTRAPS,

        w=None,

        algorithm=MCS_ALGORITHM,

        random_state=SEED,

    )

    mcs.run()

    return mcs

def make_mcs_summary(

    ticker,

    metric,

    loss_df,

    mcs,

):

    """

    Construct one-row-per-model MCS summary.

    """

    rows = []

    included_models = set(

        mcs.included

    )

    for model_name in (

        loss_df.columns

    ):

        rows.append(

            {

                "ticker":

                    ticker,

                "metric":

                    metric,

                "model":

                    model_name,

                "mean_loss":

                    float(

                        loss_df[

                            model_name

                        ].mean()

                    ),

                "included_90pct_MCS":

                    (

                        model_name

                        in included_models

                    ),

            }

        )

    return pd.DataFrame(

        rows

    )

# ============================================================

# Main

# ============================================================

def main():

    print(

        "\n"

        + "=" * 100

    )

    print(

        "DM/MCS CONFIGURATION"

    )

    print(

        "=" * 100

    )

    print(

        f"SDE results directory : "

        f"{SDE_RESULTS_DIR}"

    )

    print(

        "Evaluation metrics     : "

        + ", ".join(

            METRICS

        )

    )

    print(

        "Alignment rule         : "

        "keep last common_n observations"

    )

    print(

        f"DM benchmark           : "

        f"{DM_BENCHMARK}"

    )

    print(

        f"DM alternative         : "

        f"{DM_ALTERNATIVE}"

    )

    print(

        f"DM alpha               : "

        f"{DM_ALPHA} "

        "(10% significance level)"

    )

    print(

        f"MCS alpha              : "

        f"{MCS_ALPHA} "

        "(90% MCS)"

    )

    print(

        f"MCS bootstraps         : "

        f"{MCS_BOOTSTRAPS}"

    )

    print(

        f"Random seed            : "

        f"{SEED}"

    )

    print(

        "=" * 100

    )

    all_dm_results = []

    all_mcs_results = []

    all_alignment_results = []

    successful_tickers = []

    failed_tickers = []

    # ========================================================

    # Ticker loop

    # ========================================================

    for ticker in TICKERS:

        print(

            "\n"

            + "=" * 100

        )

        print(

            f"Running DM/MCS: "

            f"{ticker}"

        )

        print(

            "=" * 100

        )

        try:

            # =================================================

            # Construct daily losses

            # =================================================

            result = (

                construct_loss_dfs(

                    ticker

                )

            )

            alignment_df = (

                result[

                    "alignment"

                ]

            )

            all_alignment_results.append(

                alignment_df

            )

            # Save ticker-specific alignment.

            alignment_df.to_csv(

                ALIGNMENT_DIR

                / (

                    f"{ticker}_alignment.csv"

                ),

                index=False,

            )

            metric_dfs = {

                "QLIKE":

                    result[

                        "QLIKE"

                    ],

                "MSE_var":

                    result[

                        "MSE_var"

                    ],

                "MAE_var":

                    result[

                        "MAE_var"

                    ],

            }

            # =================================================

            # Print mean losses

            # =================================================

            for (

                metric_name,

                loss_df,

            ) in metric_dfs.items():

                print(

                    f"\nMean {metric_name}:"

                )

                print(

                    loss_df.mean()

                )

            # =================================================

            # Save daily losses

            # =================================================

            metric_dfs[

                "QLIKE"

            ].to_csv(

                QLIKE_DIR

                / (

                    f"{ticker}_"

                    "qlike_losses.csv"

                ),

                index=False,

            )

            metric_dfs[

                "MSE_var"

            ].to_csv(

                MSE_VAR_DIR

                / (

                    f"{ticker}_"

                    "mse_var_losses.csv"

                ),

                index=False,

            )

            metric_dfs[

                "MAE_var"

            ].to_csv(

                MAE_VAR_DIR

                / (

                    f"{ticker}_"

                    "mae_var_losses.csv"

                ),

                index=False,

            )

            # =================================================

            # DM + MCS

            # =================================================

            for (

                metric_name,

                loss_df,

            ) in metric_dfs.items():

                print(

                    "\n"

                    + "-" * 80

                )

                print(

                    f"{ticker}: "

                    f"{metric_name}"

                )

                print(

                    "-" * 80

                )

                # =============================================

                # DM HAC lag

                # =============================================

                dm_hac_lags = int(

                    np.floor(

                        len(

                            loss_df

                        )

                        ** (

                            1.0

                            / 3.0

                        )

                    )

                )

                # =============================================

                # DM

                # =============================================

                dm_df = (

                    pairwise_dm_table(

                        losses=loss_df,

                        benchmark=(

                            DM_BENCHMARK

                        ),

                        test_type=(

                            DM_TEST_TYPE

                        ),

                        alternative=(

                            DM_ALTERNATIVE

                        ),

                        h=(

                            FORECAST_HORIZON

                        ),

                        hac_lags=(

                            dm_hac_lags

                        ),

                    )

                )

                dm_df = (

                    annotate_dm_results(

                        dm_df

                    )

                )

                dm_df.insert(

                    0,

                    "metric",

                    metric_name,

                )

                dm_df.insert(

                    0,

                    "ticker",

                    ticker,

                )

                all_dm_results.append(

                    dm_df

                )

                dm_df.to_csv(

                    DM_DIR

                    / (

                        f"{ticker}_dm_"

                        f"{metric_name.lower()}"

                        ".csv"

                    ),

                    index=False,

                )

                print(

                    "\nDM results "

                    "(10% significance level):"

                )

                dm_display_columns = [

                    "competitor",

                    "mean_loss_benchmark",

                    "mean_loss_competitor",

                    "mean_loss_diff",

                    "DM_stat",

                    "p_value",

                    "DM_alpha",

                    "significant_10pct",

                    "better_model",

                    "DM_conclusion_10pct",

                    "hac_lags",

                ]

                missing_display_columns = [

                    col

                    for col in dm_display_columns

                    if col not in dm_df.columns

                ]

                if missing_display_columns:

                    raise KeyError(

                        f"{ticker} {metric_name}: "

                        "DM output is missing columns "

                        "required for printing: "

                        f"{missing_display_columns}"

                    )

                print(

                    dm_df[

                        dm_display_columns

                    ].to_string(

                        index=False

                    )

                )

                # =============================================

                # MCS

                # =============================================

                mcs = run_mcs(

                    loss_df

                )

                mcs_df = (

                    make_mcs_summary(

                        ticker=ticker,

                        metric=metric_name,

                        loss_df=loss_df,

                        mcs=mcs,

                    )

                )

                all_mcs_results.append(

                    mcs_df

                )

                mcs_df.to_csv(

                    MCS_DIR

                    / (

                        f"{ticker}_mcs_"

                        f"{metric_name.lower()}"

                        ".csv"

                    ),

                    index=False,

                )

                print(

                    "\n90% MCS included:",

                    mcs.included,

                )

                print(

                    "90% MCS excluded:",

                    mcs.excluded,

                )

            successful_tickers.append(

                ticker

            )

        except Exception as exc:

            error_message = (

                f"{type(exc).__name__}: "

                f"{exc}"

            )

            failed_tickers.append(

                {

                    "ticker":

                        ticker,

                    "error":

                        error_message,

                }

            )

            print(

                f"\n{ticker} failed: "

                f"{error_message}"

            )

    # ========================================================

    # Combined alignment results

    # ========================================================

    if all_alignment_results:

        combined_alignment = pd.concat(

            all_alignment_results,

            ignore_index=True,

        )

        combined_alignment.to_csv(

            STAT_TEST_DIR

            / "all_tickers_alignment.csv",

            index=False,

        )

    # ========================================================

    # Combined DM results

    # ========================================================

    if all_dm_results:

        combined_dm = pd.concat(

            all_dm_results,

            ignore_index=True,

        )

        combined_dm.to_csv(

            STAT_TEST_DIR

            / "all_tickers_dm_results.csv",

            index=False,

        )

        for metric_name in METRICS:

            subset = combined_dm[

                combined_dm[

                    "metric"

                ]

                == metric_name

            ].copy()

            subset.to_csv(

                STAT_TEST_DIR

                / (

                    "all_tickers_dm_"

                    f"{metric_name.lower()}"

                    ".csv"

                ),

                index=False,

            )

            significant_subset = subset.loc[

                subset[

                    "significant_10pct"

                ].astype(bool)

            ].copy()

            significant_subset.to_csv(

                DM_DIR

                / (

                    "all_tickers_dm_"

                    f"{metric_name.lower()}"

                    "_significant_10pct.csv"

                ),

                index=False,

            )

        combined_significant_dm = combined_dm.loc[

            combined_dm[

                "significant_10pct"

            ].astype(bool)

        ].copy()

        combined_significant_dm.to_csv(

            STAT_TEST_DIR

            / (

                "all_tickers_dm_"

                "significant_10pct.csv"

            ),

            index=False,

        )

    # ========================================================

    # Combined MCS results

    # ========================================================

    if all_mcs_results:

        combined_mcs = pd.concat(

            all_mcs_results,

            ignore_index=True,

        )

        combined_mcs.to_csv(

            STAT_TEST_DIR

            / "all_tickers_mcs_results.csv",

            index=False,

        )

        for metric_name in METRICS:

            subset = combined_mcs[

                combined_mcs[

                    "metric"

                ]

                == metric_name

            ].copy()

            subset.to_csv(

                STAT_TEST_DIR

                / (

                    "all_tickers_mcs_"

                    f"{metric_name.lower()}"

                    ".csv"

                ),

                index=False,

            )

            if not subset.empty:

                pivot = subset.pivot(

                    index="ticker",

                    columns="model",

                    values="included_90pct_MCS",

                )

                available_tickers = [

                    ticker

                    for ticker in TICKERS

                    if ticker in pivot.index

                ]

                pivot = pivot.loc[

                    available_tickers

                ]

                # Keep model columns in the same order as load_models().

                preferred_model_order = [

                    "SDE",

                    "GARCH-Ito",

                    "GARCH",

                    "Realized-GARCH",

                    "HEAVY",

                    "HAR-QML",

                ]

                available_models = [

                    model

                    for model in preferred_model_order

                    if model in pivot.columns

                ]

                pivot = pivot[

                    available_models

                ]

                pivot.to_csv(

                    MCS_DIR

                    / (

                        "mcs_membership_"

                        f"{metric_name.lower()}"

                        ".csv"

                    )

                )

    # ========================================================

    # Failure log

    # ========================================================

    failed_path = (

        STAT_TEST_DIR

        / "failed_tickers.csv"

    )

    if failed_tickers:

        pd.DataFrame(

            failed_tickers

        ).to_csv(

            failed_path,

            index=False,

        )

    else:

        # Remove stale failure log from an earlier run.

        if failed_path.exists():

            failed_path.unlink()

    # ========================================================

    # Final summary

    # ========================================================

    print(

        "\n"

        + "=" * 100

    )

    print(

        "DM/MCS evaluation complete"

    )

    print(

        "=" * 100

    )

    print(

        f"Successful tickers: "

        f"{len(successful_tickers)}"

    )

    print(

        successful_tickers

    )

    print(

        f"\nFailed tickers: "

        f"{len(failed_tickers)}"

    )

    if failed_tickers:

        print(

            pd.DataFrame(

                failed_tickers

            ).to_string(

                index=False

            )

        )

    print(

        f"\nResults saved to:\n"

        f"{STAT_TEST_DIR}"

    )

# ============================================================

# Entry point

# ============================================================

if __name__ == "__main__":

    main()
