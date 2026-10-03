from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================================================
# Path setup
# =========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "rv.pkl"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "HAR_RV_QML"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# Imports
# =========================================================

from SDE_Garch_utils.utils1 import (
    build_data_dict,
    load_rv,
)

from models.model_HARRV import (
    fixed_half_sample_rolling_harrv_qml,
)


# =========================================================
# Configuration
# =========================================================

EPS = 1e-8

REFIT_EVERY = 21
WEEKLY_WINDOW = 5
MONTHLY_WINDOW = 22

MAXITER = 1000
XTOL = 1e-6
FTOL = 1e-8

ALL_THESIS_TICKERS = [
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

SELECTED_TICKERS = ALL_THESIS_TICKERS



# =========================================================
# Helpers
# =========================================================

def ensure_1d_float(
    values,
    *,
    name: str,
    positive: bool = False,
    eps: float = EPS,
) -> np.ndarray:
    arr = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    if arr.size == 0:
        raise ValueError(
            f"{name} is empty."
        )

    if not np.all(
        np.isfinite(arr)
    ):
        raise ValueError(
            f"{name} contains non-finite values."
        )

    if positive:
        arr = np.maximum(
            arr,
            eps,
        )

    return arr


def get_full_price_rv_arrays(
    data_dict,
    ticker: str,
):
    """
    Recover original full log-price and raw-RV chronology.

    No first observation is dropped. Therefore N, N//2,
    the first OOS target, and OOS dates remain aligned
    with the other benchmarks.
    """
    data = data_dict[
        ticker
    ][
        "train"
    ]

    log_price_train = ensure_1d_float(
        data["y_train"],
        name=f"{ticker}.y_train",
    )
    log_price_test = ensure_1d_float(
        data["y_test"],
        name=f"{ticker}.y_test",
    )

    rv_train_raw = ensure_1d_float(
        data["rv_train"],
        name=f"{ticker}.rv_train",
        positive=True,
    )
    rv_test_raw = ensure_1d_float(
        data["rv_test"],
        name=f"{ticker}.rv_test",
        positive=True,
    )

    if len(
        log_price_train
    ) != len(
        rv_train_raw
    ):
        raise ValueError(
            f"{ticker}: training price/RV length mismatch."
        )

    if len(
        log_price_test
    ) != len(
        rv_test_raw
    ):
        raise ValueError(
            f"{ticker}: test price/RV length mismatch."
        )

    log_price_full = np.concatenate(
        [
            log_price_train,
            log_price_test,
        ]
    )

    rv_full_raw = np.concatenate(
        [
            rv_train_raw,
            rv_test_raw,
        ]
    )

    if len(
        log_price_full
    ) != len(
        rv_full_raw
    ):
        raise RuntimeError(
            f"{ticker}: full price/RV chronology is misaligned."
        )

    return (
        log_price_full,
        rv_full_raw,
    )


# =========================================================
# Evaluation metrics
# =========================================================

def qlike_rv(
    predicted_h_raw,
    actual_rv,
    eps: float = EPS,
) -> float:
    """
    QLIKE:

        log(h_t) + RV_t / h_t

    Only here, use h_t = max(h_t_raw, eps).
    """
    pred = ensure_1d_float(
        predicted_h_raw,
        name="predicted_h_raw",
    )
    true = ensure_1d_float(
        actual_rv,
        name="actual_rv",
        positive=True,
    )

    if len(pred) != len(true):
        raise ValueError(
            "QLIKE arrays have different lengths."
        )

    h = np.maximum(
        pred,
        eps,
    )
    rv = np.maximum(
        true,
        eps,
    )

    return float(
        np.mean(
            np.log(h)
            + rv / h
        )
    )


def mse_variance(
    predicted_h_raw,
    actual_rv,
) -> float:
    """
    Variance MSE uses the RAW HAR forecast exactly as produced.
    """
    pred = ensure_1d_float(
        predicted_h_raw,
        name="predicted_h_raw",
    )
    true = ensure_1d_float(
        actual_rv,
        name="actual_rv",
        positive=True,
    )

    if len(pred) != len(true):
        raise ValueError(
            "MSE arrays have different lengths."
        )

    return float(
        np.mean(
            (
                true
                - pred
            )
            ** 2
        )
    )


def mae_variance(
    predicted_h_raw,
    actual_rv,
) -> float:
    """
    Variance MAE uses the RAW HAR forecast exactly as produced.
    """
    pred = ensure_1d_float(
        predicted_h_raw,
        name="predicted_h_raw",
    )
    true = ensure_1d_float(
        actual_rv,
        name="actual_rv",
        positive=True,
    )

    if len(pred) != len(true):
        raise ValueError(
            "MAE arrays have different lengths."
        )

    return float(
        np.mean(
            np.abs(
                true
                - pred
            )
        )
    )


def mse_volatility(
    predicted_h_raw,
    actual_rv,
    eps: float = EPS,
) -> float:
    """
    sqrt() requires nonnegative input, so use max(h_raw, eps).
    """
    pred = ensure_1d_float(
        predicted_h_raw,
        name="predicted_h_raw",
    )
    true = ensure_1d_float(
        actual_rv,
        name="actual_rv",
        positive=True,
    )

    if len(pred) != len(true):
        raise ValueError(
            "Volatility-MSE arrays have different lengths."
        )

    pred_vol = np.sqrt(
        np.maximum(
            pred,
            eps,
        )
    )
    true_vol = np.sqrt(
        np.maximum(
            true,
            eps,
        )
    )

    return float(
        np.mean(
            (
                true_vol
                - pred_vol
            )
            ** 2
        )
    )


def mae_volatility(
    predicted_h_raw,
    actual_rv,
    eps: float = EPS,
) -> float:
    """
    sqrt() requires nonnegative input, so use max(h_raw, eps).
    """
    pred = ensure_1d_float(
        predicted_h_raw,
        name="predicted_h_raw",
    )
    true = ensure_1d_float(
        actual_rv,
        name="actual_rv",
        positive=True,
    )

    if len(pred) != len(true):
        raise ValueError(
            "Volatility-MAE arrays have different lengths."
        )

    pred_vol = np.sqrt(
        np.maximum(
            pred,
            eps,
        )
    )
    true_vol = np.sqrt(
        np.maximum(
            true,
            eps,
        )
    )

    return float(
        np.mean(
            np.abs(
                true_vol
                - pred_vol
            )
        )
    )


# =========================================================
# Plotting / saving
# =========================================================

def save_variance_plot(
    *,
    ticker: str,
    actual_rv: np.ndarray,
    predicted_h_raw: np.ndarray,
):
    fig, ax = plt.subplots(
        figsize=(
            12,
            5,
        )
    )

    ax.plot(
        actual_rv,
        label="Actual raw RV",
        linewidth=1.4,
    )
    ax.plot(
        predicted_h_raw,
        label="HAR-QML raw h forecast",
        linewidth=1.0,
    )

    ax.set_title(
        f"{ticker}: HAR-QML forecast vs raw RV"
    )
    ax.set_xlabel(
        "OOS observation"
    )
    ax.set_ylabel(
        "Variance"
    )
    ax.grid(
        True,
        alpha=0.3,
    )
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR
        / f"{ticker}_HAR_QML_RV_forecast.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_progress(
    results,
):
    pd.DataFrame(
        results
    ).to_csv(
        RESULTS_DIR
        / "HAR_QML_RV_evaluation_results.csv",
        index=False,
    )


# =========================================================
# One ticker
# =========================================================

def run_one_ticker(
    *,
    data_dict,
    ticker: str,
):
    print(
        "\n"
        + "=" * 100
    )
    print(
        f"HAR-QML / RAW-RV EVALUATION: {ticker}"
    )
    print(
        "=" * 100
    )

    (
        log_price_full,
        rv_full_raw,
    ) = get_full_price_rv_arrays(
        data_dict,
        ticker,
    )

    n_total = len(
        rv_full_raw
    )
    rolling_window_size = (
        n_total // 2
    )
    first_oos_index = (
        rolling_window_size
    )
    n_oos_expected = (
        n_total
        - rolling_window_size
    )

    print(
        "\n=== Common 50:50 design ==="
    )
    print(
        f"N total                    : {n_total}"
    )
    print(
        f"Rolling-window size        : {rolling_window_size}"
    )
    print(
        f"First OOS target index     : {first_oos_index}"
    )
    print(
        f"Expected OOS forecasts     : {n_oos_expected}"
    )
    print(
        "Estimation objective       : Gaussian return QML"
    )
    print(
        "HAR regressors             : raw RV d/w/m"
    )
    print(
        "Beta constraints            : none"
    )
    print(
        f"Likelihood/QLIKE h floor    : max(h_raw, {EPS:g})"
    )
    print(
        "Evaluation target           : raw RV"
    )

    # =====================================================
    # Rolling HAR-QML
    # =====================================================

    rolling_result = (
        fixed_half_sample_rolling_harrv_qml(
            ticker=ticker,
            rv_full=rv_full_raw,
            log_price_full=log_price_full,
            window_size=rolling_window_size,
            refit_every=REFIT_EVERY,
            weekly_window=WEEKLY_WINDOW,
            monthly_window=MONTHLY_WINDOW,
            eps=EPS,
        )
    )

    predicted_h_raw = ensure_1d_float(
        rolling_result[
            "predicted_h_raw"
        ],
        name="predicted_h_raw",
    )

    predicted_h_safe = ensure_1d_float(
        rolling_result[
            "predicted_h_safe"
        ],
        name="predicted_h_safe",
        positive=True,
    )

    actual_raw_rv = ensure_1d_float(
        rolling_result[
            "actual_raw_rv"
        ],
        name="actual_raw_rv",
        positive=True,
    )

    target_indices = np.asarray(
        rolling_result[
            "target_indices"
        ],
        dtype=np.int64,
    ).reshape(-1)

    if len(
        predicted_h_raw
    ) != n_oos_expected:
        raise RuntimeError(
            "HAR-QML OOS length does not match common 50:50 design."
        )

    expected_target_indices = np.arange(
        first_oos_index,
        n_total,
        dtype=np.int64,
    )

    if not np.array_equal(
        target_indices,
        expected_target_indices,
    ):
        raise RuntimeError(
            "HAR-QML target indices do not match common OOS dates."
        )

    if not np.array_equal(
        actual_raw_rv,
        rv_full_raw[
            target_indices
        ],
    ):
        raise RuntimeError(
            "HAR-QML actual RV targets are misaligned."
        )

    returns_oos = (
        100.0
        * (
            log_price_full[
                target_indices
            ]
            - log_price_full[
                target_indices
                - 1
            ]
        )
    )

    if not (
        len(predicted_h_raw)
        == len(predicted_h_safe)
        == len(actual_raw_rv)
        == len(returns_oos)
        == n_oos_expected
    ):
        raise RuntimeError(
            "HAR-QML OOS arrays are not aligned."
        )

    # =====================================================
    # Diagnostics
    # =====================================================

    n_nonpositive_forecasts = int(
        np.sum(
            predicted_h_raw
            <= 0.0
        )
    )

    print(
        "\n=== OOS forecast diagnostics ==="
    )
    print(
        f"Generated OOS forecasts    : {len(predicted_h_raw)}"
    )
    print(
        f"Mean predicted h_raw       : {np.mean(predicted_h_raw):.8f}"
    )
    print(
        f"Mean predicted h_safe      : {np.mean(predicted_h_safe):.8f}"
    )
    print(
        f"Mean OOS raw RV            : {np.mean(actual_raw_rv):.8f}"
    )
    print(
        f"Mean OOS return squared    : {np.mean(returns_oos ** 2):.8f}"
    )
    print(
        f"Min predicted h_raw        : {np.min(predicted_h_raw):.8f}"
    )
    print(
        f"Max predicted h_raw        : {np.max(predicted_h_raw):.8f}"
    )
    print(
        f"Non-positive forecasts     : {n_nonpositive_forecasts}"
    )

    # =====================================================
    # Raw-RV evaluation
    # =====================================================

    qlike = qlike_rv(
        predicted_h_raw,
        actual_raw_rv,
    )

    mse_var = mse_variance(
        predicted_h_raw,
        actual_raw_rv,
    )

    mae_var = mae_variance(
        predicted_h_raw,
        actual_raw_rv,
    )

    mse_vol = mse_volatility(
        predicted_h_raw,
        actual_raw_rv,
    )

    mae_vol = mae_volatility(
        predicted_h_raw,
        actual_raw_rv,
    )

    print(
        "\n=== HAR-QML OOS metrics evaluated on raw RV ==="
    )
    print(
        f"QLIKE_RV   : {qlike:.8f}"
    )
    print(
        f"MSE_var_RV : {mse_var:.8f}"
    )
    print(
        f"MAE_var_RV : {mae_var:.8f}"
    )
    print(
        f"MSE_vol_RV : {mse_vol:.8f}"
    )
    print(
        f"MAE_vol_RV : {mae_vol:.8f}"
    )

    # =====================================================
    # Save parameter history
    # =====================================================

    parameter_df = pd.DataFrame(
        rolling_result[
            "parameter_history"
        ]
    )

    parameter_df.to_csv(
        RESULTS_DIR
        / f"{ticker}_HAR_QML_parameter_history.csv",
        index=False,
    )

    parameter_refit_df = (
        parameter_df.loc[
            parameter_df[
                "refit"
            ].astype(bool)
        ]
        .copy()
    )

    parameter_refit_df.to_csv(
        RESULTS_DIR
        / f"{ticker}_HAR_QML_parameter_refits.csv",
        index=False,
    )

    if parameter_refit_df.empty:
        raise RuntimeError(
            "No HAR-QML refit rows were recorded."
        )

    # =====================================================
    # Save arrays
    # =====================================================

    np.save(
        RESULTS_DIR
        / f"{ticker}_HAR_QML_h_raw_forecast.npy",
        predicted_h_raw,
    )

    np.save(
        RESULTS_DIR
        / f"{ticker}_HAR_QML_h_safe_forecast.npy",
        predicted_h_safe,
    )

    np.save(
        RESULTS_DIR
        / f"{ticker}_HAR_QML_RV_target.npy",
        actual_raw_rv,
    )

    np.save(
        RESULTS_DIR
        / f"{ticker}_HAR_QML_returns_target.npy",
        returns_oos,
    )

    np.save(
        RESULTS_DIR
        / f"{ticker}_HAR_QML_target_indices.npy",
        target_indices,
    )

    np.savez(
        RESULTS_DIR
        / f"{ticker}_HAR_QML_complete.npz",
        predicted_h_raw=predicted_h_raw,
        predicted_h_safe=predicted_h_safe,
        actual_raw_rv=actual_raw_rv,
        returns_oos=returns_oos,
        target_indices=target_indices,
    )

    save_variance_plot(
        ticker=ticker,
        actual_rv=actual_raw_rv,
        predicted_h_raw=predicted_h_raw,
    )

    # =====================================================
    # Summary row
    # =====================================================

    result = {
        "ticker":
            ticker,

        "n_total":
            int(
                n_total
            ),

        "rolling_window_size":
            int(
                rolling_window_size
            ),

        "n_oos":
            int(
                len(
                    predicted_h_raw
                )
            ),

        "refit_every":
            int(
                REFIT_EVERY
            ),

        "window_step":
            int(
                REFIT_EVERY
            ),

        "weekly_window":
            int(
                WEEKLY_WINDOW
            ),

        "monthly_window":
            int(
                MONTHLY_WINDOW
            ),

        "mean_predicted_variance":
            float(
                np.mean(
                    predicted_h_raw
                )
            ),

        "mean_oos_raw_RV":
            float(
                np.mean(
                    actual_raw_rv
                )
            ),

        "mean_oos_return_squared":
            float(
                np.mean(
                    returns_oos
                    ** 2
                )
            ),

        "min_predicted_variance":
            float(
                np.min(
                    predicted_h_raw
                )
            ),

        "max_predicted_variance":
            float(
                np.max(
                    predicted_h_raw
                )
            ),

        "n_nonpositive_forecasts":
            int(
                n_nonpositive_forecasts
            ),

        "QLIKE_RV":
            float(
                qlike
            ),

        "MSE_vol_RV":
            float(
                mse_vol
            ),

        "MSE_var_RV":
            float(
                mse_var
            ),

        "MAE_vol_RV":
            float(
                mae_vol
            ),

        "MAE_var_RV":
            float(
                mae_var
            ),

        "final_beta_0":
            float(
                parameter_refit_df[
                    "beta_0"
                ].iloc[-1]
            ),

        "final_beta_daily":
            float(
                parameter_refit_df[
                    "beta_daily"
                ].iloc[-1]
            ),

        "final_beta_weekly":
            float(
                parameter_refit_df[
                    "beta_weekly"
                ].iloc[-1]
            ),

        "final_beta_monthly":
            float(
                parameter_refit_df[
                    "beta_monthly"
                ].iloc[-1]
            ),

        "mean_beta_0":
            float(
                parameter_refit_df[
                    "beta_0"
                ].mean()
            ),

        "mean_beta_daily":
            float(
                parameter_refit_df[
                    "beta_daily"
                ].mean()
            ),

        "mean_beta_weekly":
            float(
                parameter_refit_df[
                    "beta_weekly"
                ].mean()
            ),

        "mean_beta_monthly":
            float(
                parameter_refit_df[
                    "beta_monthly"
                ].mean()
            ),

        "n_refits":
            int(
                rolling_result[
                    "n_refits"
                ]
            ),

        "n_refits_accepted":
            int(
                rolling_result[
                    "n_refits_accepted"
                ]
            ),

        "model":
            "HAR-QML",

        "loss_type":
            "Gaussian_return_QML",

        "estimation_realized_measure":
            "raw_RV_as_HAR_regressors_only",

        "forecast_output":
            "raw_level_HAR_h",

        "evaluation_target":
            "raw_RV",

        "beta_constraints":
            "none",

        "positivity_handling":
            "max(h_raw, EPS) only for QML/QLIKE/sqrt",

        "evaluation":
            "50:50 fixed rolling window, refit every 21 days",
    }

    return result


# =========================================================
# Main
# =========================================================

def main():
    print(
        "\n"
        + "=" * 100
    )
    print(
        "HAR-QML BENCHMARK WITH RAW-RV EVALUATION"
    )
    print(
        "=" * 100
    )

    print(
        f"Data path              : {DATA_PATH}"
    )
    print(
        f"Results directory      : {RESULTS_DIR}"
    )
    print(
        f"Selected tickers       : {SELECTED_TICKERS}"
    )
    print(
        "HAR specification      : level HAR"
    )
    print(
        "Estimation objective   : Gaussian return QML"
    )
    print(
        "HAR inputs             : raw RV daily/weekly/monthly"
    )
    print(
        "Beta constraints       : none"
    )
    print(
        "Evaluation target      : raw RV"
    )
    print(
        f"Refit every            : {REFIT_EVERY} days"
    )

    df = load_rv(
        filepath=str(
            DATA_PATH
        )
    )

    data_dict = build_data_dict(
        df
    )

    missing_tickers = [
        ticker
        for ticker in SELECTED_TICKERS
        if ticker not in data_dict
    ]

    if missing_tickers:
        raise KeyError(
            "Missing tickers in data_dict: "
            f"{missing_tickers}"
        )

    results = []

    for ticker in SELECTED_TICKERS:
        try:
            result = run_one_ticker(
                data_dict=data_dict,
                ticker=ticker,
            )

            results.append(
                result
            )

            save_progress(
                results
            )

            print(
                f"\n{ticker} completed successfully."
            )

        except Exception as exc:
            error_message = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            print(
                f"\nFAILED {ticker}: "
                f"{error_message}"
            )

            results.append(
                {
                    "ticker":
                        ticker,
                    "model":
                        "HAR-QML",
                    "error":
                        error_message,
                }
            )

            save_progress(
                results
            )

    results_df = pd.DataFrame(
        results
    )

    if (
        "QLIKE_RV"
        in results_df.columns
    ):
        successful = (
            results_df[
                results_df[
                    "QLIKE_RV"
                ].notna()
            ]
            .copy()
        )
    else:
        successful = pd.DataFrame()

    if not successful.empty:
        successful = (
            successful
            .sort_values(
                "ticker"
            )
            .reset_index(
                drop=True
            )
        )

        successful.to_csv(
            RESULTS_DIR
            / "HAR_QML_RV_evaluation_results_sorted.csv",
            index=False,
        )

        print(
            "\n"
            + "=" * 130
        )
        print(
            "FINAL HAR-QML RAW-RV RESULTS"
        )
        print(
            "=" * 130
        )

        columns_to_print = [
            "ticker",
            "n_total",
            "rolling_window_size",
            "n_oos",
            "mean_predicted_variance",
            "mean_oos_raw_RV",
            "n_nonpositive_forecasts",
            "QLIKE_RV",
            "MSE_var_RV",
            "MAE_var_RV",
        ]

        print(
            successful[
                columns_to_print
            ].to_string(
                index=False
            )
        )

    else:
        print(
            "\nNo successful HAR-QML results were generated."
        )


if __name__ == "__main__":
    main()







# from __future__ import annotations

# import json
# import sys
# from pathlib import Path
# from typing import Dict, Tuple

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd


# # =========================================================
# # Project paths
# # =========================================================

# HERE = Path(__file__).resolve().parent
# PROJECT_ROOT = HERE.parent

# MODELS_DIR = (
#     PROJECT_ROOT
#     / "models"
# )

# DATA_DIR = (
#     PROJECT_ROOT
#     / "data"
# )

# DATA_PATH = (
#     DATA_DIR
#     / "rv.pkl"
# )

# RESULTS_DIR = (
#     PROJECT_ROOT
#     / "results"
#     / "HAR_QML_point_horizon_rolling_50_50"
# )

# PREDICTIONS_DIR = (
#     RESULTS_DIR
#     / "predictions"
# )

# PARAMETERS_DIR = (
#     RESULTS_DIR
#     / "parameters"
# )

# PLOTS_DIR = (
#     RESULTS_DIR
#     / "plots"
# )

# for directory in (
#     RESULTS_DIR,
#     PREDICTIONS_DIR,
#     PARAMETERS_DIR,
#     PLOTS_DIR,
# ):
#     directory.mkdir(
#         parents=True,
#         exist_ok=True,
#     )


# # =========================================================
# # Import paths
# # =========================================================

# sys.path.insert(
#     0,
#     str(
#         PROJECT_ROOT
#     ),
# )

# sys.path.insert(
#     0,
#     str(
#         MODELS_DIR
#     ),
# )


# # =========================================================
# # Model
# # =========================================================

# from models.model_HARRV import (
#     fixed_half_sample_rolling_harrv_qml_multihorizon,
#     mae_var_np,
#     mae_vol_np,
#     mse_var_np,
#     mse_vol_np,
#     qlike_var_np,
#     return_nll_np,
# )


# # =========================================================
# # Existing project data utilities
# # =========================================================

# from SDE_Garch_utils.utils1 import (
#     build_data_dict,
#     load_rv,
# )


# # =========================================================
# # Experiment configuration
# # =========================================================

# TICKERS = [
#     "SPX",
#     "SSMI",
#     "STI",
#     "OMXSPI",
#     "KS11",
#     "OMXHPI",
#     "SSEC",
#     "KSE",
#     "MXX",
#     "IXIC",
#     "OMXC20",
#     "STOXX50E",
#     "SMSI",
#     "N225",
#     "OSEAX",
#     "RUT",
#     "NSEI",
# ]

# START_TICKER_INDEX = 0

# FORECAST_HORIZONS = (
#     1,
#     5,
#     10,
#     20,
# )

# REFIT_EVERY = 21

# WEEKLY_WINDOW = 5
# MONTHLY_WINDOW = 22

# EPS = 1e-8


# # =========================================================
# # Data helpers
# # =========================================================

# def _finite_pair(
#     y: np.ndarray,
#     rv: np.ndarray,
# ) -> Tuple[
#     np.ndarray,
#     np.ndarray,
# ]:
#     y = np.asarray(
#         y,
#         dtype=np.float64,
#     ).reshape(-1)

#     rv = np.asarray(
#         rv,
#         dtype=np.float64,
#     ).reshape(-1)

#     if len(y) != len(rv):
#         raise ValueError(
#             "log-price and RV arrays must have equal length."
#         )

#     mask = (
#         np.isfinite(y)
#         & np.isfinite(rv)
#     )

#     y = y[
#         mask
#     ]

#     rv = rv[
#         mask
#     ]

#     if len(y) == 0:
#         raise ValueError(
#             "No finite observations remain."
#         )

#     if np.any(
#         rv <= 0.0
#     ):
#         raise ValueError(
#             "HAR-QML requires strictly positive realized variance."
#         )

#     return (
#         y,
#         rv,
#     )


# def extract_full_chronology(
#     data_dict: Dict,
#     ticker: str,
# ) -> Tuple[
#     np.ndarray,
#     np.ndarray,
# ]:
#     """
#     Reconstruct full daily log-price/RV chronology.

#     This preserves the SAME original daily index as the SDE, GARCH,
#     GARCH-Ito and Realized-GARCH point-horizon experiments.
#     """
#     if ticker not in data_dict:
#         raise KeyError(
#             f"{ticker} not found in data_dict."
#         )

#     block = (
#         data_dict[
#             ticker
#         ]
#     )

#     if (
#         isinstance(
#             block,
#             dict,
#         )
#         and "train" in block
#         and isinstance(
#             block[
#                 "train"
#             ],
#             dict,
#         )
#     ):
#         block = (
#             block[
#                 "train"
#             ]
#         )

#     required = (
#         "y_train",
#         "rv_train",
#         "y_test",
#         "rv_test",
#     )

#     missing = [
#         key
#         for key
#         in required
#         if key
#         not in block
#     ]

#     if missing:
#         raise KeyError(
#             f"{ticker}: missing data keys {missing}. "
#             f"Available keys: {list(block.keys())}"
#         )

#     y_train, rv_train = (
#         _finite_pair(
#             block[
#                 "y_train"
#             ],
#             block[
#                 "rv_train"
#             ],
#         )
#     )

#     y_test, rv_test = (
#         _finite_pair(
#             block[
#                 "y_test"
#             ],
#             block[
#                 "rv_test"
#             ],
#         )
#     )

#     full_y = np.concatenate(
#         [
#             y_train,
#             y_test,
#         ]
#     )

#     full_rv = np.concatenate(
#         [
#             rv_train,
#             rv_test,
#         ]
#     )

#     return (
#         full_y,
#         full_rv,
#     )


# def _load_project_data() -> Dict:
#     raw_data = load_rv(
#         DATA_PATH
#     )

#     if isinstance(
#         raw_data,
#         dict,
#     ):
#         sample = next(
#             (
#                 ticker
#                 for ticker
#                 in TICKERS
#                 if ticker
#                 in raw_data
#             ),
#             None,
#         )

#         if sample is not None:
#             sample_node = (
#                 raw_data[
#                     sample
#                 ]
#             )

#             if (
#                 isinstance(
#                     sample_node,
#                     dict,
#                 )
#                 and (
#                     "train"
#                     in sample_node
#                     or "y_train"
#                     in sample_node
#                 )
#             ):
#                 return raw_data

#     try:
#         return build_data_dict(
#             raw_data
#         )

#     except TypeError:
#         try:
#             return build_data_dict(
#                 DATA_PATH
#             )

#         except TypeError:
#             return build_data_dict()


# # =========================================================
# # Save helpers
# # =========================================================

# def save_horizon_arrays(
#     *,
#     ticker: str,
#     horizon: int,
#     horizon_result: Dict,
# ) -> None:
#     prefix = (
#         PREDICTIONS_DIR
#         / (
#             f"HAR_QML_point_"
#             f"{ticker}_h{horizon}"
#         )
#     )

#     np.save(
#         str(prefix)
#         + "_pred_var.npy",
#         horizon_result[
#             "predicted_h_safe"
#         ],
#     )

#     np.save(
#         str(prefix)
#         + "_pred_var_raw.npy",
#         horizon_result[
#             "predicted_h_raw"
#         ],
#     )

#     np.save(
#         str(prefix)
#         + "_rv_target.npy",
#         horizon_result[
#             "actual_rv"
#         ],
#     )

#     np.save(
#         str(prefix)
#         + "_return_target.npy",
#         horizon_result[
#             "actual_return"
#         ],
#     )

#     np.save(
#         str(prefix)
#         + "_origin_index.npy",
#         horizon_result[
#             "origin_index"
#         ],
#     )

#     np.save(
#         str(prefix)
#         + "_target_end_index.npy",
#         horizon_result[
#             "target_end_index"
#         ],
#     )

#     np.savez(
#         str(prefix)
#         + "_forecasts.npz",
#         predicted_h_safe=(
#             horizon_result[
#                 "predicted_h_safe"
#             ]
#         ),
#         predicted_h_raw=(
#             horizon_result[
#                 "predicted_h_raw"
#             ]
#         ),
#         actual_rv=(
#             horizon_result[
#                 "actual_rv"
#             ]
#         ),
#         actual_return=(
#             horizon_result[
#                 "actual_return"
#             ]
#         ),
#         origin_index=(
#             horizon_result[
#                 "origin_index"
#             ]
#         ),
#         target_end_index=(
#             horizon_result[
#                 "target_end_index"
#             ]
#         ),
#         rv_daily_forecast_regressor=(
#             horizon_result[
#                 "rv_daily_forecast_regressor"
#             ]
#         ),
#         rv_weekly_forecast_regressor=(
#             horizon_result[
#                 "rv_weekly_forecast_regressor"
#             ]
#         ),
#         rv_monthly_forecast_regressor=(
#             horizon_result[
#                 "rv_monthly_forecast_regressor"
#             ]
#         ),
#         forecast_horizon=np.asarray(
#             horizon,
#             dtype=np.int64,
#         ),
#     )


# # =========================================================
# # Fit plot
# # =========================================================

# def save_point_horizon_fit_plot(
#     *,
#     ticker: str,
#     horizon: int,
#     predicted: np.ndarray,
#     actual_rv: np.ndarray,
#     target_end_index: np.ndarray,
# ) -> None:
#     predicted = np.asarray(
#         predicted,
#         dtype=np.float64,
#     ).reshape(-1)

#     actual_rv = np.asarray(
#         actual_rv,
#         dtype=np.float64,
#     ).reshape(-1)

#     target_end_index = np.asarray(
#         target_end_index,
#         dtype=np.int64,
#     ).reshape(-1)

#     if not (
#         len(predicted)
#         == len(actual_rv)
#         == len(target_end_index)
#     ):
#         raise ValueError(
#             f"{ticker} h={horizon}: "
#             "plot arrays are not aligned."
#         )

#     fig, ax = plt.subplots(
#         figsize=(
#             12,
#             5,
#         )
#     )

#     ax.plot(
#         target_end_index,
#         actual_rv,
#         label="Actual RV",
#         linewidth=1.2,
#         alpha=0.8,
#     )

#     ax.plot(
#         target_end_index,
#         predicted,
#         label="HAR-QML point forecast",
#         linewidth=1.4,
#         alpha=0.9,
#     )

#     ax.set_title(
#         f"{ticker}: HAR-QML "
#         f"point-horizon fit (h={horizon})"
#     )

#     ax.set_xlabel(
#         "Target observation index"
#     )

#     ax.set_ylabel(
#         "Daily variance"
#     )

#     ax.legend()

#     ax.grid(
#         True,
#         alpha=0.25,
#     )

#     fig.tight_layout()

#     fig.savefig(
#         PLOTS_DIR
#         / (
#             f"HAR_QML_"
#             f"{ticker}_"
#             f"h{horizon}_"
#             "point_fit.png"
#         ),
#         dpi=300,
#         bbox_inches="tight",
#     )

#     plt.close(
#         fig
#     )


# # =========================================================
# # Main
# # =========================================================

# def main() -> None:
#     print(
#         "=" * 100
#     )

#     print(
#         "Iterated POINT-horizon HAR-QML"
#     )

#     print(
#         "=" * 100
#     )

#     print(
#         f"Data:     "
#         f"{DATA_PATH}"
#     )

#     print(
#         f"Output:   "
#         f"{RESULTS_DIR}"
#     )

#     print(
#         f"Horizons: "
#         f"{FORECAST_HORIZONS}"
#     )

#     print(
#         f"Refit:    "
#         f"every {REFIT_EVERY} origins"
#     )

#     print(
#         "Design:   "
#         "same daily HAR-QML propagated recursively"
#     )

#     print(
#         "Target:   "
#         "RV_{n+h}, not cumulative RV"
#     )

#     data_dict = (
#         _load_project_data()
#     )

#     config = {
#         "forecast_horizons":
#             list(
#                 FORECAST_HORIZONS
#             ),

#         "refit_every":
#             REFIT_EVERY,

#         "weekly_window":
#             WEEKLY_WINDOW,

#         "monthly_window":
#             MONTHLY_WINDOW,

#         "eps":
#             EPS,

#         "model":
#             "HAR-QML",

#         "estimation":
#             (
#                 "unconstrained_level_HAR_"
#                 "Gaussian_return_QML"
#             ),

#         "forecast_design":
#             (
#                 "iterated_daily_"
#                 "point_horizon"
#             ),

#         "future_rv_recursion":
#             (
#                 "append_max(raw_HAR_forecast,EPS)_"
#                 "as_pseudo_future_RV"
#             ),

#         "evaluation_target":
#             "point_raw_RV",

#         "oos_origin_design":
#             "common_across_horizons",
#     }

#     with open(
#         RESULTS_DIR
#         / "config.json",
#         "w",
#         encoding="utf-8",
#     ) as handle:
#         json.dump(
#             config,
#             handle,
#             indent=2,
#         )

#     summary_rows = []

#     all_parameter_rows = []

#     selected_tickers = (
#         TICKERS[
#             START_TICKER_INDEX:
#         ]
#     )

#     for ticker in selected_tickers:
#         print(
#             "\n"
#             + "=" * 100
#         )

#         print(
#             f"Running {ticker}"
#         )

#         print(
#             "=" * 100
#         )

#         try:
#             (
#                 log_price_full,
#                 rv_full,
#             ) = (
#                 extract_full_chronology(
#                     data_dict=(
#                         data_dict
#                     ),
#                     ticker=(
#                         ticker
#                     ),
#                 )
#             )

#             n_total = len(
#                 rv_full
#             )

#             window_size = (
#                 n_total
#                 // 2
#             )

#             expected_common_oos = (
#                 n_total
#                 - window_size
#                 - (
#                     max(
#                         FORECAST_HORIZONS
#                     )
#                     - 1
#                 )
#             )

#             print(
#                 "\n=== Fixed 50:50 design ==="
#             )

#             print(
#                 f"Total observations   : "
#                 f"{n_total}"
#             )

#             print(
#                 f"Rolling-window size  : "
#                 f"{window_size}"
#             )

#             print(
#                 f"Common OOS origins   : "
#                 f"{expected_common_oos}"
#             )

#             print(
#                 f"Mean raw RV          : "
#                 f"{np.mean(rv_full):.8f}"
#             )

#             output = (
#                 fixed_half_sample_rolling_harrv_qml_multihorizon(
#                     ticker=ticker,
#                     rv_full=(
#                         rv_full
#                     ),
#                     log_price_full=(
#                         log_price_full
#                     ),
#                     forecast_horizons=(
#                         FORECAST_HORIZONS
#                     ),
#                     window_size=(
#                         window_size
#                     ),
#                     refit_every=(
#                         REFIT_EVERY
#                     ),
#                     weekly_window=(
#                         WEEKLY_WINDOW
#                     ),
#                     monthly_window=(
#                         MONTHLY_WINDOW
#                     ),
#                     eps=(
#                         EPS
#                     ),
#                 )
#             )

#             if (
#                 output[
#                     "n_oos_common"
#                 ]
#                 != expected_common_oos
#             ):
#                 raise RuntimeError(
#                     f"{ticker}: expected "
#                     f"{expected_common_oos} "
#                     "common OOS origins but got "
#                     f"{output['n_oos_common']}."
#                 )

#             # ---------------------------------------------
#             # Parameter history
#             # ---------------------------------------------

#             parameter_df = pd.DataFrame(
#                 output[
#                     "parameter_history"
#                 ]
#             )

#             if parameter_df.empty:
#                 raise RuntimeError(
#                     "HAR-QML parameter history is empty."
#                 )

#             parameter_path = (
#                 PARAMETERS_DIR
#                 / (
#                     f"HAR_QML_point_"
#                     f"{ticker}_"
#                     "parameter_history.csv"
#                 )
#             )

#             parameter_df.to_csv(
#                 parameter_path,
#                 index=False,
#             )

#             parameter_with_ticker = (
#                 parameter_df.copy()
#             )

#             parameter_with_ticker.insert(
#                 0,
#                 "ticker",
#                 ticker,
#             )

#             all_parameter_rows.append(
#                 parameter_with_ticker
#             )

#             parameter_df.loc[
#                 parameter_df[
#                     "refit"
#                 ].astype(
#                     bool
#                 )
#             ].to_csv(
#                 PARAMETERS_DIR
#                 / (
#                     f"HAR_QML_point_"
#                     f"{ticker}_"
#                     "parameter_refits.csv"
#                 ),
#                 index=False,
#             )

#             final_param = (
#                 parameter_df.iloc[
#                     -1
#                 ]
#             )

#             # ---------------------------------------------
#             # Horizon-specific evaluation
#             # ---------------------------------------------

#             for h in FORECAST_HORIZONS:
#                 r = (
#                     output[
#                         "results"
#                     ][h]
#                 )

#                 predicted = np.asarray(
#                     r[
#                         "predicted_h_safe"
#                     ],
#                     dtype=np.float64,
#                 ).reshape(-1)

#                 predicted_raw = np.asarray(
#                     r[
#                         "predicted_h_raw"
#                     ],
#                     dtype=np.float64,
#                 ).reshape(-1)

#                 target_rv = np.asarray(
#                     r[
#                         "actual_rv"
#                     ],
#                     dtype=np.float64,
#                 ).reshape(-1)

#                 target_return = np.asarray(
#                     r[
#                         "actual_return"
#                     ],
#                     dtype=np.float64,
#                 ).reshape(-1)

#                 origin_index = np.asarray(
#                     r[
#                         "origin_index"
#                     ],
#                     dtype=np.int64,
#                 ).reshape(-1)

#                 target_end_index = np.asarray(
#                     r[
#                         "target_end_index"
#                     ],
#                     dtype=np.int64,
#                 ).reshape(-1)

#                 lengths = {
#                     len(
#                         predicted
#                     ),
#                     len(
#                         predicted_raw
#                     ),
#                     len(
#                         target_rv
#                     ),
#                     len(
#                         target_return
#                     ),
#                     len(
#                         origin_index
#                     ),
#                     len(
#                         target_end_index
#                     ),
#                 }

#                 if len(
#                     lengths
#                 ) != 1:
#                     raise RuntimeError(
#                         f"{ticker} h={h}: "
#                         "horizon arrays are not aligned."
#                     )

#                 if (
#                     len(
#                         predicted
#                     )
#                     != expected_common_oos
#                 ):
#                     raise RuntimeError(
#                         f"{ticker} h={h}: "
#                         f"expected {expected_common_oos} "
#                         "point forecasts but got "
#                         f"{len(predicted)}."
#                     )

#                 if np.any(
#                     target_end_index
#                     - origin_index
#                     != h
#                 ):
#                     raise RuntimeError(
#                         f"{ticker} h={h}: "
#                         "incorrect target indexing."
#                     )

#                 save_horizon_arrays(
#                     ticker=(
#                         ticker
#                     ),
#                     horizon=(
#                         h
#                     ),
#                     horizon_result=(
#                         r
#                     ),
#                 )

#                 save_point_horizon_fit_plot(
#                     ticker=(
#                         ticker
#                     ),
#                     horizon=(
#                         h
#                     ),
#                     predicted=(
#                         predicted
#                     ),
#                     actual_rv=(
#                         target_rv
#                     ),
#                     target_end_index=(
#                         target_end_index
#                     ),
#                 )

#                 row = {
#                     "ticker":
#                         ticker,

#                     "model":
#                         "HAR-QML",

#                     "forecast_design":
#                         (
#                             "iterated_daily_"
#                             "point_horizon"
#                         ),

#                     "forecast_horizon":
#                         int(
#                             h
#                         ),

#                     "evaluation_target":
#                         "point_raw_RV",

#                     "n_total":
#                         int(
#                             output[
#                                 "n_total"
#                             ]
#                         ),

#                     "rolling_window_size":
#                         int(
#                             output[
#                                 "rolling_window_size"
#                             ]
#                         ),

#                     "first_oos_target":
#                         int(
#                             output[
#                                 "first_oos_target"
#                             ]
#                         ),

#                     "first_origin":
#                         int(
#                             output[
#                                 "first_origin"
#                             ]
#                         ),

#                     "n_oos":
#                         int(
#                             r[
#                                 "n_oos"
#                             ]
#                         ),

#                     "refit_every":
#                         int(
#                             output[
#                                 "refit_every"
#                             ]
#                         ),

#                     "window_step":
#                         int(
#                             output[
#                                 "window_step"
#                             ]
#                         ),

#                     "weekly_window":
#                         int(
#                             output[
#                                 "weekly_window"
#                             ]
#                         ),

#                     "monthly_window":
#                         int(
#                             output[
#                                 "monthly_window"
#                             ]
#                         ),

#                     "n_refits":
#                         int(
#                             output[
#                                 "n_refits"
#                             ]
#                         ),

#                     "n_refits_accepted":
#                         int(
#                             output[
#                                 "n_refits_accepted"
#                             ]
#                         ),

#                     "n_nonpositive_forecasts":
#                         int(
#                             r[
#                                 "n_nonpositive_forecasts"
#                             ]
#                         ),

#                     "mean_predicted_variance":
#                         float(
#                             np.mean(
#                                 predicted
#                             )
#                         ),

#                     "mean_predicted_variance_raw":
#                         float(
#                             np.mean(
#                                 predicted_raw
#                             )
#                         ),

#                     "mean_actual_RV":
#                         float(
#                             np.mean(
#                                 target_rv
#                             )
#                         ),

#                     "mean_target_return_squared":
#                         float(
#                             np.mean(
#                                 target_return
#                                 ** 2
#                             )
#                         ),

#                     "min_predicted_variance":
#                         float(
#                             np.min(
#                                 predicted
#                             )
#                         ),

#                     "max_predicted_variance":
#                         float(
#                             np.max(
#                                 predicted
#                             )
#                         ),

#                     "QLIKE":
#                         qlike_var_np(
#                             predicted,
#                             target_rv,
#                             eps=EPS,
#                         ),

#                     "MSE_vol":
#                         mse_vol_np(
#                             predicted,
#                             target_rv,
#                             eps=EPS,
#                         ),

#                     "MSE_var":
#                         mse_var_np(
#                             predicted,
#                             target_rv,
#                         ),

#                     "MAE_vol":
#                         mae_vol_np(
#                             predicted,
#                             target_rv,
#                             eps=EPS,
#                         ),

#                     "MAE_var":
#                         mae_var_np(
#                             predicted,
#                             target_rv,
#                         ),

#                     "Return_NLL":
#                         return_nll_np(
#                             predicted,
#                             target_return,
#                             eps=EPS,
#                         ),

#                     "final_beta_0":
#                         float(
#                             final_param[
#                                 "beta_0"
#                             ]
#                         ),

#                     "final_beta_daily":
#                         float(
#                             final_param[
#                                 "beta_daily"
#                             ]
#                         ),

#                     "final_beta_weekly":
#                         float(
#                             final_param[
#                                 "beta_weekly"
#                             ]
#                         ),

#                     "final_beta_monthly":
#                         float(
#                             final_param[
#                                 "beta_monthly"
#                             ]
#                         ),

#                     "mean_beta_0":
#                         float(
#                             parameter_df[
#                                 "beta_0"
#                             ].mean()
#                         ),

#                     "mean_beta_daily":
#                         float(
#                             parameter_df[
#                                 "beta_daily"
#                             ].mean()
#                         ),

#                     "mean_beta_weekly":
#                         float(
#                             parameter_df[
#                                 "beta_weekly"
#                             ].mean()
#                         ),

#                     "mean_beta_monthly":
#                         float(
#                             parameter_df[
#                                 "beta_monthly"
#                             ].mean()
#                         ),

#                     "loss_type":
#                         (
#                             "Gaussian return QML"
#                         ),

#                     "estimation_realized_measure":
#                         (
#                             "raw_RV_as_HAR_regressor"
#                         ),
#                 }

#                 summary_rows.append(
#                     row
#                 )

#                 print(
#                     f"\n{ticker} | h={h}"
#                 )

#                 print(
#                     f"    N OOS      : "
#                     f"{row['n_oos']}"
#                 )

#                 print(
#                     f"    Mean pred  : "
#                     f"{row['mean_predicted_variance']:.8f}"
#                 )

#                 print(
#                     f"    Mean RV    : "
#                     f"{row['mean_actual_RV']:.8f}"
#                 )

#                 print(
#                     f"    QLIKE      : "
#                     f"{row['QLIKE']:.8f}"
#                 )

#                 print(
#                     f"    MSE_var    : "
#                     f"{row['MSE_var']:.8f}"
#                 )

#                 print(
#                     f"    MAE_var    : "
#                     f"{row['MAE_var']:.8f}"
#                 )

#                 print(
#                     f"    Return NLL : "
#                     f"{row['Return_NLL']:.8f}"
#                 )

#                 print(
#                     "    Nonpositive raw forecasts: "
#                     f"{row['n_nonpositive_forecasts']}"
#                 )

#         except Exception as exc:
#             print(
#                 f"{ticker:>8s}  "
#                 f"{type(exc).__name__}: "
#                 f"{exc}"
#             )
#             raise

#     # =====================================================
#     # Save combined outputs
#     # =====================================================

#     summary_df = pd.DataFrame(
#         summary_rows
#     )

#     summary_path = (
#         RESULTS_DIR
#         / "HAR_QML_point_horizon_summary.csv"
#     )

#     summary_df.to_csv(
#         summary_path,
#         index=False,
#     )

#     if all_parameter_rows:
#         combined_parameters = (
#             pd.concat(
#                 all_parameter_rows,
#                 ignore_index=True,
#             )
#         )

#         combined_parameters.to_csv(
#             RESULTS_DIR
#             / (
#                 "HAR_QML_point_horizon_"
#                 "all_parameter_history.csv"
#             ),
#             index=False,
#         )

#     print(
#         "\n"
#         + "=" * 100
#     )

#     print(
#         "HAR-QML point-horizon run complete"
#     )

#     print(
#         "=" * 100
#     )

#     print(
#         f"Summary saved to: "
#         f"{summary_path}"
#     )

#     print(
#         f"Predictions:      "
#         f"{PREDICTIONS_DIR}"
#     )

#     print(
#         f"Parameters:       "
#         f"{PARAMETERS_DIR}"
#     )

#     print(
#         f"Plots:            "
#         f"{PLOTS_DIR}"
#     )


# if __name__ == "__main__":
#     main()
