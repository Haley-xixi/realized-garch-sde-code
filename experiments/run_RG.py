import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =========================================================
# Path setup
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

DATA_PATH = BASE_DIR / "data" / "rv.pkl"

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "realized_garch_rolling_50_50"
)
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# =========================================================
# Local imports
# =========================================================
from SDE_Garch_utils.utils1 import (
    build_data_dict,
    load_rv,
    set_seed_all,
)

from models.model_Realized_GARCH import (
    fixed_half_sample_rolling_realized_garch,
    get_rg_arrays_from_data_dict,
    mae_variance,
    mae_volatility,
    mse_variance,
    mse_volatility,
    qlike_variance,
    return_nll_np,
)

# =========================================================
# Configuration
# =========================================================
SEED = 1313
EPS = 1e-8

REFIT_EVERY = 21
WARM_START = True

# =========================================================
# Plot settings
# =========================================================
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10


# =========================================================
# Utilities
# =========================================================
def ensure_1d_float(
    values,
    *,
    positive: bool = False,
    eps: float = EPS,
) -> np.ndarray:
    """Convert an input array to finite one-dimensional float64."""
    array = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    if not np.all(np.isfinite(array)):
        raise ValueError(
            "Array contains non-finite values."
        )

    if positive:
        array = np.maximum(
            array,
            eps,
        )

    return array


def save_progress(
    results: list[dict],
) -> None:
    """Save partial results after each ticker."""
    pd.DataFrame(results).to_csv(
        OUTPUT_DIR
        / "realized_garch_rolling_results.csv",
        index=False,
    )


# =========================================================
# Plot
# =========================================================
def plot_pred_vs_actual(
    ticker: str,
    prediction: np.ndarray,
    actual_rv: np.ndarray,
    prediction_name: str,
    filename_suffix: str,
) -> None:
    prediction = ensure_1d_float(
        prediction,
        positive=True,
    )

    actual_rv = ensure_1d_float(
        actual_rv,
        positive=True,
    )

    if len(prediction) != len(actual_rv):
        raise ValueError(
            "Prediction and actual RV arrays must have "
            "equal length."
        )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        actual_rv,
        label="Observed realised variance",
        linewidth=1.5,
    )

    ax.plot(
        prediction,
        label=prediction_name,
        linewidth=1.2,
    )

    ax.set_title(
        f"{ticker}: Realized GARCH rolling forecast"
    )

    ax.set_xlabel(
        "Out-of-sample observation"
    )

    ax.set_ylabel(
        "Variance"
    )

    ax.legend()
    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / (
            f"RealizedGARCH_{ticker}_"
            f"{filename_suffix}.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_scatter(
    ticker: str,
    prediction: np.ndarray,
    actual_rv: np.ndarray,
    prediction_name: str,
    filename_suffix: str,
    use_log_scale: bool = True,
) -> None:
    prediction = ensure_1d_float(
        prediction,
        positive=True,
    )

    actual_rv = ensure_1d_float(
        actual_rv,
        positive=True,
    )

    if len(prediction) != len(actual_rv):
        raise ValueError(
            "Prediction and actual RV arrays must have "
            "equal length."
        )

    lower = min(
        float(np.min(prediction)),
        float(np.min(actual_rv)),
    )

    upper = max(
        float(np.max(prediction)),
        float(np.max(actual_rv)),
    )

    fig, ax = plt.subplots(
        figsize=(6.5, 6.0)
    )

    ax.scatter(
        actual_rv,
        prediction,
        s=18,
        alpha=0.5,
        edgecolors="none",
    )

    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=1.2,
        label="45-degree line",
    )

    if use_log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel(
        "Observed realised variance"
    )

    ax.set_ylabel(
        prediction_name
    )

    ax.set_title(
        f"{ticker}: predicted versus observed variance"
    )

    ax.legend()
    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / (
            f"RealizedGARCH_{ticker}_"
            f"{filename_suffix}_scatter.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =========================================================
# Main experiment
# =========================================================
def main() -> None:
    set_seed_all(SEED)

    print(f"Data path: {DATA_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(
        "Rolling refit frequency: "
        f"every {REFIT_EVERY} forecasts"
    )

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------
    df = load_rv(
        filepath=str(DATA_PATH),
    )

    data_dict = build_data_dict(df)

    all_tickers = list(
        data_dict.keys()
    )

    selected_tickers = all_tickers[14:]


    results: list[dict] = []

    # -----------------------------------------------------
    # Ticker loop
    # -----------------------------------------------------
    for ticker in selected_tickers:
        print(
            "\n" + "=" * 90
        )
        print(
            f"Processing rolling Realized GARCH: {ticker}"
        )
        print(
            "=" * 90
        )

        try:
           
            (
                y_train_old,
                rv_train_old,
                y_test_old,
                rv_test_old,
            ) = get_rg_arrays_from_data_dict(
                data_dict=data_dict,
                ticker=ticker,
            )

            y_train_old = ensure_1d_float(
                y_train_old,
                positive=False,
            )

            y_test_old = ensure_1d_float(
                y_test_old,
                positive=False,
            )

            rv_train_old = ensure_1d_float(
                rv_train_old,
                positive=True,
            )

            rv_test_old = ensure_1d_float(
                rv_test_old,
                positive=True,
            )

            if (
                len(y_train_old)
                != len(rv_train_old)
            ):
                raise ValueError(
                    "Original training return and RV "
                    "lengths differ."
                )

            if (
                len(y_test_old)
                != len(rv_test_old)
            ):
                raise ValueError(
                    "Original test return and RV "
                    "lengths differ."
                )

           
            y_full = np.concatenate(
                [
                    y_train_old,
                    y_test_old,
                ]
            )

            rv_full = np.concatenate(
                [
                    rv_train_old,
                    rv_test_old,
                ]
            )

            y_full = ensure_1d_float(
                y_full,
                positive=False,
            )

            rv_full = ensure_1d_float(
                rv_full,
                positive=True,
            )

            if len(y_full) != len(rv_full):
                raise ValueError(
                    "Full return and RV lengths differ: "
                    f"{len(y_full)} versus "
                    f"{len(rv_full)}."
                )

            n_total = len(y_full)
            rolling_window_size = (
                n_total // 2
            )

            n_oos_expected = (
                n_total
                - rolling_window_size
            )

            print(
                "\n=== Fixed 50:50 design ==="
            )
            print(
                f"Total observations   : {n_total}"
            )
            print(
                "Rolling-window size : "
                f"{rolling_window_size}"
            )
            print(
                "Expected OOS size   : "
                f"{n_oos_expected}"
            )
            print(
                "Mean return squared : "
                f"{np.mean(y_full ** 2):.8f}"
            )
            print(
                "Mean raw RV         : "
                f"{np.mean(rv_full):.8f}"
            )

            # -------------------------------------------------
            # Rolling Realized GARCH estimation
            # -------------------------------------------------
            rolling_result = (
                fixed_half_sample_rolling_realized_garch(
                    ticker=ticker,
                    y_full=y_full,
                    rv_full=rv_full,
                    refit_every=REFIT_EVERY,
                    warm_start=WARM_START,
                )
            )

            predicted_variance = (
                ensure_1d_float(
                    rolling_result[
                        "predicted_variance"
                    ],
                    positive=True,
                )
            )

            predicted_rv = (
                ensure_1d_float(
                    rolling_result[
                        "predicted_rv"
                    ],
                    positive=True,
                )
            )

            actual_rv = (
                ensure_1d_float(
                    rolling_result[
                        "actual_rv"
                    ],
                    positive=True,
                )
            )

            actual_returns = (
                ensure_1d_float(
                    rolling_result[
                        "actual_returns"
                    ],
                    positive=False,
                )
            )

            parameter_history = (
                rolling_result[
                    "parameter_history"
                ]
            )

            lengths = {
                len(predicted_variance),
                len(predicted_rv),
                len(actual_rv),
                len(actual_returns),
            }

            if len(lengths) != 1:
                raise ValueError(
                    "Rolling forecast arrays are not aligned: "
                    f"{sorted(lengths)}"
                )

            n_oos = len(actual_rv)

            if n_oos != n_oos_expected:
                raise ValueError(
                    f"Expected {n_oos_expected} forecasts "
                    f"but received {n_oos}."
                )

            # -------------------------------------------------
            # Metrics for latent variance forecast h
            # -------------------------------------------------
            qlike_raw = qlike_variance(
                predicted_variance,
                actual_rv,
            )

            mse_vol_raw = mse_volatility(
                predicted_variance,
                actual_rv,
            )

            mse_var_raw = mse_variance(
                predicted_variance,
                actual_rv,
            )

            mae_vol_raw = mae_volatility(
                predicted_variance,
                actual_rv,
            )

            mae_var_raw = mae_variance(
                predicted_variance,
                actual_rv,
            )

            return_nll_value = return_nll_np(
                predicted_variance,
                actual_returns,
            )

            # -------------------------------------------------
            # Metrics for measurement forecast xi + phi h
            # -------------------------------------------------
            qlike_rv = qlike_variance(
                predicted_rv,
                actual_rv,
            )

            mse_vol_rv = mse_volatility(
                predicted_rv,
                actual_rv,
            )

            mse_var_rv = mse_variance(
                predicted_rv,
                actual_rv,
            )

            mae_vol_rv = mae_volatility(
                predicted_rv,
                actual_rv,
            )

            mae_var_rv = mae_variance(
                predicted_rv,
                actual_rv,
            )

            # -------------------------------------------------
            # Diagnostics
            # -------------------------------------------------
            print(
                "\n=== Rolling Realized GARCH metrics ==="
            )

            print(
                f"QLIKE raw    : {qlike_raw:.8f}"
            )
            print(
                f"MSE vol raw  : {mse_vol_raw:.8f}"
            )
            print(
                f"MSE var raw  : {mse_var_raw:.8f}"
            )
            print(
                f"MAE vol raw  : {mae_vol_raw:.8f}"
            )
            print(
                f"MAE var raw  : {mae_var_raw:.8f}"
            )
            print(
                f"Return NLL   : {return_nll_value:.8f}"
            )

            print(
                f"QLIKE RV     : {qlike_rv:.8f}"
            )
            print(
                f"MSE vol RV   : {mse_vol_rv:.8f}"
            )
            print(
                f"MSE var RV   : {mse_var_rv:.8f}"
            )
            print(
                f"MAE vol RV   : {mae_vol_rv:.8f}"
            )
            print(
                f"MAE var RV   : {mae_var_rv:.8f}"
            )

            np.save(
                OUTPUT_DIR
                / f"{ticker}_rg_variance_pred.npy",
                predicted_variance,
            )

            np.save(
                OUTPUT_DIR
                / f"{ticker}_rg_rv_pred.npy",
                predicted_rv,
            )

            np.save(
                OUTPUT_DIR
                / f"{ticker}_rg_rv_target.npy",
                actual_rv,
            )

            np.save(
                OUTPUT_DIR
                / f"{ticker}_rg_returns_target.npy",
                actual_returns,
            )

            np.savez(
                OUTPUT_DIR
                / f"{ticker}_rg_rolling_forecasts.npz",
                predicted_variance=predicted_variance,
                predicted_rv=predicted_rv,
                actual_rv=actual_rv,
                actual_returns=actual_returns,
            )

            # -------------------------------------------------
            # Save rolling parameter history
            # -------------------------------------------------
            parameter_df = pd.DataFrame(
                parameter_history
            )

            parameter_df.to_csv(
                OUTPUT_DIR
                / (
                    f"{ticker}_rg_rolling_"
                    "parameter_history.csv"
                ),
                index=False,
            )

            if (
                "refit"
                in parameter_df.columns
            ):
                parameter_df.loc[
                    parameter_df[
                        "refit"
                    ].astype(bool)
                ].to_csv(
                    OUTPUT_DIR
                    / (
                        f"{ticker}_rg_rolling_"
                        "parameter_refits.csv"
                    ),
                    index=False,
                )

            # -------------------------------------------------
            # Save plots
            # -------------------------------------------------
            plot_pred_vs_actual(
                ticker=ticker,
                prediction=predicted_variance,
                actual_rv=actual_rv,
                prediction_name=(
                    "Predicted latent variance"
                ),
                filename_suffix=(
                    "latent_variance"
                ),
            )

            plot_pred_vs_actual(
                ticker=ticker,
                prediction=predicted_rv,
                actual_rv=actual_rv,
                prediction_name=(
                    "Predicted realised measure"
                ),
                filename_suffix=(
                    "realised_measure"
                ),
            )

            plot_scatter(
                ticker=ticker,
                prediction=predicted_variance,
                actual_rv=actual_rv,
                prediction_name=(
                    "Predicted latent variance"
                ),
                filename_suffix=(
                    "latent_variance"
                ),
                use_log_scale=True,
            )

            plot_scatter(
                ticker=ticker,
                prediction=predicted_rv,
                actual_rv=actual_rv,
                prediction_name=(
                    "Predicted realised measure"
                ),
                filename_suffix=(
                    "realised_measure"
                ),
                use_log_scale=True,
            )

            # -------------------------------------------------
            # Store results
            # -------------------------------------------------
            result = {
                "ticker": ticker,
                "n_total": int(
                    rolling_result[
                        "n_total"
                    ]
                ),
                "rolling_window_size": int(
                    rolling_result[
                        "rolling_window_size"
                    ]
                ),
                "n_oos": int(
                    rolling_result[
                        "n_oos"
                    ]
                ),
                "refit_every": int(
                    rolling_result[
                        "refit_every"
                    ]
                ),
                "QLIKE_raw": qlike_raw,
                "MSE_vol_raw": mse_vol_raw,
                "MSE_var_raw": mse_var_raw,
                "MAE_vol_raw": mae_vol_raw,
                "MAE_var_raw": mae_var_raw,
                "Return_NLL": return_nll_value,
                "QLIKE_rv": qlike_rv,
                "MSE_vol_rv": mse_vol_rv,
                "MSE_var_rv": mse_var_rv,
                "MAE_vol_rv": mae_vol_rv,
                "MAE_var_rv": mae_var_rv,
            }

            results.append(result)
            save_progress(results)

            print(
                f"\n{ticker} completed successfully."
            )

        except Exception as exc:
            print(
                f"\nFailed on {ticker}: "
                f"{type(exc).__name__}: {exc}"
            )

            results.append(
                {
                    "ticker": ticker,
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                }
            )

            save_progress(results)

    # -----------------------------------------------------
    # Final result table
    # -----------------------------------------------------
    results_df = pd.DataFrame(
        results
    )

    column_order = [
        "ticker",
        "n_total",
        "rolling_window_size",
        "n_oos",
        "refit_every",
        "QLIKE_raw",
        "MSE_vol_raw",
        "MSE_var_raw",
        "MAE_vol_raw",
        "MAE_var_raw",
        "Return_NLL",
        "QLIKE_rv",
        "MSE_vol_rv",
        "MSE_var_rv",
        "MAE_vol_rv",
        "MAE_var_rv",
    ]

    if "QLIKE_raw" in results_df.columns:
        successful_results = results_df[
            results_df[
                "QLIKE_raw"
            ].notna()
        ].copy()
    else:
        successful_results = pd.DataFrame()

    if not successful_results.empty:
        successful_results = (
            successful_results
            .sort_values(
                "QLIKE_raw",
                ascending=True,
            )
            .reset_index(drop=True)
        )

        successful_results = (
            successful_results[
                column_order
            ]
        )

        successful_results.to_csv(
            OUTPUT_DIR
            / (
                "realized_garch_rolling_"
                "results_sorted.csv"
            ),
            index=False,
        )

        print(
            "\n" + "=" * 140
        )
        print(
            "Final rolling Realized GARCH results"
        )
        print(
            "=" * 140
        )

        print(
            successful_results.to_string(
                index=False
            )
        )

    else:
        print(
            "\nNo valid rolling Realized GARCH "
            "results were produced."
        )


if __name__ == "__main__":
    main()