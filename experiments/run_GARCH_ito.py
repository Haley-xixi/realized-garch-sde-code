import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =========================================================
# Path setup
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from SDE_Garch_utils.utils1 import (
    build_data_dict,
    load_rv,
    set_seed_all,
)

from models.model_GARCH_ito import (
    get_garch_ito_arrays_from_data_dict,
    fixed_half_sample_rolling_garch_ito,
    summarize_rolling_garch_ito,
)


# =========================================================
# Configuration
# =========================================================

SEED = 1313

REFIT_EVERY = 21

WARM_START = False
INITIALIZATION = "paper"
DROP_INITIAL = False
PERSISTENCE_LIMIT = 0.999
MAXITER = 2000
VERBOSE_OPTIMIZER = False


# =========================================================
# Paths
# =========================================================

DATA_PATH = PROJECT_ROOT / "data" / "rv.pkl"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "GARCH_ito_RV_QMLE_rolling_50_50"
)

PREDICTION_DIR = OUTPUT_DIR / "predictions"
PARAMETER_DIR = OUTPUT_DIR / "parameter_history"
PLOT_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PARAMETER_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# Utility
# =========================================================

def ensure_1d_float(
    values,
    *,
    positive=False,
    eps=1e-12,
):
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


def save_progress(results):
    pd.DataFrame(results).to_csv(
        OUTPUT_DIR
        / "garch_ito_rolling_results.csv",
        index=False,
    )


# =========================================================
# Plot
# =========================================================

def plot_garch_ito_variance_forecast(
    ticker,
    h_pred,
    rv_true,
    save_path,
):
    h_pred = ensure_1d_float(
        h_pred,
        positive=True,
    )

    rv_true = ensure_1d_float(
        rv_true,
        positive=True,
    )

    if len(h_pred) != len(rv_true):
        raise ValueError(
            "h_pred and rv_true must have "
            "equal length."
        )

    x = np.arange(
        len(h_pred)
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        x,
        rv_true,
        label="Observed realized variance",
        linewidth=1.2,
        alpha=0.8,
    )

    ax.plot(
        x,
        h_pred,
        label="GARCH-Itô predicted variance",
        linewidth=1.5,
    )

    ax.set_xlabel(
        "Out-of-sample observation"
    )

    ax.set_ylabel(
        "Variance"
    )

    ax.set_title(
        f"{ticker}: rolling GARCH-Itô "
        "one-step-ahead variance forecasts"
    )

    ax.legend()
    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)



def plot_garch_ito_volatility_forecast(
    ticker,
    h_pred,
    rv_true,
    save_path,
):
    h_pred = ensure_1d_float(
        h_pred,
        positive=True,
    )

    rv_true = ensure_1d_float(
        rv_true,
        positive=True,
    )

    if len(h_pred) != len(rv_true):
        raise ValueError(
            "h_pred and rv_true must have "
            "equal length."
        )

    predicted_volatility = np.sqrt(
        h_pred
    )

    observed_volatility = np.sqrt(
        rv_true
    )

    x = np.arange(
        len(h_pred)
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        x,
        observed_volatility,
        label="Observed volatility",
        linewidth=1.2,
        alpha=0.8,
    )

    ax.plot(
        x,
        predicted_volatility,
        label="GARCH-Itô predicted volatility",
        linewidth=1.5,
    )

    ax.set_xlabel(
        "Out-of-sample observation"
    )

    ax.set_ylabel(
        "Volatility"
    )

    ax.set_title(
        f"{ticker}: rolling GARCH-Itô "
        "one-step-ahead volatility forecasts"
    )

    ax.legend()
    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# =========================================================
# Main experiment
# =========================================================

def main():
    set_seed_all(
        seed=SEED,
    )

    print(
        f"Data path       : {DATA_PATH}"
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    print(
        f"Refit frequency : every "
        f"{REFIT_EVERY} trading days"
    )


    df = load_rv(
        filepath=str(DATA_PATH),
    )

    data_dict = build_data_dict(
        df,
    )

    all_tickers = list(
        data_dict.keys()
    )

    # Retain your previous resume rule.
    selected_tickers = all_tickers[14:]

    results = []


    for ticker in selected_tickers:
        print(
            "\n" + "=" * 100
        )

        print(
            f"Processing rolling GARCH-Itô: "
            f"{ticker}"
        )

        print(
            "=" * 100
        )

        try:

            (
                y_train_old,
                rv_train_old,
                y_test_old,
                rv_test_old,
            ) = get_garch_ito_arrays_from_data_dict(
                data_dict=data_dict,
                ticker=ticker,
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
                    "Full return and RV arrays "
                    "have different lengths."
                )

            if len(y_full) < 6:
                raise ValueError(
                    "The complete sample is too short."
                )

            n_total = len(
                y_full
            )

            rolling_window_size = (
                n_total // 2
            )

            n_oos_expected = (
                n_total
                - rolling_window_size
            )

            initial_returns = y_full[
                :rolling_window_size
            ]

            initial_rv = rv_full[
                :rolling_window_size
            ]

            print(
                "\n=== Fixed 50:50 design ==="
            )

            print(
                f"Total observations   : "
                f"{n_total}"
            )

            print(
                f"Rolling-window size  : "
                f"{rolling_window_size}"
            )

            print(
                f"Expected OOS size    : "
                f"{n_oos_expected}"
            )

            print(
                f"Window step          : "
                f"{REFIT_EVERY}"
            )

            print(
                "\n=== Initial-window scale check ==="
            )

            print(
                "Mean return squared   : "
                f"{np.mean(initial_returns ** 2):.8f}"
            )

            print(
                "Mean raw RV           : "
                f"{np.mean(initial_rv):.8f}"
            )

            print(
                "Return²/RV mean ratio : "
                f"{np.mean(initial_returns ** 2) / np.mean(initial_rv):.8f}"
            )

            rolling_result = (
                fixed_half_sample_rolling_garch_ito(
                    ticker=ticker,
                    y_full=y_full,
                    rv_full=rv_full,
                    refit_every=REFIT_EVERY,
                    warm_start=WARM_START,
                    initialization=INITIALIZATION,
                    drop_initial=DROP_INITIAL,
                    persistence_limit=(
                        PERSISTENCE_LIMIT
                    ),
                    maxiter=MAXITER,
                    verbose=VERBOSE_OPTIMIZER,
                )
            )


            h_pred = ensure_1d_float(
                rolling_result[
                    "predicted_variance"
                ],
                positive=True,
            )

            rv_true = ensure_1d_float(
                rolling_result[
                    "actual_rv"
                ],
                positive=True,
            )

            y_true = ensure_1d_float(
                rolling_result[
                    "actual_returns"
                ],
                positive=False,
            )
            target_index = np.asarray(
                rolling_result[
                    "target_index"
                ],
                dtype=np.int64,
            ).reshape(-1)

            if not (
                len(h_pred)
                == len(rv_true)
                == len(y_true)
                == len(target_index)
            ):
                raise ValueError(
                    "Forecast and target arrays "
                    "are not aligned."
                )

            n_eval = len(
                h_pred
            )
            if len(target_index) > 1:
                if not np.all(
                    np.diff(target_index) == 1
                ):
                    raise ValueError(
                        f"{ticker}: target_index is not consecutive."
                    )
    
            if n_eval != n_oos_expected:
                raise ValueError(
                    f"Expected {n_oos_expected} "
                    f"out-of-sample forecasts but "
                    f"received {n_eval}."
                )

            # ---------------------------------------------
            # Summary metrics
            # ---------------------------------------------

            summary = (
                summarize_rolling_garch_ito(
                    rolling_result
                )
            )

            qlike = float(
                summary["QLIKE_raw"]
            )

            mse_vol = float(
                summary["MSE_vol_raw"]
            )

            mse_var = float(
                summary["MSE_var_raw"]
            )

            mae_vol = float(
                summary["MAE_vol_raw"]
            )

            mae_var = float(
                summary["MAE_var_raw"]
            )

            test_return_nll = float(
                summary["Return_NLL"]
            )

            print(
                "\n=== Rolling OOS metrics ==="
            )

            print(
                f"QLIKE      : {qlike:.8f}"
            )

            print(
                f"MSE vol    : {mse_vol:.8f}"
            )

            print(
                f"MSE var    : {mse_var:.8f}"
            )

            print(
                f"MAE vol    : {mae_vol:.8f}"
            )

            print(
                f"MAE var    : {mae_var:.8f}"
            )

            print(
                f"Return NLL : "
                f"{test_return_nll:.8f}"
            )

            print(
                "\n=== Forecast diagnostics ==="
            )

            print(
                "Mean predicted variance : "
                f"{np.mean(h_pred):.8f}"
            )

            print(
                "Mean observed RV        : "
                f"{np.mean(rv_true):.8f}"
            )

            print(
                "Mean return squared     : "
                f"{np.mean(y_true ** 2):.8f}"
            )

            print(
                "Mean return²/predicted h: "
                f"{np.mean(y_true ** 2 / h_pred):.8f}"
            )

            print(
                "Minimum prediction      : "
                f"{np.min(h_pred):.8f}"
            )

            print(
                "Maximum prediction      : "
                f"{np.max(h_pred):.8f}"
            )


            np.save(
                PREDICTION_DIR
                / (
                    f"GARCH_ito_rolling_"
                    f"{ticker}_pred_var.npy"
                ),
                h_pred,
            )

            np.save(
                PREDICTION_DIR
                / (
                    f"GARCH_ito_rolling_"
                    f"{ticker}_true_rv.npy"
                ),
                rv_true,
            )
            np.save(
                PREDICTION_DIR
                / (
                    f"GARCH_ito_rolling_"
                    f"{ticker}_target_index.npy"
                ),
                target_index,
            )
            
            np.save(
                PREDICTION_DIR
                / (
                    f"GARCH_ito_rolling_"
                    f"{ticker}_returns.npy"
                ),
                y_true,
            )

            np.savez(
                PREDICTION_DIR
                / (
                    f"GARCH_ito_rolling_"
                    f"{ticker}_forecasts.npz"
                ),
                predicted_variance=h_pred,
                actual_rv=rv_true,
                actual_returns=y_true,
                target_index=target_index,
            )


            parameter_history_df = pd.DataFrame(
                rolling_result[
                    "parameter_history"
                ]
            )

            parameter_history_df.to_csv(
                PARAMETER_DIR
                / (
                    f"GARCH_ito_rolling_"
                    f"{ticker}_parameters.csv"
                ),
                index=False,
            )

            if (
                "refit"
                in parameter_history_df.columns
            ):
                refit_history_df = (
                    parameter_history_df[
                        parameter_history_df[
                            "refit"
                        ].astype(bool)
                    ]
                    .copy()
                    .reset_index(drop=True)
                )

                refit_history_df.to_csv(
                    PARAMETER_DIR
                    / (
                        f"GARCH_ito_rolling_"
                        f"{ticker}_refits.csv"
                    ),
                    index=False,
                )


            plot_garch_ito_variance_forecast(
                ticker=ticker,
                h_pred=h_pred,
                rv_true=rv_true,
                save_path=(
                    PLOT_DIR
                    / (
                        f"GARCH_ito_rolling_"
                        f"{ticker}_variance.png"
                    )
                ),
            )

            plot_garch_ito_volatility_forecast(
                ticker=ticker,
                h_pred=h_pred,
                rv_true=rv_true,
                save_path=(
                    PLOT_DIR
                    / (
                        f"GARCH_ito_rolling_"
                        f"{ticker}_volatility.png"
                    )
                ),
            )


            if parameter_history_df.empty:
                raise ValueError(
                    "Parameter history is empty."
                )

            final_row = (
                parameter_history_df.iloc[-1]
            )

            result = {
                "ticker": ticker,

                # Rolling design
                "n_total": int(
                    summary["n_total"]
                ),
                "rolling_window_size": int(
                    summary[
                        "rolling_window_size"
                    ]
                ),
                "n_oos": int(
                    summary["n_oos"]
                ),
                "refit_every": int(
                    summary["refit_every"]
                ),
                "window_step": int(
                    summary["window_step"]
                ),

                # Forecast diagnostics
                "mean_predicted_variance": float(
                    np.mean(h_pred)
                ),
                "mean_oos_raw_RV": float(
                    np.mean(rv_true)
                ),
                "min_predicted_variance": float(
                    np.min(h_pred)
                ),
                "max_predicted_variance": float(
                    np.max(h_pred)
                ),

                # OOS metrics
                "QLIKE_raw": qlike,
                "MSE_vol_raw": mse_vol,
                "MSE_var_raw": mse_var,
                "MAE_vol_raw": mae_vol,
                "MAE_var_raw": mae_var,
                "Return_NLL": test_return_nll,

                # Final structural parameters
                "final_omega": float(
                    final_row["omega"]
                ),
                "final_beta_ct": float(
                    final_row["beta_ct"]
                ),
                "final_gamma": float(
                    final_row["gamma"]
                ),

                # Final implied daily parameters
                "final_omega_g": float(
                    final_row["omega_g"]
                ),
                "final_beta_g": float(
                    final_row["beta_g"]
                ),
                "final_persistence": float(
                    final_row["persistence"]
                ),

                # Average rolling parameters
                "mean_omega": float(
                    parameter_history_df[
                        "omega"
                    ].mean()
                ),
                "mean_beta_ct": float(
                    parameter_history_df[
                        "beta_ct"
                    ].mean()
                ),
                "mean_gamma": float(
                    parameter_history_df[
                        "gamma"
                    ].mean()
                ),
                "mean_omega_g": float(
                    parameter_history_df[
                        "omega_g"
                    ].mean()
                ),
                "mean_beta_g": float(
                    parameter_history_df[
                        "beta_g"
                    ].mean()
                ),
                "mean_persistence": float(
                    parameter_history_df[
                        "persistence"
                    ].mean()
                ),

                # Optimization diagnostics
                "n_refits": int(
                    parameter_history_df[
                        "refit"
                    ].astype(bool).sum()
                ),
                "n_refits_accepted": int(
                    parameter_history_df[
                        "refit_accepted"
                    ].astype(bool).sum()
                ),

                "model": "GARCH-Itô",
                "loss_type": "RV-QMLE",
                "evaluation": (
                    "50:50 block rolling, "
                    f"refit every {REFIT_EVERY} days"
                ),
            }

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
            print(
                f"\n{ticker} failed: "
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

            save_progress(
                results
            )

    # =====================================================
    # Save final summary
    # =====================================================

    results_df = pd.DataFrame(
        results
    )

    if (
        not results_df.empty
        and "QLIKE_raw"
        in results_df.columns
    ):
        successful_results = (
            results_df[
                results_df[
                    "QLIKE_raw"
                ].notna()
            ]
            .copy()
            .sort_values(
                "QLIKE_raw",
                ascending=True,
            )
            .reset_index(drop=True)
        )

    else:
        successful_results = (
            pd.DataFrame()
        )

    sorted_save_path = (
        OUTPUT_DIR
        / "garch_ito_rolling_results_sorted.csv"
    )

    successful_results.to_csv(
        sorted_save_path,
        index=False,
    )

    print(
        "\n" + "=" * 110
    )

    print(
        "Final rolling GARCH-Itô RV-QMLE results"
    )

    print(
        "=" * 110
    )

    if successful_results.empty:
        print(
            "No successful model results."
        )

    else:
        display_columns = [
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
            "final_omega",
            "final_beta_ct",
            "final_gamma",
            "final_omega_g",
            "final_beta_g",
            "final_persistence",
            "n_refits",
            "n_refits_accepted",
        ]

        print(
            successful_results[
                display_columns
            ].to_string(
                index=False
            )
        )

    print(
        f"\nSaved summary to: "
        f"{sorted_save_path}"
    )

    print(
        f"Saved predictions to: "
        f"{PREDICTION_DIR}"
    )

    print(
        f"Saved parameter histories to: "
        f"{PARAMETER_DIR}"
    )

    print(
        f"Saved plots to: "
        f"{PLOT_DIR}"
    )


if __name__ == "__main__":
    main()