import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# =========================================================
# Path setup
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "rv.pkl"
)


# =========================================================
# Configuration
# =========================================================

SEED = 7

EPS = 1e-8

INIT_MODE = "from_rv"

TRANSFORM = "exp"

LOSS_TYPE = "joint_qnll"


FEEDBACK_SPECIFICATION = "naive_gamma_log_R"


# ---------------------------------------------------------
# Ablations
# ---------------------------------------------------------
#
# Only two ablations are supported:
#
# 1. No realized-measure feedback:
#
#       GAMMA_FIXED_ZERO = True
#
#    imposes:
#
#       gamma = 0
#
# 2. Independent Brownian motions:
#
#       INDEPENDENT_BROWNIAN = True
#
#    imposes:
#
#       rho_Ys = 0
#       rho_YR = 0
#
# For the full proposed SDE, set both flags to False.
#
# The two ablations must be run separately.
# ---------------------------------------------------------

GAMMA_FIXED_ZERO = False

INDEPENDENT_BROWNIAN = True


# ---------------------------------------------------------
# Rolling forecast
# ---------------------------------------------------------

REFIT_EVERY = 21

N_SUBSTEPS_TRAIN = 10

N_SUBSTEPS_FORECAST = 20

N_PARTICLES_TRAIN = 64

N_PARTICLES_FORECAST = 256

N_EPOCHS_INITIAL = 1000

N_EPOCHS_REFIT = 300

LEARNING_RATE = 1e-4

BATCH_SIZE = 32

LAMBDA_RV = 1.0

COVARIANCE_JITTER = 1e-4

GRAD_CLIP = 5.0


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


if (
    GAMMA_FIXED_ZERO
    and INDEPENDENT_BROWNIAN
):
    raise ValueError(
        "GAMMA_FIXED_ZERO and INDEPENDENT_BROWNIAN "
        "are separate ablations and should not both be True."
    )


if GAMMA_FIXED_ZERO:

    EXPERIMENT_NAME = (
        "ablation_gamma0"
    )

elif INDEPENDENT_BROWNIAN:

    EXPERIMENT_NAME = (
        "ablation_indep_bm111"
    )

else:

    EXPERIMENT_NAME = (
        "proposed_SDE"
    )


RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / EXPERIMENT_NAME
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
    set_seed_all,
)

from models.proposed_SDE import (
    fixed_half_sample_rolling_forecast,
    get_ou_arrays,
    mse_var_np,
    mse_vol_np,
    qlike_var_np,
    return_nll_np,
    mae_var_np,
    mae_vol_np,
    save_prediction_scatter_plot,
    save_rolling_correlation_plot,
    save_variance_plot,
    save_realised_state_plot,
)


# =========================================================
# Utilities
# =========================================================

def ensure_1d_float(
    values,
    *,
    positive=False,
    eps=EPS,
):
    array = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    if len(array) == 0:
        raise ValueError(
            "Array is empty."
        )

    if not np.all(
        np.isfinite(array)
    ):
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
    results,
):
    pd.DataFrame(
        results
    ).to_csv(
        RESULTS_DIR
        / "rolling_50_50_results.csv",
        index=False,
    )


def validate_parameter_history(
    parameter_history,
):
    required_keys = {
        "forecast_number",
        "target_index",
        "window_start",
        "window_end",
        "refit",

        "kappa",
        "kappa_z",
        "theta",
        "gamma",
        "xi",
        "phi",

        "gamma_phi",
        "stability_margin",

        "eta_s",
        "eta_R",
        "sigma_u",

        "rho_Ys",
        "rho_YR",

        "mean_reversion_drift",
        "feedback_drift",
        "measurement_reversion_drift",
    }

    missing = (
        required_keys.difference(
            parameter_history.keys()
        )
    )

    if missing:
        raise KeyError(
            "parameter_history is missing keys: "
            f"{sorted(missing)}"
        )

    lengths = {
        key:
            len(
                np.asarray(
                    parameter_history[
                        key
                    ]
                ).reshape(-1)
            )
        for key in required_keys
    }

    if len(
        set(
            lengths.values()
        )
    ) != 1:
        raise ValueError(
            "Parameter-history arrays "
            "have different lengths:\n"
            + "\n".join(
                f"  {key}: {value}"
                for key, value
                in sorted(
                    lengths.items()
                )
            )
        )


# =========================================================
# Main experiment
# =========================================================

def main():

    set_seed_all(
        SEED
    )

    # =====================================================
    # Configuration
    # =====================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "PROPOSED SDE CONFIGURATION"
    )

    print(
        "=" * 100
    )

    print(
        f"Device                     : {DEVICE}"
    )

    print(
        f"Data path                  : {DATA_PATH}"
    )

    print(
        f"Experiment                 : {EXPERIMENT_NAME}"
    )

    print(
        f"Results directory          : {RESULTS_DIR}"
    )

    print(
        "Measurement dynamics       : "
        "mean-reverting z"
    )

    print(
        "z drift                    : "
        "kappa_z * (xi + phi*x - z)"
    )

    print(
        "Naive feedback             : "
        "gamma * z"
    )

    print(
        "Feedback specification     : "
        f"{FEEDBACK_SPECIFICATION}"
    )

    print(
        "Stability diagnostic       : "
        "kappa - gamma*phi > 0"
    )

    print(
        "Gamma fixed at zero        : "
        f"{GAMMA_FIXED_ZERO}"
    )

    print(
        "Independent Brownian       : "
        f"{INDEPENDENT_BROWNIAN}"
    )

    print(
        "Rolling refit frequency    : "
        f"every {REFIT_EVERY} forecast(s)"
    )

    print(
        "=" * 100
    )

    # =====================================================
    # Load data
    # =====================================================

    df = load_rv(
        filepath=str(
            DATA_PATH
        )
    )

    data_dict = build_data_dict(
        df
    )

    all_tickers = list(
        data_dict.keys()
    )

    # -----------------------------------------------------
    # Keep this selection unchanged as requested.
    # -----------------------------------------------------

    selected_tickers = (
        all_tickers[14:]
    )

    results = []

    # =====================================================
    # Ticker loop
    # =====================================================

    for ticker in selected_tickers:

        print(
            "\n"
            + "=" * 80
        )

        print(
            "Processing fixed 50:50 "
            f"rolling window: {ticker}"
        )

        print(
            "=" * 80
        )

        try:

            # =================================================
            # Build full chronological sample
            # =================================================

            (
                log_prices_train_old,
                rv_train_old,
                log_prices_test_old,
                rv_test_old,
            ) = get_ou_arrays(
                data_dict=data_dict,
                ticker=ticker,
                eps=EPS,
            )

            log_prices_full = np.concatenate(
                [
                    log_prices_train_old,
                    log_prices_test_old,
                ]
            )

            rv_full = np.concatenate(
                [
                    rv_train_old,
                    rv_test_old,
                ]
            )

            log_prices_full = ensure_1d_float(
                log_prices_full,
                positive=False,
            )

            rv_full = ensure_1d_float(
                rv_full,
                positive=True,
            )

            if (
                len(log_prices_full)
                != len(rv_full)
            ):
                raise ValueError(
                    "Full log-price and RV lengths differ: "
                    f"{len(log_prices_full)} versus "
                    f"{len(rv_full)}."
                )

            if len(log_prices_full) < 4:
                raise ValueError(
                    "Insufficient full-sample observations: "
                    f"{len(log_prices_full)}."
                )

            # =================================================
            # Fixed 50:50 design
            # =================================================

            n_total = len(
                log_prices_full
            )

            rolling_window_size = (
                n_total // 2
            )

            n_oos_expected = (
                n_total
                - rolling_window_size
            )

            initial_y = (
                log_prices_full[
                    :rolling_window_size
                ]
            )

            initial_rv = (
                rv_full[
                    :rolling_window_size
                ]
            )

            initial_returns = (
                100.0
                * np.diff(
                    initial_y
                )
            )

            aligned_initial_rv = (
                initial_rv[
                    1:
                ]
            )

            print(
                "\n=== Fixed 50:50 design ==="
            )

            print(
                "Full observations        : "
                f"{n_total}"
            )

            print(
                "Rolling-window size      : "
                f"{rolling_window_size}"
            )

            print(
                "Expected OOS forecasts   : "
                f"{n_oos_expected}"
            )

            print(
                "First OOS target index   : "
                f"{rolling_window_size}"
            )

            print(
                "\n=== Initial-window diagnostics ==="
            )

            print(
                "Mean squared initial return: "
                f"{np.mean(initial_returns ** 2):.8f}"
            )

            print(
                "Mean aligned initial RV     : "
                f"{np.mean(aligned_initial_rv):.8f}"
            )

            print(
                "Median aligned initial RV   : "
                f"{np.median(aligned_initial_rv):.8f}"
            )

            print(
                "Mean return² / mean RV      : "
                f"{np.mean(initial_returns ** 2) / np.mean(aligned_initial_rv):.8f}"
            )

            # =================================================
            # Rolling forecast
            # =================================================

            (
                predicted_log_price,
                predicted_variance,
                predicted_rv_state,
                actual_next_log_price,
                returns_test_eval,
                rv_test_eval,
                parameter_history,
            ) = fixed_half_sample_rolling_forecast(
                ticker=ticker,

                y_full=log_prices_full,
                rv_full=rv_full,

                refit_every=REFIT_EVERY,

                init_mode=INIT_MODE,
                transform=TRANSFORM,
                loss_type=LOSS_TYPE,

                n_substeps_train=(
                    N_SUBSTEPS_TRAIN
                ),

                n_particles_train=(
                    N_PARTICLES_TRAIN
                ),

                n_epochs_initial=(
                    N_EPOCHS_INITIAL
                ),

                n_epochs_refit=(
                    N_EPOCHS_REFIT
                ),

                lr=LEARNING_RATE,

                batch_size=BATCH_SIZE,

                lambda_rv=LAMBDA_RV,

                covariance_jitter=(
                    COVARIANCE_JITTER
                ),

                grad_clip=GRAD_CLIP,

                n_particles_forecast=(
                    N_PARTICLES_FORECAST
                ),

                n_substeps_forecast=(
                    N_SUBSTEPS_FORECAST
                ),

                device=DEVICE,

                # ---------------------------------------------
                # Ablation switches
                # ---------------------------------------------

                gamma_fixed_zero=(
                    GAMMA_FIXED_ZERO
                ),

                independent_brownian=(
                    INDEPENDENT_BROWNIAN
                ),
            )

            # =================================================
            # Convert rolling outputs
            # =================================================

            predicted_log_price = ensure_1d_float(
                predicted_log_price,
                positive=False,
            )

            predicted_variance = ensure_1d_float(
                predicted_variance,
                positive=True,
            )

            predicted_rv_state = ensure_1d_float(
                predicted_rv_state,
                positive=True,
            )

            actual_next_log_price = ensure_1d_float(
                actual_next_log_price,
                positive=False,
            )

            returns_test_eval = ensure_1d_float(
                returns_test_eval,
                positive=False,
            )

            rv_test_eval = ensure_1d_float(
                rv_test_eval,
                positive=True,
            )

            forecast_lengths = {
                len(predicted_log_price),
                len(predicted_variance),
                len(predicted_rv_state),
                len(actual_next_log_price),
                len(returns_test_eval),
                len(rv_test_eval),
            }

            if len(
                forecast_lengths
            ) != 1:
                raise ValueError(
                    "Rolling output arrays are not aligned: "
                    f"{sorted(forecast_lengths)}"
                )

            n_eval = len(
                rv_test_eval
            )

            if n_eval == 0:
                raise ValueError(
                    "No rolling forecasts generated "
                    f"for {ticker}."
                )

            if (
                n_eval
                != n_oos_expected
            ):
                raise ValueError(
                    f"Expected {n_oos_expected} "
                    "OOS forecasts but received "
                    f"{n_eval}."
                )

            # =================================================
            # Parameter-history checks
            # =================================================

            validate_parameter_history(
                parameter_history
            )

            history_length = len(
                np.asarray(
                    parameter_history[
                        "forecast_number"
                    ]
                ).reshape(-1)
            )

            if (
                history_length
                != n_eval
            ):
                raise ValueError(
                    "Parameter-history length does not "
                    "match forecast length: "
                    f"{history_length} versus "
                    f"{n_eval}."
                )

            # =================================================
            # Refit mask
            # =================================================

            refit_mask = np.asarray(
                parameter_history[
                    "refit"
                ],
                dtype=bool,
            )

            if not np.any(
                refit_mask
            ):
                raise RuntimeError(
                    "No refit observations were found "
                    "in parameter_history."
                )

            # =================================================
            # Parameter diagnostics
            # =================================================

            kappa_refits = np.asarray(
                parameter_history[
                    "kappa"
                ],
                dtype=np.float64,
            )[
                refit_mask
            ]

            kappa_z_refits = np.asarray(
                parameter_history[
                    "kappa_z"
                ],
                dtype=np.float64,
            )[
                refit_mask
            ]

            gamma_refits = np.asarray(
                parameter_history[
                    "gamma"
                ],
                dtype=np.float64,
            )[
                refit_mask
            ]

            gamma_phi_refits = np.asarray(
                parameter_history[
                    "gamma_phi"
                ],
                dtype=np.float64,
            )[
                refit_mask
            ]

            stability_margin_refits = np.asarray(
                parameter_history[
                    "stability_margin"
                ],
                dtype=np.float64,
            )[
                refit_mask
            ]

            rho_Ys_history = np.asarray(
                parameter_history[
                    "rho_Ys"
                ],
                dtype=np.float64,
            )

            rho_YR_history = np.asarray(
                parameter_history[
                    "rho_YR"
                ],
                dtype=np.float64,
            )

            feedback_history = np.asarray(
                parameter_history[
                    "feedback_drift"
                ],
                dtype=np.float64,
            )

            arrays_to_check = {
                "kappa":
                    kappa_refits,

                "kappa_z":
                    kappa_z_refits,

                "gamma":
                    gamma_refits,

                "gamma_phi":
                    gamma_phi_refits,

                "stability_margin":
                    stability_margin_refits,

                "rho_Ys":
                    rho_Ys_history,

                "rho_YR":
                    rho_YR_history,

                "feedback_drift":
                    feedback_history,
            }

            for (
                parameter_name,
                parameter_values,
            ) in arrays_to_check.items():

                if not np.all(
                    np.isfinite(
                        parameter_values
                    )
                ):
                    raise RuntimeError(
                        f"{parameter_name} contains "
                        "non-finite values."
                    )

            # =================================================
            # Stability diagnostics
            # =================================================
            #
            # For
            #
            # dx = [-kappa*x + gamma*z + ...] dt
            #
            # dz = [kappa_z*phi*x - kappa_z*z + ...] dt
            #
            # with kappa > 0 and kappa_z > 0,
            # stability requires:
            #
            #     kappa - gamma*phi > 0
            #
            # =================================================

            stable_all_refits = bool(
                np.all(
                    stability_margin_refits
                    > 0.0
                )
            )

            print(
                "\n=== Stability diagnostics ==="
            )

            print(
                "Mean kappa across refits        : "
                f"{np.mean(kappa_refits):.6f}"
            )

            print(
                "Minimum kappa across refits     : "
                f"{np.min(kappa_refits):.6f}"
            )

            print(
                "Mean kappa_z across refits      : "
                f"{np.mean(kappa_z_refits):.6f}"
            )

            print(
                "Minimum kappa_z across refits   : "
                f"{np.min(kappa_z_refits):.6f}"
            )

            print(
                "Mean gamma across refits        : "
                f"{np.mean(gamma_refits):.6f}"
            )

            print(
                "Mean gamma*phi across refits    : "
                f"{np.mean(gamma_phi_refits):.6f}"
            )

            print(
                "Maximum gamma*phi across refits : "
                f"{np.max(gamma_phi_refits):.6f}"
            )

            print(
                "Mean kappa-gamma*phi            : "
                f"{np.mean(stability_margin_refits):.6f}"
            )

            print(
                "Minimum kappa-gamma*phi         : "
                f"{np.min(stability_margin_refits):.6f}"
            )

            print(
                "Stable at every refit           : "
                f"{stable_all_refits}"
            )

            print(
                "Mean naive feedback drift       : "
                f"{np.mean(feedback_history):.6f}"
            )

            # =================================================
            # gamma = 0 ablation sanity check
            # =================================================

            if GAMMA_FIXED_ZERO:

                gamma_history = np.asarray(
                    parameter_history[
                        "gamma"
                    ],
                    dtype=np.float64,
                )

                gamma_phi_history = np.asarray(
                    parameter_history[
                        "gamma_phi"
                    ],
                    dtype=np.float64,
                )

                if not np.allclose(
                    gamma_history,
                    0.0,
                    atol=1e-12,
                ):
                    raise RuntimeError(
                        "gamma=0 ablation failed: "
                        "gamma is not identically zero."
                    )

                if not np.allclose(
                    gamma_phi_history,
                    0.0,
                    atol=1e-12,
                ):
                    raise RuntimeError(
                        "gamma=0 ablation failed: "
                        "gamma*phi is not identically zero."
                    )

                if not np.allclose(
                    feedback_history,
                    0.0,
                    atol=1e-12,
                ):
                    raise RuntimeError(
                        "gamma=0 ablation failed: "
                        "feedback drift is not identically zero."
                    )

                print(
                    "\n=== gamma=0 sanity check ==="
                )

                print(
                    "gamma identically zero         : True"
                )

                print(
                    "gamma*phi identically zero     : True"
                )

                print(
                    "feedback drift identically zero: True"
                )

            # =================================================
            # Independent-Brownian sanity check
            # =================================================

            if INDEPENDENT_BROWNIAN:

                if not np.allclose(
                    rho_Ys_history,
                    0.0,
                    atol=1e-12,
                ):
                    raise RuntimeError(
                        "Independent-Brownian ablation failed: "
                        "rho_Ys is not identically zero."
                    )

                if not np.allclose(
                    rho_YR_history,
                    0.0,
                    atol=1e-12,
                ):
                    raise RuntimeError(
                        "Independent-Brownian ablation failed: "
                        "rho_YR is not identically zero."
                    )

                print(
                    "\n=== Independent-Brownian sanity check ==="
                )

                print(
                    "rho_Ys identically zero: True"
                )

                print(
                    "rho_YR identically zero: True"
                )

            else:

                rho_Ys_refits = (
                    rho_Ys_history[
                        refit_mask
                    ]
                )

                rho_YR_refits = (
                    rho_YR_history[
                        refit_mask
                    ]
                )

                print(
                    "\n=== Brownian-correlation diagnostics ==="
                )

                print(
                    "Mean rho_Ys across refits   : "
                    f"{np.mean(rho_Ys_refits):.6f}"
                )

                print(
                    "Median rho_Ys across refits : "
                    f"{np.median(rho_Ys_refits):.6f}"
                )

                print(
                    "Mean rho_YR across refits   : "
                    f"{np.mean(rho_YR_refits):.6f}"
                )

                print(
                    "Median rho_YR across refits : "
                    f"{np.median(rho_YR_refits):.6f}"
                )

            # =================================================
            # Forecast diagnostics
            # =================================================

            print(
                "\n=== Rolling forecast diagnostics ==="
            )

            print(
                f"Generated OOS forecasts : {n_eval}"
            )

            print(
                "Mean percentage return  : "
                f"{np.mean(returns_test_eval):.8f}"
            )

            print(
                "Std percentage return   : "
                f"{np.std(returns_test_eval):.8f}"
            )

            print(
                "Mean squared return     : "
                f"{np.mean(returns_test_eval ** 2):.8f}"
            )

            print(
                "Mean predicted variance : "
                f"{np.mean(predicted_variance):.8f}"
            )

            print(
                "Mean OOS raw RV         : "
                f"{np.mean(rv_test_eval):.8f}"
            )

            if not np.all(
                np.isfinite(
                    predicted_variance
                )
            ):
                raise ValueError(
                    "Non-finite variance forecasts "
                    f"for {ticker}."
                )

            if np.any(
                predicted_variance
                <= 0.0
            ):
                raise ValueError(
                    "Non-positive variance forecasts "
                    f"for {ticker}."
                )

            # =================================================
            # Evaluation metrics
            # =================================================

            qlike = qlike_var_np(
                predicted_variance,
                rv_test_eval,
            )

            mse_vol = mse_vol_np(
                predicted_variance,
                rv_test_eval,
            )

            mse_var = mse_var_np(
                predicted_variance,
                rv_test_eval,
            )

            mae_vol = mae_vol_np(
                predicted_variance,
                rv_test_eval,
            )

            mae_var = mae_var_np(
                predicted_variance,
                rv_test_eval,
            )

            return_nll = return_nll_np(
                predicted_variance,
                returns_test_eval,
            )

            qlike_rv = qlike_var_np(
                predicted_rv_state,
                rv_test_eval,
            )

            mse_vol_rv = mse_vol_np(
                predicted_rv_state,
                rv_test_eval,
            )

            mse_var_rv = mse_var_np(
                predicted_rv_state,
                rv_test_eval,
            )

            mae_vol_rv = mae_vol_np(
                predicted_rv_state,
                rv_test_eval,
            )

            mae_var_rv = mae_var_np(
                predicted_rv_state,
                rv_test_eval,
            )

            print(
                "\n=== Fixed 50:50 rolling OOS metrics ==="
            )

            print(
                f"QLIKE H      : {qlike:.8f}"
            )

            print(
                f"MSE vol H    : {mse_vol:.8f}"
            )

            print(
                f"MSE var H    : {mse_var:.8f}"
            )

            print(
                f"MAE vol H    : {mae_vol:.8f}"
            )

            print(
                f"MAE var H    : {mae_var:.8f}"
            )

            print(
                f"Return NLL H : {return_nll:.8f}"
            )

            print(
                f"QLIKE R      : {qlike_rv:.8f}"
            )

            print(
                f"MSE vol R    : {mse_vol_rv:.8f}"
            )

            print(
                f"MSE var R    : {mse_var_rv:.8f}"
            )

            print(
                f"MAE vol R    : {mae_vol_rv:.8f}"
            )

            print(
                f"MAE var R    : {mae_var_rv:.8f}"
            )

            # =================================================
            # Save forecast arrays
            # =================================================

            np.save(
                RESULTS_DIR
                / f"{ticker}_rolling_log_price_pred.npy",
                predicted_log_price,
            )

            np.save(
                RESULTS_DIR
                / f"{ticker}_rolling_log_price_target.npy",
                actual_next_log_price,
            )

            np.save(
                RESULTS_DIR
                / f"{ticker}_rolling_variance_pred.npy",
                predicted_variance,
            )

            np.save(
                RESULTS_DIR
                / f"{ticker}_rolling_rv_state_pred.npy",
                predicted_rv_state,
            )

            np.save(
                RESULTS_DIR
                / f"{ticker}_rolling_rv_target.npy",
                rv_test_eval,
            )

            np.save(
                RESULTS_DIR
                / f"{ticker}_rolling_returns_target.npy",
                returns_test_eval,
            )

            # =================================================
            # Parameter history
            # =================================================

            parameter_df = pd.DataFrame(
                {
                    key:
                        parameter_history[
                            key
                        ]
                    for key in [
                        "forecast_number",
                        "target_index",
                        "window_start",
                        "window_end",
                        "refit",
                        "kappa",
                        "kappa_z",
                        "theta",
                        "gamma",
                        "xi",
                        "phi",
                        "gamma_phi",
                        "stability_margin",
                        "eta_s",
                        "eta_R",
                        "sigma_u",
                        "rho_Ys",
                        "rho_YR",
                        "mean_reversion_drift",
                        "feedback_drift",
                        "measurement_reversion_drift",
                    ]
                }
            )

            parameter_df.to_csv(
                RESULTS_DIR
                / f"{ticker}_rolling_parameter_history.csv",
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
                / f"{ticker}_rolling_parameter_refits.csv",
                index=False,
            )

            # =================================================
            # Complete NPZ
            # =================================================

            np.savez(
                RESULTS_DIR
                / f"{ticker}_rolling_50_50_forecasts.npz",

                predicted_log_price=(
                    predicted_log_price
                ),

                predicted_variance=(
                    predicted_variance
                ),

                predicted_rv_state=(
                    predicted_rv_state
                ),

                actual_next_log_price=(
                    actual_next_log_price
                ),

                returns_test_eval=(
                    returns_test_eval
                ),

                rv_test_eval=(
                    rv_test_eval
                ),

                **{
                    key:
                        np.asarray(
                            parameter_history[
                                key
                            ]
                        )
                    for key in [
                        "forecast_number",
                        "target_index",
                        "window_start",
                        "window_end",
                        "refit",
                        "kappa",
                        "kappa_z",
                        "theta",
                        "gamma",
                        "xi",
                        "phi",
                        "gamma_phi",
                        "stability_margin",
                        "eta_s",
                        "eta_R",
                        "sigma_u",
                        "rho_Ys",
                        "rho_YR",
                        "mean_reversion_drift",
                        "feedback_drift",
                        "measurement_reversion_drift",
                    ]
                },
            )

            # =================================================
            # Plots
            # =================================================

            save_variance_plot(
                ticker=ticker,
                actual_rv=rv_test_eval,
                predicted_variance=predicted_variance,
                output_dir=RESULTS_DIR,
            )

            save_realised_state_plot(
                ticker=ticker,
                actual_rv=rv_test_eval,
                predicted_rv_state=predicted_rv_state,
                output_dir=RESULTS_DIR,
            )

            save_prediction_scatter_plot(
                ticker=ticker,
                actual_rv=rv_test_eval,
                predicted_variance=predicted_variance,
                output_dir=RESULTS_DIR,
                prediction_label=(
                    "Predicted integrated variance"
                ),
                use_log_scale=False,
            )

            save_prediction_scatter_plot(
                ticker=f"{ticker}_H_log",
                actual_rv=rv_test_eval,
                predicted_variance=predicted_variance,
                output_dir=RESULTS_DIR,
                prediction_label=(
                    "Predicted integrated variance"
                ),
                use_log_scale=True,
            )

            save_prediction_scatter_plot(
                ticker=f"{ticker}_R_log",
                actual_rv=rv_test_eval,
                predicted_variance=predicted_rv_state,
                output_dir=RESULTS_DIR,
                prediction_label=(
                    "Predicted realised-measure state"
                ),
                use_log_scale=True,
            )

            save_rolling_correlation_plot(
                ticker=ticker,
                parameter_history=parameter_history,
                output_dir=RESULTS_DIR,
                plot_only_refits=True,
            )

            # =================================================
            # Model name
            # =================================================

            if GAMMA_FIXED_ZERO:

                model_name = (
                    "Naive-Gamma-zero-MR-z"
                )

            elif INDEPENDENT_BROWNIAN:

                model_name = (
                    "Naive-Independent-Brownian-MR-z"
                )

            else:

                model_name = (
                    "Naive-Full-SDE-MR-z"
                )

            # =================================================
            # Feedback label
            # =================================================

            if GAMMA_FIXED_ZERO:

                result_feedback_specification = (
                    "gamma_zero"
                )

            else:

                result_feedback_specification = (
                    FEEDBACK_SPECIFICATION
                )

            # =================================================
            # Summary row
            # =================================================

            result = {
                "ticker":
                    ticker,

                "model":
                    model_name,

                "measurement_dynamics":
                    "mean_reverting_z",

                "z_drift_specification":
                    "kappa_z*(xi+phi*x-z)",

                "feedback_specification":
                    result_feedback_specification,

                "stability_condition":
                    "kappa-gamma*phi>0",

                # -------------------------------------------------
                # Ablation indicators
                # -------------------------------------------------

                "gamma_fixed_zero":
                    GAMMA_FIXED_ZERO,

                "independent_brownian":
                    INDEPENDENT_BROWNIAN,

                # -------------------------------------------------
                # Diffusion/correlation metadata
                # -------------------------------------------------

                "eta_s_fixed_zero":
                    False,

                "eta_R_fixed_zero":
                    False,

                "sigma_u_fixed_zero":
                    False,

                "rho_Ys_fixed_zero":
                    INDEPENDENT_BROWNIAN,

                "rho_YR_fixed_zero":
                    INDEPENDENT_BROWNIAN,

                "rho_Ys_active": (
                    not INDEPENDENT_BROWNIAN
                ),

                "rho_YR_active": (
                    not INDEPENDENT_BROWNIAN
                ),

                # -------------------------------------------------
                # Stability diagnostics
                # -------------------------------------------------

                "mean_kappa_refit":
                    float(
                        np.mean(
                            kappa_refits
                        )
                    ),

                "min_kappa_refit":
                    float(
                        np.min(
                            kappa_refits
                        )
                    ),

                "mean_kappa_z_refit":
                    float(
                        np.mean(
                            kappa_z_refits
                        )
                    ),

                "min_kappa_z_refit":
                    float(
                        np.min(
                            kappa_z_refits
                        )
                    ),

                "mean_gamma_refit":
                    float(
                        np.mean(
                            gamma_refits
                        )
                    ),

                "mean_gamma_phi_refit":
                    float(
                        np.mean(
                            gamma_phi_refits
                        )
                    ),

                "max_gamma_phi_refit":
                    float(
                        np.max(
                            gamma_phi_refits
                        )
                    ),

                "mean_stability_margin_refit":
                    float(
                        np.mean(
                            stability_margin_refits
                        )
                    ),

                "min_stability_margin_refit":
                    float(
                        np.min(
                            stability_margin_refits
                        )
                    ),

                "stable_all_refits":
                    stable_all_refits,

                "mean_feedback_drift":
                    float(
                        np.mean(
                            feedback_history
                        )
                    ),

                # -------------------------------------------------
                # Experiment information
                # -------------------------------------------------

                "n_total":
                    n_total,

                "rolling_window_size":
                    rolling_window_size,

                "n_oos":
                    n_eval,

                "refit_every":
                    REFIT_EVERY,

                "n_epochs_initial":
                    N_EPOCHS_INITIAL,

                "n_epochs_refit":
                    N_EPOCHS_REFIT,

                # -------------------------------------------------
                # Forecast diagnostics
                # -------------------------------------------------

                "mean_predicted_variance":
                    float(
                        np.mean(
                            predicted_variance
                        )
                    ),

                "mean_oos_raw_RV":
                    float(
                        np.mean(
                            rv_test_eval
                        )
                    ),

                "min_predicted_variance":
                    float(
                        np.min(
                            predicted_variance
                        )
                    ),

                "max_predicted_variance":
                    float(
                        np.max(
                            predicted_variance
                        )
                    ),

                # -------------------------------------------------
                # Forecast metrics
                # -------------------------------------------------

                "QLIKE_raw":
                    qlike,

                "MSE_vol_raw":
                    mse_vol,

                "MSE_var_raw":
                    mse_var,

                "MAE_vol_raw":
                    mae_vol,

                "MAE_var_raw":
                    mae_var,

                "Return_NLL":
                    return_nll,

                "QLIKE_rv":
                    qlike_rv,

                "MSE_vol_rv":
                    mse_vol_rv,

                "MSE_var_rv":
                    mse_var_rv,

                "MAE_vol_rv":
                    mae_vol_rv,

                "MAE_var_rv":
                    mae_var_rv,
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

            error_message = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            print(
                f"\nFailed on {ticker}: "
                f"{error_message}"
            )

            if GAMMA_FIXED_ZERO:

                failed_model_name = (
                    "Naive-Gamma-zero-MR-z"
                )

                failed_feedback_specification = (
                    "gamma_zero"
                )

            elif INDEPENDENT_BROWNIAN:

                failed_model_name = (
                    "Naive-Independent-Brownian-MR-z"
                )

                failed_feedback_specification = (
                    FEEDBACK_SPECIFICATION
                )

            else:

                failed_model_name = (
                    "Naive-Full-SDE-MR-z"
                )

                failed_feedback_specification = (
                    FEEDBACK_SPECIFICATION
                )

            results.append(
                {
                    "ticker":
                        ticker,

                    "model":
                        failed_model_name,

                    "feedback_specification":
                        failed_feedback_specification,

                    "gamma_fixed_zero":
                        GAMMA_FIXED_ZERO,

                    "independent_brownian":
                        INDEPENDENT_BROWNIAN,

                    "error":
                        error_message,
                }
            )

            save_progress(
                results
            )

    # =========================================================
    # Final summary
    # =========================================================

    results_df = pd.DataFrame(
        results
    )

    if (
        "QLIKE_raw"
        in results_df.columns
    ):

        successful_results = (
            results_df[
                results_df[
                    "QLIKE_raw"
                ].notna()
            ]
            .copy()
        )

    else:

        successful_results = (
            pd.DataFrame()
        )

    if not successful_results.empty:

        successful_results = (
            successful_results
            .sort_values(
                "QLIKE_raw",
                ascending=True,
            )
            .reset_index(
                drop=True
            )
        )

        successful_results.to_csv(
            RESULTS_DIR
            / "rolling_50_50_results_sorted.csv",
            index=False,
        )

        print(
            "\n"
            + "=" * 120
        )

        print(
            "Final fixed 50:50 rolling-window results"
        )

        print(
            "=" * 120
        )

        columns_to_print = [
            "ticker",
            "model",
            "feedback_specification",

            "gamma_fixed_zero",
            "independent_brownian",

            "n_total",
            "rolling_window_size",
            "n_oos",
            "refit_every",

            "mean_kappa_refit",
            "mean_kappa_z_refit",
            "mean_gamma_refit",
            "mean_gamma_phi_refit",
            "min_stability_margin_refit",
            "stable_all_refits",
            "mean_feedback_drift",

            "QLIKE_raw",
            "MSE_vol_raw",
            "MSE_var_raw",
            "MAE_vol_raw",
            "MAE_var_raw",

            "QLIKE_rv",
            "MSE_vol_rv",
            "MSE_var_rv",
            "MAE_vol_rv",
            "MAE_var_rv",

            "Return_NLL",
        ]

        available_columns = [
            column
            for column
            in columns_to_print
            if column
            in successful_results.columns
        ]

        print(
            successful_results[
                available_columns
            ].to_string(
                index=False
            )
        )

    else:

        print(
            "\nNo successful rolling-window "
            "results were generated."
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()