import numpy as np
from scipy.optimize import minimize

from models.model_Realized_GARCH import get_rg_arrays_from_data_dict


# =========================================================
# Data helper
# =========================================================

def get_garch_arrays_from_data_dict(
    data_dict,
    ticker,
):

    data = data_dict[ticker]["train"]

    # -----------------------------------------------------
    # utils1: y_train / y_test are log-price levels
    # -----------------------------------------------------
    log_price_train = np.asarray(
        data["y_train"],
        dtype=np.float64,
    ).reshape(-1)

    log_price_test = np.asarray(
        data["y_test"],
        dtype=np.float64,
    ).reshape(-1)

    rv_train_original = np.asarray(
        data["rv_train"],
        dtype=np.float64,
    ).reshape(-1)

    rv_test_original = np.asarray(
        data["rv_test"],
        dtype=np.float64,
    ).reshape(-1)


    if len(log_price_train) != len(rv_train_original):
        raise ValueError(
            f"{ticker}: train log-price/RV length mismatch: "
            f"{len(log_price_train)} vs "
            f"{len(rv_train_original)}."
        )

    if len(log_price_test) != len(rv_test_original):
        raise ValueError(
            f"{ticker}: test log-price/RV length mismatch: "
            f"{len(log_price_test)} vs "
            f"{len(rv_test_original)}."
        )

    log_price_full = np.concatenate(
        [
            log_price_train,
            log_price_test,
        ]
    )

    rv_full_original = np.concatenate(
        [
            rv_train_original,
            rv_test_original,
        ]
    )

    returns_full = (
        100.0
        * np.diff(
            log_price_full
        )
    )

    rv_full = (
        rv_full_original[
            1:
        ]
    )

    if len(returns_full) != len(rv_full):
        raise RuntimeError(
            f"{ticker}: GARCH return/RV alignment failed: "
            f"{len(returns_full)} vs {len(rv_full)}."
        )


    valid = (
        np.isfinite(returns_full)
        & np.isfinite(rv_full)
    )

    returns_full = returns_full[
        valid
    ]

    rv_full = rv_full[
        valid
    ]

    rv_full = np.maximum(
        rv_full,
        1e-12,
    )


    original_train_n = len(
        log_price_train
    )

    return_split = (
        original_train_n
        - 1
    )

    y_train = returns_full[
        :return_split
    ]

    rv_train = rv_full[
        :return_split
    ]

    y_test = returns_full[
        return_split:
    ]

    rv_test = rv_full[
        return_split:
    ]


    if len(y_train) != len(rv_train):
        raise RuntimeError(
            f"{ticker}: GARCH train return/RV "
            "length mismatch."
        )

    if len(y_test) != len(rv_test):
        raise RuntimeError(
            f"{ticker}: GARCH test return/RV "
            "length mismatch."
        )

    if len(y_train) < 10:
        raise ValueError(
            f"{ticker}: insufficient GARCH "
            "training observations."
        )


    print(
        f"\n{ticker}: GARCH data construction"
    )

    print(
        f"Original log-price N : "
        f"{len(log_price_full)}"
    )

    print(
        f"Return/RV N          : "
        f"{len(returns_full)}"
    )

    print(
        "Mean return squared  : "
        f"{np.mean(returns_full ** 2):.8f}"
    )

    print(
        "Mean raw RV          : "
        f"{np.mean(rv_full):.8f}"
    )


    return (
        y_train,
        rv_train,
        y_test,
        rv_test,
    )

# =========================================================
# Standard GARCH(1,1)
#
# y_t | F_{t-1} ~ N(0, h_t)
# h_t = omega + alpha y_{t-1}^2 + beta h_{t-1}
# =========================================================

def garch11_filter(y, omega, alpha, beta, h0=None):
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    T = len(y)
    h = np.empty(T, dtype=np.float64)

    if h0 is None:
        h_prev = max(np.var(y), 1e-12)
    else:
        h_prev = max(float(h0), 1e-12)

    y_prev_sq = h_prev

    for t in range(T):
        h_t = omega + alpha * y_prev_sq + beta * h_prev
        h_t = max(float(h_t), 1e-12)

        h[t] = h_t

        h_prev = h_t
        y_prev_sq = y[t] ** 2

    return h


def garch11_negloglik(y, theta):
    omega, alpha, beta = theta

    if omega <= 0 or alpha < 0 or beta < 0:
        return 1e50

    if alpha + beta >= 0.999:
        return 1e50

    h = garch11_filter(y, omega, alpha, beta)

    if np.any(h <= 0) or np.any(~np.isfinite(h)):
        return 1e50

    log2pi = np.log(2.0 * np.pi)

    nll = 0.5 * np.sum(
        log2pi + np.log(h) + (y ** 2) / h
    )

    return float(nll)


def fit_garch11(
    y_train,
    initial_theta=None,
):
    
    y_train = np.asarray(
        y_train,
        dtype=np.float64,
    ).reshape(-1)

    y_train = y_train[
        np.isfinite(y_train)
    ]

    if len(y_train) < 10:
        raise ValueError(
            "At least 10 observations are required "
            "for GARCH estimation."
        )

    var_y = max(
        float(np.var(y_train)),
        1e-8,
    )

    fresh_starts = [
        np.array(
            [0.05 * var_y, 0.05, 0.90],
            dtype=np.float64,
        ),
        np.array(
            [0.10 * var_y, 0.10, 0.80],
            dtype=np.float64,
        ),
        np.array(
            [0.05 * var_y, 0.15, 0.75],
            dtype=np.float64,
        ),
        np.array(
            [0.20 * var_y, 0.20, 0.60],
            dtype=np.float64,
        ),
    ]

    starts = list(
        fresh_starts
    )

    if initial_theta is not None:
        previous_theta = np.asarray(
            initial_theta,
            dtype=np.float64,
        ).reshape(-1)

        previous_valid = (
            previous_theta.shape == (3,)
            and np.all(np.isfinite(previous_theta))
            and previous_theta[0] > 0.0
            and previous_theta[1] >= 0.0
            and previous_theta[2] >= 0.0
            and (
                previous_theta[1]
                + previous_theta[2]
                < 0.999
            )
        )

        if previous_valid:
            starts.insert(
                0,
                previous_theta.copy(),
            )

    bounds = [
        (1e-12, None),
        (0.0, 0.999),
        (0.0, 0.999),
    ]

    def objective(theta):
        return garch11_negloglik(
            y_train,
            theta,
        )

    candidate_results = []

    for start_index, x0 in enumerate(
        starts
    ):
        result = minimize(
            objective,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 20000,
                "ftol": 1e-12,
                "gtol": 1e-8,
                "maxls": 50,
            },
        )

        theta_candidate = np.asarray(
            result.x,
            dtype=np.float64,
        ).reshape(-1)

        candidate_nll = objective(
            theta_candidate
        )

        candidate_valid = (
            theta_candidate.shape == (3,)
            and np.all(
                np.isfinite(
                    theta_candidate
                )
            )
            and np.isfinite(
                candidate_nll
            )
            and candidate_nll < 1e49
            and theta_candidate[0] > 0.0
            and theta_candidate[1] >= 0.0
            and theta_candidate[2] >= 0.0
            and (
                theta_candidate[1]
                + theta_candidate[2]
                < 0.999
            )
        )

        if candidate_valid:
            candidate_results.append(
                {
                    "result": result,
                    "theta_hat": theta_candidate,
                    "nll": float(
                        candidate_nll
                    ),
                    "selected_start":
                        start_index,
                }
            )

    if not candidate_results:
        raise RuntimeError(
            "GARCH(1,1) estimation failed to "
            "find a valid parameter vector."
        )

    best = min(
        candidate_results,
        key=lambda item: item["nll"],
    )

    best_result = best[
        "result"
    ]

    theta_hat = best[
        "theta_hat"
    ]

    h_train = garch11_filter(
        y=y_train,
        omega=float(theta_hat[0]),
        alpha=float(theta_hat[1]),
        beta=float(theta_hat[2]),
    )

    return {
        "result": best_result,
        "theta_hat": theta_hat,
        "omega": float(
            theta_hat[0]
        ),
        "alpha": float(
            theta_hat[1]
        ),
        "beta": float(
            theta_hat[2]
        ),
        "persistence": float(
            theta_hat[1]
            + theta_hat[2]
        ),
        "h_train": h_train,
        "nll": float(
            best["nll"]
        ),
        "success": bool(
            best_result.success
        ),
        "message": str(
            best_result.message
        ),
        "nit": int(
            getattr(
                best_result,
                "nit",
                -1,
            )
        ),
        "nfev": int(
            getattr(
                best_result,
                "nfev",
                -1,
            )
        ),
        "selected_start": int(
            best["selected_start"]
        ),
    }


def garch11_forecast_1step(y_train, y_test, theta_hat):
    omega, alpha, beta = map(float, theta_hat)

    h_train = garch11_filter(y_train, omega, alpha, beta)

    h_prev = float(h_train[-1])
    y_prev_sq = float(y_train[-1] ** 2)

    h_pred = np.empty(len(y_test), dtype=np.float64)

    for t in range(len(y_test)):
        h_t = omega + alpha * y_prev_sq + beta * h_prev
        h_t = max(float(h_t), 1e-12)

        h_pred[t] = h_t

        h_prev = h_t
        y_prev_sq = float(y_test[t] ** 2)

    return h_pred

def fixed_half_sample_rolling_garch11(
    ticker: str,
    y_full: np.ndarray,
    rv_full: np.ndarray,
    refit_every: int = 21,
    warm_start: bool = True,
):
    y_full = np.asarray(
        y_full,
        dtype=np.float64,
    ).reshape(-1)

    rv_full = np.maximum(
        np.asarray(
            rv_full,
            dtype=np.float64,
        ).reshape(-1),
        1e-12,
    )

    if len(y_full) != len(rv_full):
        raise ValueError(
            "y_full and rv_full must have equal length."
        )

    valid = (
        np.isfinite(y_full)
        & np.isfinite(rv_full)
    )

    y_full = y_full[
        valid
    ]

    rv_full = rv_full[
        valid
    ]

    if len(y_full) < 20:
        raise ValueError(
            "The full sample is too short."
        )

    if refit_every < 1:
        raise ValueError(
            "refit_every must be at least 1."
        )

    n_total = len(
        y_full
    )

    window_size = (
        n_total // 2
    )

    first_oos_index = (
        window_size
    )

    n_oos = (
        n_total
        - first_oos_index
    )

    predicted_variance = []
    actual_rv = []
    actual_returns = []
    parameter_history = []

    fitted_theta = None

    active_window_start = None
    active_window_end = None

    h_prev = None
    y_prev_sq = None

    refit_count = 0

    print(
        "\n" + "=" * 90
    )

    print(
        f"{ticker}: block rolling GARCH(1,1)"
    )

    print(
        "=" * 90
    )

    print(
        f"Total observations       : {n_total}"
    )

    print(
        f"Estimation-window size   : {window_size}"
    )

    print(
        f"Out-of-sample forecasts  : {n_oos}"
    )

    print(
        "Window/parameter update : "
        f"every {refit_every} trading days"
    )

    print(
        "Forecast frequency       : daily"
    )

    for forecast_number, target_idx in enumerate(
        range(
            first_oos_index,
            n_total,
        ),
        start=1,
    ):
        should_refit = (
            fitted_theta is None
            or (
                forecast_number - 1
            ) % refit_every == 0
        )

        refit_accepted = False
        optimizer_success = False
        optimizer_message = ""
        optimizer_nit = -1
        optimizer_nfev = -1
        selected_start = -1
        candidate_nll = np.nan
        candidate_persistence = np.nan

        if should_refit:
            refit_count += 1

            # The current target is excluded from estimation.
            active_window_end = (
                target_idx
            )

            active_window_start = (
                active_window_end
                - window_size
            )

            if active_window_start < 0:
                raise RuntimeError(
                    "The active estimation window "
                    "begins before the sample."
                )

            y_window = y_full[
                active_window_start:
                active_window_end
            ]

            if len(y_window) != window_size:
                raise RuntimeError(
                    "Unexpected GARCH estimation-window "
                    f"length: {len(y_window)} instead "
                    f"of {window_size}."
                )

            previous_theta = (
                None
                if fitted_theta is None
                else fitted_theta.copy()
            )

            try:
                fit_result = fit_garch11(
                    y_train=y_window,
                    initial_theta=(
                        previous_theta
                        if warm_start
                        else None
                    ),
                )

                candidate_theta = np.asarray(
                    fit_result[
                        "theta_hat"
                    ],
                    dtype=np.float64,
                ).reshape(-1)

                candidate_nll = (
                    garch11_negloglik(
                        y_window,
                        candidate_theta,
                    )
                )

                omega_candidate = float(
                    candidate_theta[0]
                )

                alpha_candidate = float(
                    candidate_theta[1]
                )

                beta_candidate = float(
                    candidate_theta[2]
                )

                candidate_persistence = (
                    alpha_candidate
                    + beta_candidate
                )

                optimizer_success = bool(
                    fit_result.get(
                        "success",
                        False,
                    )
                )

                optimizer_message = str(
                    fit_result.get(
                        "message",
                        "",
                    )
                )

                optimizer_nit = int(
                    fit_result.get(
                        "nit",
                        -1,
                    )
                )

                optimizer_nfev = int(
                    fit_result.get(
                        "nfev",
                        -1,
                    )
                )

                selected_start = int(
                    fit_result.get(
                        "selected_start",
                        -1,
                    )
                )

                candidate_valid = (
                    candidate_theta.shape == (3,)
                    and np.all(
                        np.isfinite(
                            candidate_theta
                        )
                    )
                    and np.isfinite(
                        candidate_nll
                    )
                    and candidate_nll < 1e49
                    and omega_candidate > 0.0
                    and alpha_candidate >= 0.0
                    and beta_candidate >= 0.0
                    and candidate_persistence < 0.999
                )

            except Exception as exc:
                candidate_valid = False
                candidate_theta = None

                optimizer_message = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

            if candidate_valid:
                fitted_theta = (
                    candidate_theta
                )

                refit_accepted = True

            elif previous_theta is not None:
                fitted_theta = (
                    previous_theta
                )

                refit_accepted = False

                print(
                    f"Rejected GARCH refit at "
                    f"forecast {forecast_number}; "
                    "retained previous parameters."
                )

            else:
                raise RuntimeError(
                    f"Initial GARCH fit failed for "
                    f"{ticker}: "
                    f"{optimizer_message}"
                )

            omega, alpha, beta = map(
                float,
                fitted_theta,
            )

           
            h_window = garch11_filter(
                y=y_window,
                omega=omega,
                alpha=alpha,
                beta=beta,
            )

            if (
                len(h_window) == 0
                or not np.isfinite(
                    h_window[-1]
                )
                or h_window[-1] <= 0.0
            ):
                raise RuntimeError(
                    "Invalid GARCH variance at "
                    "the end of the active window."
                )

            h_prev = float(
                h_window[-1]
            )

            y_prev_sq = float(
                y_window[-1] ** 2
            )

            print(
                f"\nRefit #{refit_count} "
                f"at forecast "
                f"{forecast_number} | "
                f"window=[{active_window_start}, "
                f"{active_window_end - 1}] | "
                f"success={optimizer_success} | "
                f"accepted={refit_accepted} | "
                f"nit={optimizer_nit} | "
                f"start={selected_start} | "
                f"train_NLL={candidate_nll:.6f} | "
                f"persistence="
                f"{candidate_persistence:.6f}"
            )

        if (
            fitted_theta is None
            or h_prev is None
            or y_prev_sq is None
            or active_window_start is None
            or active_window_end is None
        ):
            raise RuntimeError(
                "The GARCH rolling state has "
                "not been initialized."
            )

        omega, alpha, beta = map(
            float,
            fitted_theta,
        )

        # -------------------------------------------------
        # Daily one-step-ahead forecast
        # -------------------------------------------------

        h_next = (
            omega
            + alpha * y_prev_sq
            + beta * h_prev
        )

        h_next = max(
            float(h_next),
            1e-12,
        )

        actual_return_value = float(
            y_full[target_idx]
        )

        actual_rv_value = float(
            rv_full[target_idx]
        )

        predicted_variance.append(
            h_next
        )

        actual_returns.append(
            actual_return_value
        )

        actual_rv.append(
            actual_rv_value
        )

        parameter_history.append(
            {
                "forecast_number":
                    forecast_number,
                "target_index":
                    target_idx,

                "window_start":
                    active_window_start,
                "window_end":
                    active_window_end - 1,

                "refit":
                    should_refit,
                "refit_accepted":
                    refit_accepted,
                "optimizer_success":
                    optimizer_success,
                "optimizer_message":
                    optimizer_message,
                "optimizer_nit":
                    optimizer_nit,
                "optimizer_nfev":
                    optimizer_nfev,
                "selected_start":
                    selected_start,
                "candidate_nll":
                    candidate_nll,
                "candidate_persistence":
                    candidate_persistence,

                "omega":
                    omega,
                "alpha":
                    alpha,
                "beta":
                    beta,
                "persistence":
                    alpha + beta,

                "h_origin":
                    h_prev,
                "return_squared_origin":
                    y_prev_sq,
                "predicted_variance":
                    h_next,
            }
        )

        print(
            f"{ticker} forecast "
            f"{forecast_number}/{n_oos} | "
            f"estimation_window="
            f"[{active_window_start}, "
            f"{active_window_end - 1}] | "
            f"target={target_idx} | "
            f"refit={should_refit} | "
            f"h={h_next:.6f}"
        )


        h_prev = h_next

        y_prev_sq = (
            actual_return_value ** 2
        )

    return {
        "predicted_variance": np.asarray(
            predicted_variance,
            dtype=np.float64,
        ),
        "actual_rv": np.asarray(
            actual_rv,
            dtype=np.float64,
        ),
        "actual_returns": np.asarray(
            actual_returns,
            dtype=np.float64,
        ),
        "parameter_history":
            parameter_history,
        "n_total":
            n_total,
        "rolling_window_size":
            window_size,
        "n_oos":
            n_oos,
        "refit_every":
            refit_every,
        "window_step":
            refit_every,
    }

# =========================================================
# Evaluation metrics
# =========================================================

def qlike_variance(
    h_pred,
    rv_true,
    eps=1e-8,
):
    h = np.maximum(
        np.asarray(
            h_pred,
            dtype=np.float64,
        ).reshape(-1),
        eps,
    )

    rv = np.maximum(
        np.asarray(
            rv_true,
            dtype=np.float64,
        ).reshape(-1),
        eps,
    )

    if len(h) != len(rv):
        raise ValueError(
            "h_pred and rv_true must have equal length."
        )

    valid = (
        np.isfinite(h)
        & np.isfinite(rv)
    )

    h = h[
        valid
    ]

    rv = rv[
        valid
    ]

    if len(h) == 0:
        return np.nan

    return float(
        np.mean(
            np.log(h)
            + rv / h
        )
    )

def mae_volatility(
    h_pred,
    rv_true,
    eps=1e-8,
):
    h = np.maximum(
        np.asarray(
            h_pred,
            dtype=np.float64,
        ).reshape(-1),
        eps,
    )

    rv = np.maximum(
        np.asarray(
            rv_true,
            dtype=np.float64,
        ).reshape(-1),
        eps,
    )

    if len(h) != len(rv):
        raise ValueError(
            "h_pred and rv_true must have equal length."
        )

    valid = (
        np.isfinite(h)
        & np.isfinite(rv)
    )

    h = h[
        valid
    ]

    rv = rv[
        valid
    ]

    if len(h) == 0:
        return np.nan

    return float(
        np.mean(
            np.abs(
                np.sqrt(rv)
                - np.sqrt(h)
            )
        )
    )


def mae_variance(
    h_pred,
    rv_true,
    eps=1e-8,
):
    h = np.maximum(
        np.asarray(
            h_pred,
            dtype=np.float64,
        ).reshape(-1),
        eps,
    )

    rv = np.maximum(
        np.asarray(
            rv_true,
            dtype=np.float64,
        ).reshape(-1),
        eps,
    )

    if len(h) != len(rv):
        raise ValueError(
            "h_pred and rv_true must have equal length."
        )

    valid = (
        np.isfinite(h)
        & np.isfinite(rv)
    )

    h = h[
        valid
    ]

    rv = rv[
        valid
    ]

    if len(h) == 0:
        return np.nan

    return float(
        np.mean(
            np.abs(
                rv - h
            )
        )
    )


def return_nll(
    h_pred,
    y_true,
    eps=1e-8,
):
    h = np.maximum(
        np.asarray(
            h_pred,
            dtype=np.float64,
        ).reshape(-1),
        eps,
    )

    y = np.asarray(
        y_true,
        dtype=np.float64,
    ).reshape(-1)

    if len(h) != len(y):
        raise ValueError(
            "h_pred and y_true must have equal length."
        )

    valid = (
        np.isfinite(h)
        & np.isfinite(y)
    )

    h = h[
        valid
    ]

    y = y[
        valid
    ]

    if len(h) == 0:
        return np.nan

    nll = 0.5 * (
        np.log(
            2.0 * np.pi
        )
        + np.log(h)
        + y ** 2 / h
    )

    return float(
        np.mean(nll)
    )


def mse_volatility(h_pred, rv_true, eps=1e-8):
    h_pred = np.asarray(h_pred, dtype=np.float64).reshape(-1)
    rv_true = np.asarray(rv_true, dtype=np.float64).reshape(-1)

    mask = np.isfinite(h_pred) & np.isfinite(rv_true)
    h_pred = h_pred[mask]
    rv_true = rv_true[mask]

    if len(h_pred) == 0:
        return np.nan

    h = np.maximum(h_pred, eps)
    rv = np.maximum(rv_true, eps)

    return float(np.mean((np.sqrt(rv) - np.sqrt(h)) ** 2))


def mse_variance(h_pred, rv_true, eps=1e-8):
    h_pred = np.asarray(h_pred, dtype=np.float64).reshape(-1)
    rv_true = np.asarray(rv_true, dtype=np.float64).reshape(-1)

    mask = np.isfinite(h_pred) & np.isfinite(rv_true)
    h_pred = h_pred[mask]
    rv_true = rv_true[mask]

    if len(h_pred) == 0:
        return np.nan

    h = np.maximum(h_pred, eps)
    rv = np.maximum(rv_true, eps)

    return float(np.mean((rv - h) ** 2))

def summarize_rolling_garch11(
    rolling_result,
):
    h_pred = np.asarray(
        rolling_result[
            "predicted_variance"
        ],
        dtype=np.float64,
    ).reshape(-1)

    rv_true = np.asarray(
        rolling_result[
            "actual_rv"
        ],
        dtype=np.float64,
    ).reshape(-1)

    returns_true = np.asarray(
        rolling_result[
            "actual_returns"
        ],
        dtype=np.float64,
    ).reshape(-1)

    if not (
        len(h_pred)
        == len(rv_true)
        == len(returns_true)
    ):
        raise ValueError(
            "GARCH rolling forecasts and "
            "targets are not aligned."
        )

    return {
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
        "window_step": int(
            rolling_result[
                "window_step"
            ]
        ),
        "QLIKE_raw": qlike_variance(
            h_pred,
            rv_true,
        ),
        "MSE_vol_raw": mse_volatility(
            h_pred,
            rv_true,
        ),
        "MSE_var_raw": mse_variance(
            h_pred,
            rv_true,
        ),
        "MAE_vol_raw": mae_volatility(
            h_pred,
            rv_true,
        ),
        "MAE_var_raw": mae_variance(
            h_pred,
            rv_true,
        ),
        "Return_NLL": return_nll(
            h_pred,
            returns_true,
        ),
    }

def run_rolling_garch11_for_ticker(
    data_dict,
    ticker,
    refit_every=21,
    warm_start=True,
):
    """
    Run a fixed 50:50 monthly block rolling GARCH(1,1)
    experiment for one ticker.
    """
    (
        y_train_old,
        rv_train_old,
        y_test_old,
        rv_test_old,
    ) = get_garch_arrays_from_data_dict(
        data_dict,
        ticker,
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

    rolling_result = (
        fixed_half_sample_rolling_garch11(
            ticker=ticker,
            y_full=y_full,
            rv_full=rv_full,
            refit_every=refit_every,
            warm_start=warm_start,
        )
    )

    metrics = {
        "ticker": ticker,
        **summarize_rolling_garch11(
            rolling_result
        ),
    }

    return {
        "rolling_result":
            rolling_result,
        "h_pred":
            rolling_result[
                "predicted_variance"
            ],
        "rv_oos":
            rolling_result[
                "actual_rv"
            ],
        "y_oos":
            rolling_result[
                "actual_returns"
            ],
        "parameter_history":
            rolling_result[
                "parameter_history"
            ],
        "metrics":
            metrics,
    }