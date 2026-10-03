import numpy as np
from scipy.optimize import minimize


# =========================================================
# Data helper
# =========================================================

def get_heavy_arrays_from_data_dict(
    data_dict,
    ticker,
):
    """
    Construct HEAVY return/RV arrays from the SAME raw data
    used by the SDE.

    Important
    ---------
    utils1 stores:

        y_t = log price

    NO additional scaling factor c is used.

    In particular, we DO NOT use:

        c = mean(r^2) / mean(RV)

    or:

        RV <- c * RV

    Alignment
    ---------
        r_t corresponds to RV_t.

    Therefore, after differencing log prices:

        return_full = 100 * diff(log_price_full)
        rv_full     = rv_full[1:]
    """

    data = data_dict[ticker]["train"]

    # =====================================================
    # utils1 stores log-price levels
    # =====================================================

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
            f"{ticker}: original train log-price/RV "
            f"length mismatch: "
            f"{len(log_price_train)} vs "
            f"{len(rv_train_original)}."
        )

    if len(log_price_test) != len(rv_test_original):
        raise ValueError(
            f"{ticker}: original test log-price/RV "
            f"length mismatch: "
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
            f"{ticker}: HEAVY full return/RV "
            f"alignment failed: "
            f"{len(returns_full)} vs "
            f"{len(rv_full)}."
        )


    valid = (
        np.isfinite(returns_full)
        & np.isfinite(rv_full)
    )

    returns_full = (
        returns_full[
            valid
        ]
    )

    rv_full = (
        rv_full[
            valid
        ]
    )

    rv_full = np.maximum(
        rv_full,
        1e-12,
    )


    original_train_price_n = len(
        log_price_train
    )

    return_split = (
        original_train_price_n
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
            f"{ticker}: HEAVY training return/RV "
            "length mismatch."
        )

    if len(y_test) != len(rv_test):
        raise RuntimeError(
            f"{ticker}: HEAVY test return/RV "
            "length mismatch."
        )

    if len(y_train) < 10:
        raise ValueError(
            f"{ticker}: insufficient HEAVY "
            "training observations."
        )

    # =====================================================
    # Diagnostics
    # =====================================================

    print(
        f"\n{ticker}: HEAVY data construction"
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
# HEAVY-r model
#
# y_t | F_{t-1} ~ N(0, h_t)
#
# h_t = omega + alpha * RV_{t-1} + beta * h_{t-1}
#
# h_t:
#     close-to-close conditional return variance
#
# RV_{t-1}:
#     lagged observed realized variance
#
# For one-step forecasting, the HEAVY-RM equation is not needed.
# =========================================================

def heavy_filter(
    y,
    rv,
    omega,
    alpha,
    beta,
    h0=None,
    rv0=None,
):
    """
    Filter the in-sample HEAVY-r conditional variance.

    The output h[t] is aligned with y[t].

    For t >= 1:

        h[t] = omega + alpha * rv[t - 1] + beta * h[t - 1]

    For the first observation, an initialization value rv0 is used.
    """
    y = np.asarray(
        y,
        dtype=np.float64,
    ).reshape(-1)

    rv = np.asarray(
        rv,
        dtype=np.float64,
    ).reshape(-1)

    if len(y) != len(rv):
        raise ValueError(
            "y and rv must have the same number of observations."
        )

    if len(y) == 0:
        raise ValueError("y and rv must not be empty.")

    T = len(y)
    h = np.empty(T, dtype=np.float64)

    if h0 is None:
        h_prev = max(
            float(np.mean(y ** 2)),
            1e-12,
        )
    else:
        h_prev = max(
            float(h0),
            1e-12,
        )

    if rv0 is None:
        rv_prev = max(
            float(np.mean(rv)),
            1e-12,
        )
    else:
        rv_prev = max(
            float(rv0),
            1e-12,
        )

    for t in range(T):
        h_t = (
            omega
            + alpha * rv_prev
            + beta * h_prev
        )

        h_t = max(
            float(h_t),
            1e-12,
        )

        h[t] = h_t
        h_prev = h_t
        rv_prev = float(rv[t])

    return h


# =========================================================
# Gaussian quasi-negative log-likelihood
# =========================================================

def heavy_negloglik(y, rv, theta):
    """
    Gaussian quasi-negative log-likelihood for the HEAVY-r model.

    The Gaussian assumption is used as a quasi-likelihood:

        NLL_t = 0.5 * [
            log(2*pi)
            + log(h_t)
            + y_t^2 / h_t
        ]
    """
    omega, alpha, beta = theta

    if omega <= 0.0:
        return 1e50

    if alpha < 0.0:
        return 1e50

    if beta < 0.0 or beta >= 0.999:
        return 1e50

    h = heavy_filter(
        y=y,
        rv=rv,
        omega=omega,
        alpha=alpha,
        beta=beta,
    )

    if np.any(h <= 0.0):
        return 1e50

    if np.any(~np.isfinite(h)):
        return 1e50

    log2pi = np.log(2.0 * np.pi)

    nll = 0.5 * np.sum(
        log2pi
        + np.log(h)
        + (y ** 2) / h
    )

    if not np.isfinite(nll):
        return 1e50

    return float(nll)



def fit_heavy(
    y_train,
    rv_train,
):
    """
    Estimate HEAVY-r by Gaussian QMLE using one
    predetermined starting value.
    """

    y_train = np.asarray(
        y_train,
        dtype=np.float64,
    ).reshape(-1)

    rv_train = np.maximum(
        np.asarray(
            rv_train,
            dtype=np.float64,
        ).reshape(-1),
        1e-12,
    )

    if len(y_train) != len(rv_train):
        raise ValueError(
            "y_train and rv_train must "
            "have the same length."
        )

    valid = (
        np.isfinite(y_train)
        & np.isfinite(rv_train)
    )

    y_train = y_train[valid]
    rv_train = rv_train[valid]

    if len(y_train) < 10:
        raise ValueError(
            "Too few training observations "
            "for HEAVY estimation."
        )

    mean_y2 = max(
        float(np.mean(y_train ** 2)),
        1e-8,
    )

    mean_rv = max(
        float(np.mean(rv_train)),
        1e-8,
    )

    # -----------------------------------------
    # Single fixed starting value
    # -----------------------------------------

    alpha0 = 0.10
    beta0 = 0.80

    omega0 = (
        (1.0 - beta0)
        * mean_y2
        - alpha0
        * mean_rv
    )

    x0 = np.array(
        [
            max(
                float(omega0),
                1e-6,
            ),
            alpha0,
            beta0,
        ],
        dtype=np.float64,
    )

    bounds = [
        (1e-12, None),
        (0.0, 10.0),
        (0.0, 0.999),
    ]

    def objective(theta):
        return heavy_negloglik(
            y=y_train,
            rv=rv_train,
            theta=theta,
        )

    result = minimize(
        objective,
        x0=x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": 2000,
            "ftol": 1e-10,
            "gtol": 1e-8,
            "maxls": 50,
        },
    )

    theta_hat = np.asarray(
        result.x,
        dtype=np.float64,
    ).reshape(-1)

    nll = objective(
        theta_hat
    )

    valid_solution = (
        theta_hat.shape == (3,)
        and np.all(
            np.isfinite(theta_hat)
        )
        and np.isfinite(nll)
        and nll < 1e49
        and theta_hat[0] > 0.0
        and theta_hat[1] >= 0.0
        and 0.0 <= theta_hat[2] < 0.999
    )

    if not valid_solution:
        raise RuntimeError(
            "HEAVY estimation failed."
        )

    h_train = heavy_filter(
        y=y_train,
        rv=rv_train,
        omega=float(
            theta_hat[0]
        ),
        alpha=float(
            theta_hat[1]
        ),
        beta=float(
            theta_hat[2]
        ),
    )

    return {
        "result":
            result,

        "theta_hat":
            theta_hat,

        "omega":
            float(
                theta_hat[0]
            ),

        "alpha":
            float(
                theta_hat[1]
            ),

        "beta":
            float(
                theta_hat[2]
            ),

        "h_train":
            h_train,

        "nll":
            float(
                nll
            ),

        "success":
            bool(
                result.success
            ),

        "message":
            str(
                result.message
            ),

        "nit":
            int(
                getattr(
                    result,
                    "nit",
                    -1,
                )
            ),

        "nfev":
            int(
                getattr(
                    result,
                    "nfev",
                    -1,
                )
            ),

        "selected_start":
            0,
    }

# =========================================================
# Rolling one-step-ahead forecasting
# =========================================================

def heavy_forecast_1step(
    y_train,
    rv_train,
    y_test,
    rv_test,
    theta_hat,
):
    """
    Produce rolling one-step-ahead HEAVY forecasts.

    First test forecast:

        h_{T+1|T}
        = omega
        + alpha * RV_T
        + beta * h_T

    Later forecasts:

        h_{T+j+1|T+j}
        = omega
        + alpha * RV_{T+j}
        + beta * h_{T+j}

    h_pred[t] is aligned with:

        y_test[t]
        rv_test[t]

    y_test is included for interface consistency with GARCH,
    but test returns do not enter the HEAVY recursion.
    """
    y_train = np.asarray(
        y_train,
        dtype=np.float64,
    ).reshape(-1)

    rv_train = np.asarray(
        rv_train,
        dtype=np.float64,
    ).reshape(-1)

    y_test = np.asarray(
        y_test,
        dtype=np.float64,
    ).reshape(-1)

    rv_test = np.asarray(
        rv_test,
        dtype=np.float64,
    ).reshape(-1)

    if len(y_train) != len(rv_train):
        raise ValueError(
            "y_train and rv_train must have the same length."
        )

    if len(y_test) != len(rv_test):
        raise ValueError(
            "y_test and rv_test must have the same length."
        )

    if len(y_train) == 0:
        raise ValueError(
            "Training arrays must not be empty."
        )

    omega, alpha, beta = map(
        float,
        theta_hat,
    )

    h_train = heavy_filter(
        y=y_train,
        rv=rv_train,
        omega=omega,
        alpha=alpha,
        beta=beta,
    )

    h_prev = float(h_train[-1])
    rv_prev = float(rv_train[-1])

    h_pred = np.empty(
        len(y_test),
        dtype=np.float64,
    )

    for t in range(len(y_test)):
        
        h_t = (
            omega
            + alpha * rv_prev
            + beta * h_prev
        )

        h_t = max(
            float(h_t),
            1e-12,
        )

        h_pred[t] = h_t

        
        h_prev = h_t
        rv_prev = float(rv_test[t])

    return h_pred

def fixed_half_sample_rolling_heavy(
    ticker: str,
    y_full: np.ndarray,
    rv_full: np.ndarray,
    refit_every: int = 21,
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

    y_full = y_full[valid]
    rv_full = rv_full[valid]

    if len(y_full) < 20:
        raise ValueError(
            "The complete sample is too short."
        )

    if refit_every < 1:
        raise ValueError(
            "refit_every must be at least 1."
        )

    n_total = len(y_full)
    window_size = n_total // 2
    first_oos_index = window_size
    n_oos = n_total - first_oos_index

    predicted_variance = []
    actual_rv = []
    actual_returns = []
    parameter_history = []

    fitted_theta = None

    active_window_start = None
    active_window_end = None

    h_prev = None
    rv_prev = None

    refit_count = 0

    print("\n" + "=" * 90)
    print(f"{ticker}: block rolling HEAVY-r")
    print("=" * 90)
    print(f"Total observations       : {n_total}")
    print(f"Estimation-window size   : {window_size}")
    print(f"Out-of-sample forecasts  : {n_oos}")
    print(
        "Window/parameter update : "
        f"every {refit_every} trading days"
    )
    print("Forecast frequency       : daily")

    for forecast_number, target_idx in enumerate(
        range(first_oos_index, n_total),
        start=1,
    ):
        should_refit = (
            fitted_theta is None
            or (forecast_number - 1) % refit_every == 0
        )

        refit_accepted = False
        optimizer_success = False
        optimizer_message = ""
        optimizer_nit = -1
        optimizer_nfev = -1
        selected_start = -1
        candidate_nll = np.nan

        if should_refit:
            refit_count += 1

            active_window_end = target_idx
            active_window_start = (
                active_window_end - window_size
            )

            if active_window_start < 0:
                raise RuntimeError(
                    "The active estimation window begins "
                    "before the sample."
                )

            y_window = y_full[
                active_window_start:
                active_window_end
            ]

            rv_window = rv_full[
                active_window_start:
                active_window_end
            ]

            if len(y_window) != window_size:
                raise RuntimeError(
                    "Unexpected HEAVY window length: "
                    f"{len(y_window)} instead of "
                    f"{window_size}."
                )

            if len(y_window) != len(rv_window):
                raise RuntimeError(
                    "Return and RV windows are not aligned."
                )

            previous_theta = (
                None
                if fitted_theta is None
                else fitted_theta.copy()
            )

            try:
                fit_result = fit_heavy(
                    y_train=y_window,
                    rv_train=rv_window,
                )

                candidate_theta = np.asarray(
                    fit_result["theta_hat"],
                    dtype=np.float64,
                ).reshape(-1)

                candidate_nll = heavy_negloglik(
                    y=y_window,
                    rv=rv_window,
                    theta=candidate_theta,
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

                omega_candidate = float(
                    candidate_theta[0]
                )

                alpha_candidate = float(
                    candidate_theta[1]
                )

                beta_candidate = float(
                    candidate_theta[2]
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
                    and 0.0 <= beta_candidate < 0.999
                )

            except Exception as exc:
                candidate_valid = False
                candidate_theta = None
                optimizer_message = (
                    f"{type(exc).__name__}: {exc}"
                )

            if candidate_valid:
                fitted_theta = candidate_theta
                refit_accepted = True

            elif previous_theta is not None:
                fitted_theta = previous_theta
                refit_accepted = False

                print(
                    f"Rejected HEAVY refit at forecast "
                    f"{forecast_number}; retained previous "
                    "parameters."
                )

            else:
                raise RuntimeError(
                    f"Initial HEAVY fit failed for {ticker}: "
                    f"{optimizer_message}"
                )

            omega, alpha, beta = map(
                float,
                fitted_theta,
            )

            h_window = heavy_filter(
                y=y_window,
                rv=rv_window,
                omega=omega,
                alpha=alpha,
                beta=beta,
            )

            if (
                len(h_window) == 0
                or not np.isfinite(h_window[-1])
                or h_window[-1] <= 0.0
            ):
                raise RuntimeError(
                    "Invalid HEAVY variance at the end "
                    "of the active window."
                )
            h_prev = float(
                h_window[-1]
            )

            rv_prev = float(
                rv_window[-1]
            )

            print(
                f"\nRefit #{refit_count} "
                f"at forecast {forecast_number} | "
                f"window=[{active_window_start}, "
                f"{active_window_end - 1}] | "
                f"success={optimizer_success} | "
                f"accepted={refit_accepted} | "
                f"nit={optimizer_nit} | "
                f"start={selected_start} | "
                f"train_NLL={candidate_nll:.6f}"
            )

        if (
            fitted_theta is None
            or h_prev is None
            or rv_prev is None
            or active_window_start is None
            or active_window_end is None
        ):
            raise RuntimeError(
                "The HEAVY rolling state has not been initialized."
            )

        omega, alpha, beta = map(
            float,
            fitted_theta,
        )

        # =================================================
        # Daily one-step-ahead forecast
        # =================================================

        h_next = (
            omega
            + alpha * rv_prev
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
                "forecast_number": forecast_number,
                "target_index": target_idx,

                # Fixed for all forecasts inside the block.
                "window_start": active_window_start,
                "window_end": active_window_end - 1,

                "refit": should_refit,
                "refit_accepted": refit_accepted,
                "optimizer_success": optimizer_success,
                "optimizer_message": optimizer_message,
                "optimizer_nit": optimizer_nit,
                "optimizer_nfev": optimizer_nfev,
                "selected_start": selected_start,
                "candidate_nll": candidate_nll,

                "omega": omega,
                "alpha": alpha,
                "beta": beta,

                "alpha_plus_beta": alpha + beta,

                "h_origin": h_prev,
                "rv_origin": rv_prev,
                "predicted_variance": h_next,
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
        rv_prev = actual_rv_value

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
        "parameter_history": parameter_history,
        "n_total": n_total,
        "rolling_window_size": window_size,
        "n_oos": n_oos,
        "refit_every": refit_every,
        "window_step": refit_every,
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

    h = h[valid]
    rv = rv[valid]

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

    h = h[valid]
    rv = rv[valid]

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


def mse_volatility(
    h_pred,
    rv_true,
    eps=1e-8,
):
    h_pred = np.asarray(
        h_pred,
        dtype=np.float64,
    ).reshape(-1)

    rv_true = np.asarray(
        rv_true,
        dtype=np.float64,
    ).reshape(-1)

    mask = (
        np.isfinite(h_pred)
        & np.isfinite(rv_true)
    )

    h_pred = h_pred[mask]
    rv_true = rv_true[mask]

    if len(h_pred) == 0:
        return np.nan

    h = np.maximum(
        h_pred,
        eps,
    )

    rv = np.maximum(
        rv_true,
        eps,
    )

    return float(
        np.mean(
            (
                np.sqrt(rv)
                - np.sqrt(h)
            ) ** 2
        )
    )


def mse_variance(
    h_pred,
    rv_true,
    eps=1e-8,
):
    h_pred = np.asarray(
        h_pred,
        dtype=np.float64,
    ).reshape(-1)

    rv_true = np.asarray(
        rv_true,
        dtype=np.float64,
    ).reshape(-1)

    mask = (
        np.isfinite(h_pred)
        & np.isfinite(rv_true)
    )

    h_pred = h_pred[mask]
    rv_true = rv_true[mask]

    if len(h_pred) == 0:
        return np.nan

    h = np.maximum(
        h_pred,
        eps,
    )

    rv = np.maximum(
        rv_true,
        eps,
    )

    return float(
        np.mean(
            (rv - h) ** 2
        )
    )

def mae_variance(h_pred, rv_true, eps=1e-8):
    """
    Mean absolute error between predicted variance and realized variance.

    MAE_var = mean(|RV_t - h_t|)
    """
    h_pred = np.asarray(
        h_pred,
        dtype=np.float64,
    ).reshape(-1)

    rv_true = np.asarray(
        rv_true,
        dtype=np.float64,
    ).reshape(-1)

    mask = (
        np.isfinite(h_pred)
        & np.isfinite(rv_true)
    )

    h_pred = h_pred[mask]
    rv_true = rv_true[mask]

    if len(h_pred) == 0:
        return np.nan

    h = np.maximum(h_pred, eps)
    rv = np.maximum(rv_true, eps)

    return float(np.mean(np.abs(rv - h)))

def return_nll(
    h_pred,
    y_true,
    eps=1e-12,
):
    """
    Gaussian return negative log-likelihood.
    """
    h_pred = np.asarray(
        h_pred,
        dtype=np.float64,
    ).reshape(-1)

    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    ).reshape(-1)

    mask = (
        np.isfinite(h_pred)
        & np.isfinite(y_true)
    )

    h_pred = h_pred[mask]
    y_true = y_true[mask]

    if len(h_pred) == 0:
        return np.nan

    h = np.maximum(
        h_pred,
        eps,
    )

    return float(
        np.mean(
            0.5
            * (
                np.log(2.0 * np.pi)
                + np.log(h)
                + (y_true ** 2) / h
            )
        )
    )

import matplotlib.pyplot as plt


def plot_heavy_fit(
    rv_train,
    h_train,
    rv_test,
    h_pred,
    ticker,
    save_path,
):
    """
    Plot fitted variance (train) and one-step forecasts (test).
    """

    rv_train = np.asarray(rv_train)
    h_train = np.asarray(h_train)

    rv_test = np.asarray(rv_test)
    h_pred = np.asarray(h_pred)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=False,
    )


    axes[0].plot(
        rv_train,
        lw=1.2,
        label="Realized variance",
    )

    axes[0].plot(
        h_train,
        lw=1.2,
        label="HEAVY fitted variance",
    )

    axes[0].set_title(f"{ticker} - Training fit")
    axes[0].set_ylabel("Variance")
    axes[0].legend()


    axes[1].plot(
        rv_test,
        lw=1.2,
        label="Realized variance",
    )

    axes[1].plot(
        h_pred,
        lw=1.2,
        label="One-step forecast",
    )

    axes[1].set_title(f"{ticker} - Test forecast")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Variance")
    axes[1].legend()

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()
    
def summarize_rolling_heavy(
    rolling_result,
):
    h_pred = np.asarray(
        rolling_result["predicted_variance"],
        dtype=np.float64,
    ).reshape(-1)

    rv_true = np.asarray(
        rolling_result["actual_rv"],
        dtype=np.float64,
    ).reshape(-1)

    returns_true = np.asarray(
        rolling_result["actual_returns"],
        dtype=np.float64,
    ).reshape(-1)

    if not (
        len(h_pred)
        == len(rv_true)
        == len(returns_true)
    ):
        raise ValueError(
            "HEAVY rolling forecasts and targets "
            "are not aligned."
        )

    return {
        "n_total": int(
            rolling_result["n_total"]
        ),
        "rolling_window_size": int(
            rolling_result[
                "rolling_window_size"
            ]
        ),
        "n_oos": int(
            rolling_result["n_oos"]
        ),
        "refit_every": int(
            rolling_result["refit_every"]
        ),
        "window_step": int(
            rolling_result["window_step"]
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


def run_rolling_heavy_for_ticker(
    data_dict,
    ticker,
    refit_every=21,
    warm_start=True,
):
    """
    Run fixed 50:50 monthly block rolling HEAVY-r evaluation.
    """
    (
        y_train_old,
        rv_train_old,
        y_test_old,
        rv_test_old,
    ) = get_heavy_arrays_from_data_dict(
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
        fixed_half_sample_rolling_heavy(
            ticker=ticker,
            y_full=y_full,
            rv_full=rv_full,
            refit_every=refit_every,
            warm_start=warm_start,
        )
    )

    metrics = {
        "ticker": ticker,
        **summarize_rolling_heavy(
            rolling_result
        ),
    }

    return {
        "rolling_result": rolling_result,
        "h_pred": rolling_result[
            "predicted_variance"
        ],
        "y_oos": rolling_result[
            "actual_returns"
        ],
        "rv_oos": rolling_result[
            "actual_rv"
        ],
        "parameter_history": rolling_result[
            "parameter_history"
        ],
        "metrics": metrics,
    }