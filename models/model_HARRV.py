"""
Likelihood-trained level HAR-RV (HAR-QML).

Specification
-------------
For t >= monthly_window,

    h_t_raw =
        beta_0
        + beta_d * RV_{t-1}
        + beta_w * mean(RV_{t-5}, ..., RV_{t-1})
        + beta_m * mean(RV_{t-22}, ..., RV_{t-1})

and
 
    r_t | F_{t-1} ~ N(0, h_t),

where

    r_t = 100 * (log_price_t - log_price_{t-1}).

Important
---------
- beta is completely UNCONSTRAINED.
- No softplus, no positivity bounds, no parameter restrictions.
- Because the Gaussian likelihood requires positive variance, only the
  likelihood evaluation uses

      h_t_safe = max(h_t_raw, EPS).

- The raw HAR forecast is retained and returned unchanged.
- The model is fitted using RETURN Gaussian QML, not RV-MSE/OLS.
- RV enters only as the HAR regressor history.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import minimize


EPS = 1e-8


def _as_finite_1d(values, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)

    if arr.size == 0:
        raise ValueError(f"{name} is empty.")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")

    return arr


def build_harrv_qml_training_arrays(
    rv: np.ndarray,
    log_price: np.ndarray,
    weekly_window: int = 5,
    monthly_window: int = 22,
):
    """
    Build HAR regressors and same-day returns for QML estimation.

    For each target day t:

        daily   = RV_{t-1}
        weekly  = mean(RV_{t-5}, ..., RV_{t-1})
        monthly = mean(RV_{t-22}, ..., RV_{t-1})

        r_t = 100 * (log_price_t - log_price_{t-1})

    Thus X_t is F_{t-1}-measurable, while r_t is the likelihood observation.
    RV_t itself is NOT used as the estimation target.
    """
    rv = _as_finite_1d(rv, name="rv")
    log_price = _as_finite_1d(log_price, name="log_price")

    if len(rv) != len(log_price):
        raise ValueError(
            "rv and log_price must have exactly the same length."
        )

    if weekly_window < 1:
        raise ValueError("weekly_window must be >= 1.")

    if monthly_window < weekly_window:
        raise ValueError(
            "monthly_window must be >= weekly_window."
        )

    if len(rv) <= monthly_window:
        raise ValueError(
            "Insufficient observations for HAR-QML estimation."
        )

    X: List[List[float]] = []
    returns: List[float] = []
    target_indices: List[int] = []

    for t in range(monthly_window, len(rv)):
        daily = float(rv[t - 1])
        weekly = float(
            np.mean(rv[t - weekly_window:t])
        )
        monthly = float(
            np.mean(rv[t - monthly_window:t])
        )

        r_t = float(
            100.0
            * (
                log_price[t]
                - log_price[t - 1]
            )
        )

        X.append(
            [
                1.0,
                daily,
                weekly,
                monthly,
            ]
        )
        returns.append(r_t)
        target_indices.append(t)

    X_arr = np.asarray(
        X,
        dtype=np.float64,
    )
    returns_arr = np.asarray(
        returns,
        dtype=np.float64,
    )
    target_idx_arr = np.asarray(
        target_indices,
        dtype=np.int64,
    )

    if X_arr.ndim != 2 or X_arr.shape[1] != 4:
        raise RuntimeError(
            "Unexpected HAR-QML design-matrix shape."
        )

    if len(X_arr) != len(returns_arr):
        raise RuntimeError(
            "HAR regressors and returns are misaligned."
        )

    return (
        X_arr,
        returns_arr,
        target_idx_arr,
    )


def gaussian_return_qml_loss(
    beta: np.ndarray,
    X: np.ndarray,
    returns: np.ndarray,
    eps: float = EPS,
) -> float:
    """
    Mean Gaussian negative log-likelihood, omitting the constant:

        0.5 * [log(h_t) + r_t^2 / h_t]

    beta is unconstrained.

    Only inside the likelihood:
        h_safe = max(X beta, eps)
    """
    beta = np.asarray(
        beta,
        dtype=np.float64,
    ).reshape(-1)

    if beta.shape != (4,):
        return 1e100

    if not np.all(np.isfinite(beta)):
        return 1e100

    h_raw = X @ beta

    if not np.all(np.isfinite(h_raw)):
        return 1e100

    h_safe = np.maximum(
        h_raw,
        eps,
    )

    losses = 0.5 * (
        np.log(h_safe)
        + (returns ** 2) / h_safe
    )

    if not np.all(np.isfinite(losses)):
        return 1e100

    return float(
        np.mean(losses)
    )


def _default_initial_beta(
    X: np.ndarray,
    returns: np.ndarray,
) -> np.ndarray:
    """
    Obtain a numerically useful unconstrained starting point by regressing
    squared returns on the HAR regressors.

    This is ONLY an optimizer initialization. Final estimation is entirely
    by Gaussian return QML.
    """
    y0 = returns ** 2

    beta0, *_ = np.linalg.lstsq(
        X,
        y0,
        rcond=None,
    )

    beta0 = np.asarray(
        beta0,
        dtype=np.float64,
    ).reshape(-1)

    if (
        beta0.shape != (4,)
        or not np.all(np.isfinite(beta0))
    ):
        mean_r2 = max(
            float(np.mean(y0)),
            EPS,
        )

        beta0 = np.asarray(
            [
                0.10 * mean_r2,
                0.40,
                0.30,
                0.20,
            ],
            dtype=np.float64,
        )

    h0_raw = X @ beta0

    # If the initial point is numerically pathological, use a conservative
    # positive-intercept starting point. This is still only initialization;
    # beta remains unconstrained during optimization.
    if (
        not np.all(np.isfinite(h0_raw))
        or np.mean(h0_raw > EPS) < 0.90
    ):
        mean_r2 = max(
            float(np.mean(y0)),
            EPS,
        )

        beta0 = np.asarray(
            [
                mean_r2,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )

    return beta0


def fit_harrv_qml(
    rv_train: np.ndarray,
    log_price_train: np.ndarray,
    weekly_window: int = 5,
    monthly_window: int = 22,
    beta_init: Optional[np.ndarray] = None,
    eps: float = EPS,
    maxiter=1000,
    xtol=1e-6,
    ftol=1e-8,
) -> Dict:
    """
    Estimate unconstrained level HAR parameters using Gaussian return QML.

    No beta constraints are imposed.

    Because h_raw = X @ beta can be non-positive during optimization,
    the likelihood uses h_safe = max(h_raw, eps).

    Powell is used because the max() floor creates a non-smooth objective.
    """
    rv_train = _as_finite_1d(
        rv_train,
        name="rv_train",
    )
    log_price_train = _as_finite_1d(
        log_price_train,
        name="log_price_train",
    )

    if len(rv_train) != len(log_price_train):
        raise ValueError(
            "rv_train and log_price_train must have equal length."
        )

    if len(rv_train) <= monthly_window + 4:
        raise ValueError(
            "Insufficient observations for HAR-QML estimation."
        )

    (
        X,
        returns,
        target_indices,
    ) = build_harrv_qml_training_arrays(
        rv=rv_train,
        log_price=log_price_train,
        weekly_window=weekly_window,
        monthly_window=monthly_window,
    )

    if beta_init is None:
        x0 = _default_initial_beta(
            X,
            returns,
        )
    else:
        x0 = np.asarray(
            beta_init,
            dtype=np.float64,
        ).reshape(-1)

        if (
            x0.shape != (4,)
            or not np.all(np.isfinite(x0))
        ):
            x0 = _default_initial_beta(
                X,
                returns,
            )

    initial_loss = gaussian_return_qml_loss(
        beta=x0,
        X=X,
        returns=returns,
        eps=eps,
    )

    result = minimize(
        gaussian_return_qml_loss,
        x0=x0,
        args=(
            X,
            returns,
            eps,
        ),
        method="Powell",
        options={
            "maxiter": maxiter,
            "xtol": xtol,
            "ftol": ftol,
            "disp": False,
        },
    )

    beta_hat = np.asarray(
        result.x,
        dtype=np.float64,
    ).reshape(-1)

    final_loss = float(result.fun)

    # Conservative fallback: never accept a non-finite or materially worse
    # optimizer result.
    accepted = bool(
        beta_hat.shape == (4,)
        and np.all(np.isfinite(beta_hat))
        and np.isfinite(final_loss)
        and final_loss <= initial_loss + 1e-10
    )

    if not accepted:
        beta_hat = x0.copy()
        final_loss = initial_loss

    h_raw = X @ beta_hat
    h_safe = np.maximum(
        h_raw,
        eps,
    )

    return {
        "beta_hat": beta_hat,
        "beta_0": float(beta_hat[0]),
        "beta_daily": float(beta_hat[1]),
        "beta_weekly": float(beta_hat[2]),
        "beta_monthly": float(beta_hat[3]),
        "h_raw_in_sample": h_raw,
        "h_safe_in_sample": h_safe,
        "returns_in_sample": returns,
        "qml_loss": float(final_loss),
        "initial_qml_loss": float(initial_loss),
        "optimizer_success": bool(result.success),
        "optimizer_accepted": bool(accepted),
        "optimizer_message": str(result.message),
        "optimizer_nit": int(
            getattr(result, "nit", -1)
        ),
        "optimizer_nfev": int(
            getattr(result, "nfev", -1)
        ),
        "n_nonpositive_fitted_h": int(
            np.sum(h_raw <= 0.0)
        ),
        "n_regression_obs": int(
            len(returns)
        ),
        "target_indices": target_indices,
        "weekly_window": int(
            weekly_window
        ),
        "monthly_window": int(
            monthly_window
        ),
    }


def harrv_qml_forecast_1step_from_history(
    rv_history: np.ndarray,
    beta_hat: np.ndarray,
    weekly_window: int = 5,
    monthly_window: int = 22,
    eps: float = EPS,
) -> Dict[str, float]:
    """
    Generate a one-step-ahead level HAR forecast.

    If rv_history ends at RV_t, this forecasts h_{t+1}.

    The RAW unconstrained forecast is returned as predicted_h_raw.
    A floored version is also returned for operations requiring h > 0.
    """
    rv_history = _as_finite_1d(
        rv_history,
        name="rv_history",
    )

    beta_hat = np.asarray(
        beta_hat,
        dtype=np.float64,
    ).reshape(-1)

    if beta_hat.shape != (4,):
        raise ValueError(
            "beta_hat must contain four HAR coefficients."
        )

    if not np.all(np.isfinite(beta_hat)):
        raise ValueError(
            "beta_hat contains non-finite values."
        )

    if len(rv_history) < monthly_window:
        raise ValueError(
            "Insufficient RV history for HAR forecast."
        )

    daily = float(
        rv_history[-1]
    )
    weekly = float(
        np.mean(
            rv_history[
                -weekly_window:
            ]
        )
    )
    monthly = float(
        np.mean(
            rv_history[
                -monthly_window:
            ]
        )
    )

    x_next = np.asarray(
        [
            1.0,
            daily,
            weekly,
            monthly,
        ],
        dtype=np.float64,
    )

    predicted_h_raw = float(
        x_next @ beta_hat
    )

    predicted_h_safe = float(
        max(
            predicted_h_raw,
            eps,
        )
    )

    return {
        "predicted_h_raw": predicted_h_raw,
        "predicted_h_safe": predicted_h_safe,
        "rv_daily": daily,
        "rv_weekly": weekly,
        "rv_monthly": monthly,
    }


def fixed_half_sample_rolling_harrv_qml(
    *,
    ticker: str,
    rv_full: np.ndarray,
    log_price_full: np.ndarray,
    window_size: Optional[int] = None,
    refit_every: int = 21,
    weekly_window: int = 5,
    monthly_window: int = 22,
    eps: float = EPS,
    maxiter: int = 1000,
    xtol: float = 1e-6,
    ftol: float = 1e-8,
) -> Dict:
    """
    Fixed-window rolling HAR-QML.

    Design
    ------
    1. Uses full raw-RV and log-price chronology.
    2. Default rolling window length is floor(N/2).
    3. Refit at forecasts 1, 22, 43, ... when refit_every=21.
    4. HAR beta is estimated by Gaussian RETURN QML.
    5. beta is completely unconstrained.
    6. h is floored only where positivity is mathematically required.
    7. Forecast every trading day.
    8. Target RV_t is excluded from its own forecast information set.
    """
    rv_full = _as_finite_1d(
        rv_full,
        name="rv_full",
    )
    log_price_full = _as_finite_1d(
        log_price_full,
        name="log_price_full",
    )

    if len(rv_full) != len(log_price_full):
        raise ValueError(
            "rv_full and log_price_full must have equal length."
        )

    if refit_every < 1:
        raise ValueError(
            "refit_every must be at least 1."
        )

    n_total = len(rv_full)

    if window_size is None:
        window_size = n_total // 2

    window_size = int(
        window_size
    )

    if (
        window_size <= monthly_window
        or window_size > n_total
    ):
        raise ValueError(
            "Invalid HAR-QML estimation-window size."
        )

    first_oos_index = window_size
    n_oos = n_total - first_oos_index

    predicted_h_raw: List[float] = []
    predicted_h_safe: List[float] = []
    actual_raw_rv: List[float] = []
    target_indices: List[int] = []
    parameter_history: List[Dict] = []

    fitted_beta = None
    active_window_start = None
    active_window_end = None
    refit_count = 0
    accepted_refit_count = 0

    print(
        "\n"
        + "=" * 90
    )
    print(
        f"{ticker}: fixed 50:50 rolling HAR-QML"
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
        f"Weekly HAR lag           : {weekly_window}"
    )
    print(
        f"Monthly HAR lag          : {monthly_window}"
    )
    print(
        f"Parameter update         : every {refit_every} trading days"
    )
    print(
        "Beta constraints         : none"
    )
    print(
        f"Likelihood floor         : max(h_raw, {eps:g})"
    )

    for forecast_number, target_idx in enumerate(
        range(
            first_oos_index,
            n_total,
        ),
        start=1,
    ):
        should_refit = (
            fitted_beta is None
            or (
                forecast_number - 1
            )
            % refit_every
            == 0
        )

        fit_qml_loss = np.nan
        n_regression_obs = np.nan
        optimizer_success = False
        optimizer_accepted = False
        optimizer_message = ""
        n_nonpositive_fitted_h = np.nan

        if should_refit:
            refit_count += 1

            active_window_end = target_idx
            active_window_start = (
                active_window_end
                - window_size
            )

            if active_window_start < 0:
                raise RuntimeError(
                    "HAR-QML estimation window begins before sample."
                )

            rv_window = rv_full[
                active_window_start:
                active_window_end
            ]

            log_price_window = log_price_full[
                active_window_start:
                active_window_end
            ]

            if (
                len(rv_window)
                != window_size
                or len(log_price_window)
                != window_size
            ):
                raise RuntimeError(
                    "Unexpected HAR-QML rolling-window length."
                )

            fit_result = fit_harrv_qml(
                rv_train=rv_window,
                log_price_train=log_price_window,
                weekly_window=weekly_window,
                monthly_window=monthly_window,
                beta_init=fitted_beta,
                eps=eps,
            )

            fitted_beta = np.asarray(
                fit_result["beta_hat"],
                dtype=np.float64,
            ).reshape(-1)

            fit_qml_loss = float(
                fit_result["qml_loss"]
            )
            n_regression_obs = int(
                fit_result[
                    "n_regression_obs"
                ]
            )
            optimizer_success = bool(
                fit_result[
                    "optimizer_success"
                ]
            )
            optimizer_accepted = bool(
                fit_result[
                    "optimizer_accepted"
                ]
            )
            optimizer_message = str(
                fit_result[
                    "optimizer_message"
                ]
            )
            n_nonpositive_fitted_h = int(
                fit_result[
                    "n_nonpositive_fitted_h"
                ]
            )

            if optimizer_accepted:
                accepted_refit_count += 1

            print(
                f"\nRefit #{refit_count} "
                f"at forecast {forecast_number} | "
                f"window=[{active_window_start}, "
                f"{active_window_end - 1}] | "
                f"N_qml={n_regression_obs} | "
                f"loss={fit_qml_loss:.8f} | "
                f"accepted={optimizer_accepted}"
            )

        if fitted_beta is None:
            raise RuntimeError(
                "HAR-QML model has not been fitted."
            )

        # Uses RV information only through target_idx - 1.
        rv_history = rv_full[
            :target_idx
        ]

        forecast_result = (
            harrv_qml_forecast_1step_from_history(
                rv_history=rv_history,
                beta_hat=fitted_beta,
                weekly_window=weekly_window,
                monthly_window=monthly_window,
                eps=eps,
            )
        )

        h_raw = float(
            forecast_result[
                "predicted_h_raw"
            ]
        )
        h_safe = float(
            forecast_result[
                "predicted_h_safe"
            ]
        )
        rv_true = float(
            rv_full[
                target_idx
            ]
        )

        (
            beta_0,
            beta_daily,
            beta_weekly,
            beta_monthly,
        ) = map(
            float,
            fitted_beta,
        )

        predicted_h_raw.append(
            h_raw
        )
        predicted_h_safe.append(
            h_safe
        )
        actual_raw_rv.append(
            rv_true
        )
        target_indices.append(
            target_idx
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
                "fit_qml_loss":
                    fit_qml_loss,
                "n_regression_obs":
                    n_regression_obs,
                "optimizer_success":
                    optimizer_success,
                "optimizer_accepted":
                    optimizer_accepted,
                "optimizer_message":
                    optimizer_message,
                "n_nonpositive_fitted_h":
                    n_nonpositive_fitted_h,
                "beta_0":
                    beta_0,
                "beta_daily":
                    beta_daily,
                "beta_weekly":
                    beta_weekly,
                "beta_monthly":
                    beta_monthly,
                "rv_daily_origin":
                    float(
                        forecast_result[
                            "rv_daily"
                        ]
                    ),
                "rv_weekly_origin":
                    float(
                        forecast_result[
                            "rv_weekly"
                        ]
                    ),
                "rv_monthly_origin":
                    float(
                        forecast_result[
                            "rv_monthly"
                        ]
                    ),
                "predicted_h_raw":
                    h_raw,
                "predicted_h_safe":
                    h_safe,
            }
        )

    predicted_h_raw_arr = np.asarray(
        predicted_h_raw,
        dtype=np.float64,
    )
    predicted_h_safe_arr = np.asarray(
        predicted_h_safe,
        dtype=np.float64,
    )
    actual_raw_rv_arr = np.asarray(
        actual_raw_rv,
        dtype=np.float64,
    )
    target_indices_arr = np.asarray(
        target_indices,
        dtype=np.int64,
    )

    if not (
        len(predicted_h_raw_arr)
        == len(predicted_h_safe_arr)
        == len(actual_raw_rv_arr)
        == len(target_indices_arr)
        == n_oos
    ):
        raise RuntimeError(
            "HAR-QML rolling outputs are not aligned."
        )

    return {
        "predicted_h_raw":
            predicted_h_raw_arr,
        "predicted_h_safe":
            predicted_h_safe_arr,
        "actual_raw_rv":
            actual_raw_rv_arr,
        "target_indices":
            target_indices_arr,
        "parameter_history":
            parameter_history,
        "n_total":
            int(n_total),
        "rolling_window_size":
            int(window_size),
        "first_oos_index":
            int(first_oos_index),
        "n_oos":
            int(n_oos),
        "refit_every":
            int(refit_every),
        "window_step":
            int(refit_every),
        "weekly_window":
            int(weekly_window),
        "monthly_window":
            int(monthly_window),
        "n_refits":
            int(refit_count),
        "n_refits_accepted":
            int(accepted_refit_count),
    }











# """
# Likelihood-trained level HAR-RV (HAR-QML), extended to POINT multi-horizon forecasts.

# Specification
# -------------
# For t >= monthly_window,

#     h_t_raw =
#         beta_0
#         + beta_d * RV_{t-1}
#         + beta_w * mean(RV_{t-5}, ..., RV_{t-1})
#         + beta_m * mean(RV_{t-22}, ..., RV_{t-1})

# and

#     r_t | F_{t-1} ~ N(0, h_t),

# where

#     r_t = 100 * (log_price_t - log_price_{t-1}).

# Estimation
# ----------
# - beta is completely unconstrained.
# - No softplus, no positivity bounds, no parameter restrictions.
# - Because the Gaussian likelihood requires positive variance, only the
#   likelihood evaluation uses

#       h_t_safe = max(h_t_raw, EPS).

# - The raw HAR fitted value is retained.
# - The model is fitted using RETURN Gaussian QML, not RV-MSE/OLS.
# - RV enters only as the HAR regressor history.

# Point multi-horizon forecasting
# -------------------------------
# At forecast origin n, the same fitted daily HAR-QML model is propagated
# recursively to horizons h = 1, 5, 10, 20.

# For each future day k = 1, ..., H,

#     h_{n+k|n, raw}
#         = beta_0
#         + beta_d * RV^*_{n+k-1}
#         + beta_w * mean(last 5 values of the augmented RV history)
#         + beta_m * mean(last 22 values of the augmented RV history),

# where the augmented RV history contains:
#     - observed RV through day n; and
#     - for already-forecast future days, the previous positive HAR forecast.

# Because RV is non-negative and the Gaussian variance must be positive, the
# recursive pseudo-RV update uses

#     RV^*_{n+k} = h_{n+k|n, safe}
#                = max(h_{n+k|n, raw}, EPS).

# The raw forecast is still saved unchanged for diagnostics.

# This is an ITERATED POINT-HORIZON design:
#     horizon h uses h_{n+h|n},
# not
#     sum_{j=1}^h h_{n+j|n}.

# Evaluation target:
#     RV_{n+h}.

# Return diagnostic:
#     r_{n+h}
#         = 100 * (log_price_{n+h} - log_price_{n+h-1}).
# """

# from __future__ import annotations

# from typing import Dict, List, Optional, Sequence, Tuple

# import numpy as np
# from scipy.optimize import minimize


# EPS = 1e-8


# # =========================================================
# # Basic helpers
# # =========================================================

# def _as_finite_1d(
#     values,
#     *,
#     name: str,
# ) -> np.ndarray:
#     arr = np.asarray(
#         values,
#         dtype=np.float64,
#     ).reshape(-1)

#     if arr.size == 0:
#         raise ValueError(
#             f"{name} is empty."
#         )

#     if not np.all(
#         np.isfinite(arr)
#     ):
#         raise ValueError(
#             f"{name} contains non-finite values."
#         )

#     return arr


# def _validate_horizons(
#     forecast_horizons: Sequence[int],
# ) -> Tuple[int, ...]:
#     horizons = tuple(
#         sorted(
#             set(
#                 int(h)
#                 for h
#                 in forecast_horizons
#             )
#         )
#     )

#     if (
#         len(horizons) == 0
#         or any(
#             h < 1
#             for h
#             in horizons
#         )
#     ):
#         raise ValueError(
#             "forecast_horizons must contain positive integers."
#         )

#     return horizons


# # =========================================================
# # Training arrays
# # =========================================================

# def build_harrv_qml_training_arrays(
#     rv: np.ndarray,
#     log_price: np.ndarray,
#     weekly_window: int = 5,
#     monthly_window: int = 22,
# ):
#     """
#     Build HAR regressors and same-day returns for QML estimation.

#     For each target day t:

#         daily   = RV_{t-1}
#         weekly  = mean(RV_{t-5}, ..., RV_{t-1})
#         monthly = mean(RV_{t-22}, ..., RV_{t-1})

#         r_t = 100 * (log_price_t - log_price_{t-1})

#     X_t is F_{t-1}-measurable, while r_t is the likelihood observation.
#     RV_t itself is NOT used as the estimation target.
#     """
#     rv = _as_finite_1d(
#         rv,
#         name="rv",
#     )

#     log_price = _as_finite_1d(
#         log_price,
#         name="log_price",
#     )

#     if len(rv) != len(log_price):
#         raise ValueError(
#             "rv and log_price must have exactly the same length."
#         )

#     if weekly_window < 1:
#         raise ValueError(
#             "weekly_window must be >= 1."
#         )

#     if monthly_window < weekly_window:
#         raise ValueError(
#             "monthly_window must be >= weekly_window."
#         )

#     if len(rv) <= monthly_window:
#         raise ValueError(
#             "Insufficient observations for HAR-QML estimation."
#         )

#     X: List[List[float]] = []
#     returns: List[float] = []
#     target_indices: List[int] = []

#     for t in range(
#         monthly_window,
#         len(rv),
#     ):
#         daily = float(
#             rv[
#                 t - 1
#             ]
#         )

#         weekly = float(
#             np.mean(
#                 rv[
#                     t - weekly_window:
#                     t
#                 ]
#             )
#         )

#         monthly = float(
#             np.mean(
#                 rv[
#                     t - monthly_window:
#                     t
#                 ]
#             )
#         )

#         r_t = float(
#             100.0
#             * (
#                 log_price[t]
#                 - log_price[
#                     t - 1
#                 ]
#             )
#         )

#         X.append(
#             [
#                 1.0,
#                 daily,
#                 weekly,
#                 monthly,
#             ]
#         )

#         returns.append(
#             r_t
#         )

#         target_indices.append(
#             t
#         )

#     X_arr = np.asarray(
#         X,
#         dtype=np.float64,
#     )

#     returns_arr = np.asarray(
#         returns,
#         dtype=np.float64,
#     )

#     target_idx_arr = np.asarray(
#         target_indices,
#         dtype=np.int64,
#     )

#     if (
#         X_arr.ndim != 2
#         or X_arr.shape[1] != 4
#     ):
#         raise RuntimeError(
#             "Unexpected HAR-QML design-matrix shape."
#         )

#     if len(X_arr) != len(
#         returns_arr
#     ):
#         raise RuntimeError(
#             "HAR regressors and returns are misaligned."
#         )

#     return (
#         X_arr,
#         returns_arr,
#         target_idx_arr,
#     )


# # =========================================================
# # Gaussian return QML
# # =========================================================

# def gaussian_return_qml_loss(
#     beta: np.ndarray,
#     X: np.ndarray,
#     returns: np.ndarray,
#     eps: float = EPS,
# ) -> float:
#     """
#     Mean Gaussian negative log-likelihood, omitting the constant:

#         0.5 * [log(h_t) + r_t^2 / h_t]

#     beta is unconstrained.

#     Only inside the likelihood:
#         h_safe = max(X beta, eps)
#     """
#     beta = np.asarray(
#         beta,
#         dtype=np.float64,
#     ).reshape(-1)

#     if beta.shape != (4,):
#         return 1e100

#     if not np.all(
#         np.isfinite(beta)
#     ):
#         return 1e100

#     h_raw = (
#         X
#         @ beta
#     )

#     if not np.all(
#         np.isfinite(h_raw)
#     ):
#         return 1e100

#     h_safe = np.maximum(
#         h_raw,
#         eps,
#     )

#     losses = 0.5 * (
#         np.log(
#             h_safe
#         )
#         + (
#             returns ** 2
#         )
#         / h_safe
#     )

#     if not np.all(
#         np.isfinite(losses)
#     ):
#         return 1e100

#     return float(
#         np.mean(
#             losses
#         )
#     )


# def _default_initial_beta(
#     X: np.ndarray,
#     returns: np.ndarray,
# ) -> np.ndarray:
#     """
#     Numerically useful unconstrained starting point.

#     This OLS calculation is ONLY an optimizer initializer.
#     Final estimation is entirely Gaussian return QML.
#     """
#     y0 = (
#         returns ** 2
#     )

#     beta0, *_ = (
#         np.linalg.lstsq(
#             X,
#             y0,
#             rcond=None,
#         )
#     )

#     beta0 = np.asarray(
#         beta0,
#         dtype=np.float64,
#     ).reshape(-1)

#     if (
#         beta0.shape != (4,)
#         or not np.all(
#             np.isfinite(beta0)
#         )
#     ):
#         mean_r2 = max(
#             float(
#                 np.mean(y0)
#             ),
#             EPS,
#         )

#         beta0 = np.asarray(
#             [
#                 0.10
#                 * mean_r2,
#                 0.40,
#                 0.30,
#                 0.20,
#             ],
#             dtype=np.float64,
#         )

#     h0_raw = (
#         X
#         @ beta0
#     )

#     if (
#         not np.all(
#             np.isfinite(h0_raw)
#         )
#         or np.mean(
#             h0_raw > EPS
#         )
#         < 0.90
#     ):
#         mean_r2 = max(
#             float(
#                 np.mean(y0)
#             ),
#             EPS,
#         )

#         beta0 = np.asarray(
#             [
#                 mean_r2,
#                 0.0,
#                 0.0,
#                 0.0,
#             ],
#             dtype=np.float64,
#         )

#     return beta0


# def fit_harrv_qml(
#     rv_train: np.ndarray,
#     log_price_train: np.ndarray,
#     weekly_window: int = 5,
#     monthly_window: int = 22,
#     beta_init: Optional[
#         np.ndarray
#     ] = None,
#     eps: float = EPS,
# ) -> Dict:
#     """
#     Estimate unconstrained level HAR parameters using Gaussian return QML.

#     No beta constraints are imposed.

#     Because h_raw = X @ beta can be non-positive during optimization,
#     the likelihood uses h_safe = max(h_raw, eps).

#     Powell is retained because the max() floor makes the objective non-smooth.
#     """
#     rv_train = _as_finite_1d(
#         rv_train,
#         name="rv_train",
#     )

#     log_price_train = _as_finite_1d(
#         log_price_train,
#         name="log_price_train",
#     )

#     if len(rv_train) != len(
#         log_price_train
#     ):
#         raise ValueError(
#             "rv_train and log_price_train must have equal length."
#         )

#     if len(rv_train) <= (
#         monthly_window
#         + 4
#     ):
#         raise ValueError(
#             "Insufficient observations for HAR-QML estimation."
#         )

#     (
#         X,
#         returns,
#         target_indices,
#     ) = (
#         build_harrv_qml_training_arrays(
#             rv=rv_train,
#             log_price=(
#                 log_price_train
#             ),
#             weekly_window=(
#                 weekly_window
#             ),
#             monthly_window=(
#                 monthly_window
#             ),
#         )
#     )

#     if beta_init is None:
#         x0 = (
#             _default_initial_beta(
#                 X,
#                 returns,
#             )
#         )

#     else:
#         x0 = np.asarray(
#             beta_init,
#             dtype=np.float64,
#         ).reshape(-1)

#         if (
#             x0.shape != (4,)
#             or not np.all(
#                 np.isfinite(x0)
#             )
#         ):
#             x0 = (
#                 _default_initial_beta(
#                     X,
#                     returns,
#                 )
#             )

#     initial_loss = (
#         gaussian_return_qml_loss(
#             beta=x0,
#             X=X,
#             returns=returns,
#             eps=eps,
#         )
#     )

#     result = minimize(
#         gaussian_return_qml_loss,
#         x0=x0,
#         args=(
#             X,
#             returns,
#             eps,
#         ),
#         method="Powell",
#         options={
#             "maxiter":
#                 5000,

#             "xtol":
#                 1e-8,

#             "ftol":
#                 1e-10,

#             "disp":
#                 False,
#         },
#     )

#     beta_hat = np.asarray(
#         result.x,
#         dtype=np.float64,
#     ).reshape(-1)

#     final_loss = (
#         gaussian_return_qml_loss(
#             beta=beta_hat,
#             X=X,
#             returns=returns,
#             eps=eps,
#         )
#     )

#     # Never accept a non-finite or materially worse solution.
#     accepted = bool(
#         beta_hat.shape == (4,)
#         and np.all(
#             np.isfinite(beta_hat)
#         )
#         and np.isfinite(
#             final_loss
#         )
#         and (
#             final_loss
#             <= initial_loss
#             + 1e-10
#         )
#     )

#     if not accepted:
#         beta_hat = (
#             x0.copy()
#         )
#         final_loss = (
#             initial_loss
#         )

#     h_raw = (
#         X
#         @ beta_hat
#     )

#     h_safe = np.maximum(
#         h_raw,
#         eps,
#     )

#     return {
#         "beta_hat":
#             beta_hat,

#         "beta_0":
#             float(
#                 beta_hat[0]
#             ),

#         "beta_daily":
#             float(
#                 beta_hat[1]
#             ),

#         "beta_weekly":
#             float(
#                 beta_hat[2]
#             ),

#         "beta_monthly":
#             float(
#                 beta_hat[3]
#             ),

#         "h_raw_in_sample":
#             h_raw,

#         "h_safe_in_sample":
#             h_safe,

#         "returns_in_sample":
#             returns,

#         "qml_loss":
#             float(
#                 final_loss
#             ),

#         "initial_qml_loss":
#             float(
#                 initial_loss
#             ),

#         "optimizer_success":
#             bool(
#                 result.success
#             ),

#         "optimizer_accepted":
#             bool(
#                 accepted
#             ),

#         "optimizer_message":
#             str(
#                 result.message
#             ),

#         "optimizer_nit":
#             int(
#                 getattr(
#                     result,
#                     "nit",
#                     -1,
#                 )
#             ),

#         "optimizer_nfev":
#             int(
#                 getattr(
#                     result,
#                     "nfev",
#                     -1,
#                 )
#             ),

#         "n_nonpositive_fitted_h":
#             int(
#                 np.sum(
#                     h_raw <= 0.0
#                 )
#             ),

#         "n_regression_obs":
#             int(
#                 len(
#                     returns
#                 )
#             ),

#         "target_indices":
#             target_indices,

#         "weekly_window":
#             int(
#                 weekly_window
#             ),

#         "monthly_window":
#             int(
#                 monthly_window
#             ),
#     }


# # =========================================================
# # One-step helper
# # =========================================================

# def harrv_qml_forecast_1step_from_history(
#     rv_history: np.ndarray,
#     beta_hat: np.ndarray,
#     weekly_window: int = 5,
#     monthly_window: int = 22,
#     eps: float = EPS,
# ) -> Dict[str, float]:
#     """
#     One-step level HAR forecast.

#     If rv_history ends at RV_n, this forecasts h_{n+1|n}.
#     """
#     rv_history = (
#         _as_finite_1d(
#             rv_history,
#             name="rv_history",
#         )
#     )

#     beta_hat = np.asarray(
#         beta_hat,
#         dtype=np.float64,
#     ).reshape(-1)

#     if beta_hat.shape != (4,):
#         raise ValueError(
#             "beta_hat must contain four HAR coefficients."
#         )

#     if not np.all(
#         np.isfinite(
#             beta_hat
#         )
#     ):
#         raise ValueError(
#             "beta_hat contains non-finite values."
#         )

#     if len(rv_history) < (
#         monthly_window
#     ):
#         raise ValueError(
#             "Insufficient RV history for HAR forecast."
#         )

#     daily = float(
#         rv_history[-1]
#     )

#     weekly = float(
#         np.mean(
#             rv_history[
#                 -weekly_window:
#             ]
#         )
#     )

#     monthly = float(
#         np.mean(
#             rv_history[
#                 -monthly_window:
#             ]
#         )
#     )

#     x_next = np.asarray(
#         [
#             1.0,
#             daily,
#             weekly,
#             monthly,
#         ],
#         dtype=np.float64,
#     )

#     predicted_h_raw = float(
#         x_next
#         @ beta_hat
#     )

#     predicted_h_safe = float(
#         max(
#             predicted_h_raw,
#             eps,
#         )
#     )

#     return {
#         "predicted_h_raw":
#             predicted_h_raw,

#         "predicted_h_safe":
#             predicted_h_safe,

#         "rv_daily":
#             daily,

#         "rv_weekly":
#             weekly,

#         "rv_monthly":
#             monthly,
#     }


# # =========================================================
# # Iterated POINT-horizon HAR forecast
# # =========================================================

# def harrv_qml_point_forecast_path(
#     *,
#     rv_history: np.ndarray,
#     beta_hat: np.ndarray,
#     max_horizon: int,
#     weekly_window: int = 5,
#     monthly_window: int = 22,
#     eps: float = EPS,
# ) -> Dict[str, np.ndarray]:
#     """
#     Propagate the SAME fitted daily HAR-QML model to future days.

#     The observed RV history ends at forecast origin n.

#     For k = 1,...,H:
#         1. Construct daily/weekly/monthly HAR regressors from the augmented
#            history available under the forecasting recursion.
#         2. Compute the RAW point forecast h_{n+k|n,raw}.
#         3. Compute the positive variance forecast
#                h_{n+k|n,safe} = max(raw, eps).
#         4. Append the positive forecast to the pseudo-RV history so that
#            future HAR regressors can be formed without using future data.

#     This returns daily point forecasts, NOT cumulative sums.
#     """
#     observed_history = (
#         _as_finite_1d(
#             rv_history,
#             name="rv_history",
#         )
#     )

#     beta_hat = np.asarray(
#         beta_hat,
#         dtype=np.float64,
#     ).reshape(-1)

#     if beta_hat.shape != (4,):
#         raise ValueError(
#             "beta_hat must contain four HAR coefficients."
#         )

#     if not np.all(
#         np.isfinite(
#             beta_hat
#         )
#     ):
#         raise ValueError(
#             "beta_hat contains non-finite values."
#         )

#     max_horizon = int(
#         max_horizon
#     )

#     if max_horizon < 1:
#         raise ValueError(
#             "max_horizon must be >= 1."
#         )

#     if len(
#         observed_history
#     ) < monthly_window:
#         raise ValueError(
#             "Insufficient observed RV history "
#             "for multi-horizon HAR forecast."
#         )

#     # Only the last monthly_window observations are needed to propagate
#     # future HAR lags. Keeping a list makes the recursion explicit.
#     augmented_history = list(
#         np.asarray(
#             observed_history[
#                 -monthly_window:
#             ],
#             dtype=np.float64,
#         )
#     )

#     h_raw_path = np.empty(
#         max_horizon,
#         dtype=np.float64,
#     )

#     h_safe_path = np.empty(
#         max_horizon,
#         dtype=np.float64,
#     )

#     daily_path = np.empty(
#         max_horizon,
#         dtype=np.float64,
#     )

#     weekly_path = np.empty(
#         max_horizon,
#         dtype=np.float64,
#     )

#     monthly_path = np.empty(
#         max_horizon,
#         dtype=np.float64,
#     )

#     for k in range(
#         max_horizon
#     ):
#         hist = np.asarray(
#             augmented_history,
#             dtype=np.float64,
#         )

#         daily = float(
#             hist[-1]
#         )

#         weekly = float(
#             np.mean(
#                 hist[
#                     -weekly_window:
#                 ]
#             )
#         )

#         monthly = float(
#             np.mean(
#                 hist[
#                     -monthly_window:
#                 ]
#             )
#         )

#         x_next = np.asarray(
#             [
#                 1.0,
#                 daily,
#                 weekly,
#                 monthly,
#             ],
#             dtype=np.float64,
#         )

#         h_raw = float(
#             x_next
#             @ beta_hat
#         )

#         if not np.isfinite(
#             h_raw
#         ):
#             raise RuntimeError(
#                 f"Non-finite HAR raw forecast "
#                 f"at horizon {k + 1}."
#             )

#         h_safe = float(
#             max(
#                 h_raw,
#                 eps,
#             )
#         )

#         h_raw_path[k] = (
#             h_raw
#         )

#         h_safe_path[k] = (
#             h_safe
#         )

#         daily_path[k] = (
#             daily
#         )

#         weekly_path[k] = (
#             weekly
#         )

#         monthly_path[k] = (
#             monthly
#         )

#         # Future RV is unknown. Use the positive daily variance forecast as
#         # the no-look-ahead pseudo-RV input for subsequent HAR regressors.
#         augmented_history.append(
#             h_safe
#         )

#         # Keep only what the monthly HAR component can use.
#         if len(
#             augmented_history
#         ) > monthly_window:
#             augmented_history = (
#                 augmented_history[
#                     -monthly_window:
#                 ]
#             )

#     return {
#         "predicted_h_raw_path":
#             h_raw_path,

#         "predicted_h_safe_path":
#             h_safe_path,

#         "rv_daily_path":
#             daily_path,

#         "rv_weekly_path":
#             weekly_path,

#         "rv_monthly_path":
#             monthly_path,
#     }


# # =========================================================
# # Fixed 50:50 rolling POINT-horizon HAR-QML
# # =========================================================

# def fixed_half_sample_rolling_harrv_qml_multihorizon(
#     *,
#     ticker: str,
#     rv_full: np.ndarray,
#     log_price_full: np.ndarray,
#     forecast_horizons: Sequence[int] = (
#         1,
#         5,
#         10,
#         20,
#     ),
#     window_size: Optional[
#         int
#     ] = None,
#     refit_every: int = 21,
#     weekly_window: int = 5,
#     monthly_window: int = 22,
#     eps: float = EPS,
# ) -> Dict:
#     """
#     Fixed-window rolling ITERATED POINT-HORIZON HAR-QML.

#     Fair-comparison design
#     ----------------------
#     - Estimate ONE daily HAR-QML model.
#     - Propagate that SAME model recursively to h = 1, 5, 10, 20.
#     - No horizon-specific re-estimation.
#     - No cumulative variance target.
#     - All horizons use the SAME OOS forecast origins.

#     At origin n:
#         forecast(h) = h_{n+h|n}

#         target(h)   = RV_{n+h}

#         return(h)   = 100 * (
#                           log P_{n+h}
#                           - log P_{n+h-1}
#                       ).

#     Future observed RV is NEVER used in constructing h > 1 forecasts.
#     """
#     rv_full = _as_finite_1d(
#         rv_full,
#         name="rv_full",
#     )

#     log_price_full = (
#         _as_finite_1d(
#             log_price_full,
#             name="log_price_full",
#         )
#     )

#     if len(rv_full) != len(
#         log_price_full
#     ):
#         raise ValueError(
#             "rv_full and log_price_full must have equal length."
#         )

#     if refit_every < 1:
#         raise ValueError(
#             "refit_every must be at least 1."
#         )

#     if weekly_window < 1:
#         raise ValueError(
#             "weekly_window must be >= 1."
#         )

#     if monthly_window < (
#         weekly_window
#     ):
#         raise ValueError(
#             "monthly_window must be >= weekly_window."
#         )

#     horizons = (
#         _validate_horizons(
#             forecast_horizons
#         )
#     )

#     max_horizon = max(
#         horizons
#     )

#     n_total = len(
#         rv_full
#     )

#     if window_size is None:
#         window_size = (
#             n_total
#             // 2
#         )

#     window_size = int(
#         window_size
#     )

#     if (
#         window_size
#         <= monthly_window
#         or window_size
#         > n_total
#     ):
#         raise ValueError(
#             "Invalid HAR-QML estimation-window size."
#         )

#     # Same daily indexing as SDE / GARCH / RG:
#     #
#     # first OOS target = window_size
#     # first forecast origin = window_size - 1
#     first_oos_target = (
#         window_size
#     )

#     first_origin = (
#         first_oos_target
#         - 1
#     )

#     # Common origins: every origin must have the h=max_horizon target.
#     last_origin = (
#         n_total
#         - max_horizon
#         - 1
#     )

#     if (
#         last_origin
#         < first_origin
#     ):
#         raise ValueError(
#             "Not enough observations for "
#             "the requested maximum horizon."
#         )

#     origins = np.arange(
#         first_origin,
#         last_origin + 1,
#         dtype=np.int64,
#     )

#     n_oos_common = len(
#         origins
#     )

#     results = {
#         h: {
#             "predicted_h_raw":
#                 [],

#             "predicted_h_safe":
#                 [],

#             "actual_rv":
#                 [],

#             "actual_return":
#                 [],

#             "origin_index":
#                 [],

#             "target_end_index":
#                 [],

#             "rv_daily_forecast_regressor":
#                 [],

#             "rv_weekly_forecast_regressor":
#                 [],

#             "rv_monthly_forecast_regressor":
#                 [],
#         }
#         for h
#         in horizons
#     }

#     parameter_history: List[
#         Dict
#     ] = []

#     fitted_beta = None

#     active_window_start = None
#     active_window_end = None

#     refit_count = 0
#     accepted_refit_count = 0

#     print(
#         "\n"
#         + "=" * 100
#     )

#     print(
#         f"{ticker}: fixed 50:50 rolling "
#         "HAR-QML POINT forecasts"
#     )

#     print(
#         "=" * 100
#     )

#     print(
#         f"Total observations       : "
#         f"{n_total}"
#     )

#     print(
#         f"Estimation-window size   : "
#         f"{window_size}"
#     )

#     print(
#         f"Forecast horizons        : "
#         f"{horizons}"
#     )

#     print(
#         f"Common OOS origins       : "
#         f"{n_oos_common}"
#     )

#     print(
#         f"Weekly HAR lag           : "
#         f"{weekly_window}"
#     )

#     print(
#         f"Monthly HAR lag          : "
#         f"{monthly_window}"
#     )

#     print(
#         "Parameter update         : "
#         f"every {refit_every} origins"
#     )

#     print(
#         "Forecast design          : "
#         "same daily HAR recursively propagated"
#     )

#     print(
#         "Evaluation target        : "
#         "RV_{n+h}, not cumulative RV"
#     )

#     print(
#         "Beta constraints         : none"
#     )

#     print(
#         f"Likelihood/forecast floor: "
#         f"max(h_raw, {eps:g})"
#     )

#     for (
#         forecast_number,
#         origin_idx,
#     ) in enumerate(
#         origins,
#         start=1,
#     ):
#         should_refit = (
#             fitted_beta is None
#             or (
#                 forecast_number
#                 - 1
#             )
#             % refit_every
#             == 0
#         )

#         fit_qml_loss = np.nan
#         n_regression_obs = np.nan
#         optimizer_success = False
#         optimizer_accepted = False
#         optimizer_message = ""
#         optimizer_nit = -1
#         optimizer_nfev = -1
#         n_nonpositive_fitted_h = np.nan

#         if should_refit:
#             refit_count += 1

#             # Estimation data end at the forecast origin n.
#             active_window_end = (
#                 origin_idx
#                 + 1
#             )

#             active_window_start = (
#                 active_window_end
#                 - window_size
#             )

#             if (
#                 active_window_start
#                 < 0
#             ):
#                 raise RuntimeError(
#                     "HAR-QML estimation window "
#                     "begins before sample."
#                 )

#             rv_window = (
#                 rv_full[
#                     active_window_start:
#                     active_window_end
#                 ]
#             )

#             log_price_window = (
#                 log_price_full[
#                     active_window_start:
#                     active_window_end
#                 ]
#             )

#             if (
#                 len(rv_window)
#                 != window_size
#                 or len(
#                     log_price_window
#                 )
#                 != window_size
#             ):
#                 raise RuntimeError(
#                     "Unexpected HAR-QML rolling-window length."
#                 )

#             fit_result = (
#                 fit_harrv_qml(
#                     rv_train=(
#                         rv_window
#                     ),
#                     log_price_train=(
#                         log_price_window
#                     ),
#                     weekly_window=(
#                         weekly_window
#                     ),
#                     monthly_window=(
#                         monthly_window
#                     ),
#                     beta_init=(
#                         fitted_beta
#                     ),
#                     eps=eps,
#                 )
#             )

#             fitted_beta = np.asarray(
#                 fit_result[
#                     "beta_hat"
#                 ],
#                 dtype=np.float64,
#             ).reshape(-1)

#             fit_qml_loss = float(
#                 fit_result[
#                     "qml_loss"
#                 ]
#             )

#             n_regression_obs = int(
#                 fit_result[
#                     "n_regression_obs"
#                 ]
#             )

#             optimizer_success = bool(
#                 fit_result[
#                     "optimizer_success"
#                 ]
#             )

#             optimizer_accepted = bool(
#                 fit_result[
#                     "optimizer_accepted"
#                 ]
#             )

#             optimizer_message = str(
#                 fit_result[
#                     "optimizer_message"
#                 ]
#             )

#             optimizer_nit = int(
#                 fit_result[
#                     "optimizer_nit"
#                 ]
#             )

#             optimizer_nfev = int(
#                 fit_result[
#                     "optimizer_nfev"
#                 ]
#             )

#             n_nonpositive_fitted_h = int(
#                 fit_result[
#                     "n_nonpositive_fitted_h"
#                 ]
#             )

#             if (
#                 optimizer_accepted
#             ):
#                 accepted_refit_count += 1

#             print(
#                 f"\nRefit #{refit_count} | "
#                 f"origin={origin_idx} | "
#                 f"window=[{active_window_start}, "
#                 f"{active_window_end - 1}] | "
#                 f"N_qml={n_regression_obs} | "
#                 f"loss={fit_qml_loss:.8f} | "
#                 f"accepted={optimizer_accepted}"
#             )

#         if fitted_beta is None:
#             raise RuntimeError(
#                 "HAR-QML model has not been fitted."
#             )

#         # Observed history only through forecast origin n.
#         rv_history = (
#             rv_full[
#                 :origin_idx
#                 + 1
#             ]
#         )

#         path = (
#             harrv_qml_point_forecast_path(
#                 rv_history=(
#                     rv_history
#                 ),
#                 beta_hat=(
#                     fitted_beta
#                 ),
#                 max_horizon=(
#                     max_horizon
#                 ),
#                 weekly_window=(
#                     weekly_window
#                 ),
#                 monthly_window=(
#                     monthly_window
#                 ),
#                 eps=eps,
#             )
#         )

#         (
#             beta_0,
#             beta_daily,
#             beta_weekly,
#             beta_monthly,
#         ) = map(
#             float,
#             fitted_beta,
#         )

#         for h in horizons:
#             target_idx = int(
#                 origin_idx
#                 + h
#             )

#             h_raw = float(
#                 path[
#                     "predicted_h_raw_path"
#                 ][
#                     h - 1
#                 ]
#             )

#             h_safe = float(
#                 path[
#                     "predicted_h_safe_path"
#                 ][
#                     h - 1
#                 ]
#             )

#             rv_true = float(
#                 rv_full[
#                     target_idx
#                 ]
#             )

#             return_true = float(
#                 100.0
#                 * (
#                     log_price_full[
#                         target_idx
#                     ]
#                     - log_price_full[
#                         target_idx - 1
#                     ]
#                 )
#             )

#             results[h][
#                 "predicted_h_raw"
#             ].append(
#                 h_raw
#             )

#             results[h][
#                 "predicted_h_safe"
#             ].append(
#                 h_safe
#             )

#             results[h][
#                 "actual_rv"
#             ].append(
#                 rv_true
#             )

#             results[h][
#                 "actual_return"
#             ].append(
#                 return_true
#             )

#             results[h][
#                 "origin_index"
#             ].append(
#                 int(
#                     origin_idx
#                 )
#             )

#             results[h][
#                 "target_end_index"
#             ].append(
#                 target_idx
#             )

#             results[h][
#                 "rv_daily_forecast_regressor"
#             ].append(
#                 float(
#                     path[
#                         "rv_daily_path"
#                     ][
#                         h - 1
#                     ]
#                 )
#             )

#             results[h][
#                 "rv_weekly_forecast_regressor"
#             ].append(
#                 float(
#                     path[
#                         "rv_weekly_path"
#                     ][
#                         h - 1
#                     ]
#                 )
#             )

#             results[h][
#                 "rv_monthly_forecast_regressor"
#             ].append(
#                 float(
#                     path[
#                         "rv_monthly_path"
#                     ][
#                         h - 1
#                     ]
#                 )
#             )

#         parameter_history.append(
#             {
#                 "forecast_number":
#                     int(
#                         forecast_number
#                     ),

#                 "origin_index":
#                     int(
#                         origin_idx
#                     ),

#                 "h1_target_index":
#                     int(
#                         origin_idx
#                         + 1
#                     ),

#                 "window_start":
#                     int(
#                         active_window_start
#                     ),

#                 "window_end":
#                     int(
#                         active_window_end
#                         - 1
#                     ),

#                 "refit":
#                     bool(
#                         should_refit
#                     ),

#                 "fit_qml_loss":
#                     float(
#                         fit_qml_loss
#                     ),

#                 "n_regression_obs":
#                     (
#                         int(
#                             n_regression_obs
#                         )
#                         if np.isfinite(
#                             n_regression_obs
#                         )
#                         else np.nan
#                     ),

#                 "optimizer_success":
#                     bool(
#                         optimizer_success
#                     ),

#                 "optimizer_accepted":
#                     bool(
#                         optimizer_accepted
#                     ),

#                 "optimizer_message":
#                     optimizer_message,

#                 "optimizer_nit":
#                     int(
#                         optimizer_nit
#                     ),

#                 "optimizer_nfev":
#                     int(
#                         optimizer_nfev
#                     ),

#                 "n_nonpositive_fitted_h":
#                     (
#                         int(
#                             n_nonpositive_fitted_h
#                         )
#                         if np.isfinite(
#                             n_nonpositive_fitted_h
#                         )
#                         else np.nan
#                     ),

#                 "beta_0":
#                     beta_0,

#                 "beta_daily":
#                     beta_daily,

#                 "beta_weekly":
#                     beta_weekly,

#                 "beta_monthly":
#                     beta_monthly,

#                 "h1_predicted_raw":
#                     float(
#                         path[
#                             "predicted_h_raw_path"
#                         ][0]
#                     ),

#                 "h1_predicted_safe":
#                     float(
#                         path[
#                             "predicted_h_safe_path"
#                         ][0]
#                     ),

#                 "rv_daily_origin":
#                     float(
#                         path[
#                             "rv_daily_path"
#                         ][0]
#                     ),

#                 "rv_weekly_origin":
#                     float(
#                         path[
#                             "rv_weekly_path"
#                         ][0]
#                     ),

#                 "rv_monthly_origin":
#                     float(
#                         path[
#                             "rv_monthly_path"
#                         ][0]
#                     ),
#             }
#         )

#         if (
#             forecast_number == 1
#             or should_refit
#             or forecast_number
#             % 100
#             == 0
#         ):
#             shown = ", ".join(
#                 f"h={h}: "
#                 f"{path['predicted_h_safe_path'][h - 1]:.6f}"
#                 for h
#                 in horizons
#             )

#             print(
#                 f"{ticker} origin "
#                 f"{forecast_number}/"
#                 f"{n_oos_common} | "
#                 f"origin_idx={origin_idx} | "
#                 f"refit={should_refit} | "
#                 f"{shown}"
#             )

#     # -----------------------------------------------------
#     # Convert and validate horizon outputs
#     # -----------------------------------------------------

#     for h in horizons:
#         for key in (
#             "predicted_h_raw",
#             "predicted_h_safe",
#             "actual_rv",
#             "actual_return",
#             "rv_daily_forecast_regressor",
#             "rv_weekly_forecast_regressor",
#             "rv_monthly_forecast_regressor",
#         ):
#             results[h][
#                 key
#             ] = np.asarray(
#                 results[h][
#                     key
#                 ],
#                 dtype=np.float64,
#             )

#         for key in (
#             "origin_index",
#             "target_end_index",
#         ):
#             results[h][
#                 key
#             ] = np.asarray(
#                 results[h][
#                     key
#                 ],
#                 dtype=np.int64,
#             )

#         lengths = {
#             len(
#                 results[h][
#                     key
#                 ]
#             )
#             for key in (
#                 "predicted_h_raw",
#                 "predicted_h_safe",
#                 "actual_rv",
#                 "actual_return",
#                 "origin_index",
#                 "target_end_index",
#                 "rv_daily_forecast_regressor",
#                 "rv_weekly_forecast_regressor",
#                 "rv_monthly_forecast_regressor",
#             )
#         }

#         if len(
#             lengths
#         ) != 1:
#             raise RuntimeError(
#                 f"{ticker} h={h}: "
#                 "HAR-QML rolling outputs are not aligned."
#             )

#         if (
#             len(
#                 results[h][
#                     "predicted_h_safe"
#                 ]
#             )
#             != n_oos_common
#         ):
#             raise RuntimeError(
#                 f"{ticker} h={h}: expected "
#                 f"{n_oos_common} common-origin forecasts."
#             )

#         if np.any(
#             results[h][
#                 "target_end_index"
#             ]
#             - results[h][
#                 "origin_index"
#             ]
#             != h
#         ):
#             raise RuntimeError(
#                 f"{ticker} h={h}: "
#                 "incorrect target indexing."
#             )

#         results[h][
#             "n_oos"
#         ] = int(
#             n_oos_common
#         )

#         results[h][
#             "n_nonpositive_forecasts"
#         ] = int(
#             np.sum(
#                 results[h][
#                     "predicted_h_raw"
#                 ]
#                 <= 0.0
#             )
#         )

#     return {
#         "results":
#             results,

#         "parameter_history":
#             parameter_history,

#         "n_total":
#             int(
#                 n_total
#             ),

#         "rolling_window_size":
#             int(
#                 window_size
#             ),

#         "first_oos_target":
#             int(
#                 first_oos_target
#             ),

#         "first_origin":
#             int(
#                 first_origin
#             ),

#         "last_origin":
#             int(
#                 last_origin
#             ),

#         "forecast_horizons":
#             horizons,

#         "n_oos_common":
#             int(
#                 n_oos_common
#             ),

#         "refit_every":
#             int(
#                 refit_every
#             ),

#         "window_step":
#             int(
#                 refit_every
#             ),

#         "weekly_window":
#             int(
#                 weekly_window
#             ),

#         "monthly_window":
#             int(
#                 monthly_window
#             ),

#         "n_refits":
#             int(
#                 refit_count
#             ),

#         "n_refits_accepted":
#             int(
#                 accepted_refit_count
#             ),
#     }


# # More explicit alias.
# fixed_half_sample_rolling_harrv_qml_point_horizon = (
#     fixed_half_sample_rolling_harrv_qml_multihorizon
# )


# # =========================================================
# # Evaluation metrics
# # =========================================================

# def qlike_var_np(
#     pred: np.ndarray,
#     actual: np.ndarray,
#     *,
#     eps: float = EPS,
# ) -> float:
#     pred = np.maximum(
#         np.asarray(
#             pred,
#             dtype=np.float64,
#         ).reshape(-1),
#         eps,
#     )

#     actual = np.maximum(
#         np.asarray(
#             actual,
#             dtype=np.float64,
#         ).reshape(-1),
#         eps,
#     )

#     if len(pred) != len(actual):
#         raise ValueError(
#             "QLIKE arrays must have equal length."
#         )

#     mask = (
#         np.isfinite(pred)
#         & np.isfinite(actual)
#     )

#     if not np.any(mask):
#         return np.nan

#     pred = pred[
#         mask
#     ]

#     actual = actual[
#         mask
#     ]

#     return float(
#         np.mean(
#             np.log(
#                 pred
#             )
#             + actual
#             / pred
#         )
#     )


# def mse_var_np(
#     pred: np.ndarray,
#     actual: np.ndarray,
# ) -> float:
#     pred = np.asarray(
#         pred,
#         dtype=np.float64,
#     ).reshape(-1)

#     actual = np.asarray(
#         actual,
#         dtype=np.float64,
#     ).reshape(-1)

#     if len(pred) != len(actual):
#         raise ValueError(
#             "MSE arrays must have equal length."
#         )

#     mask = (
#         np.isfinite(pred)
#         & np.isfinite(actual)
#     )

#     if not np.any(mask):
#         return np.nan

#     return float(
#         np.mean(
#             (
#                 actual[
#                     mask
#                 ]
#                 - pred[
#                     mask
#                 ]
#             ) ** 2
#         )
#     )


# def mae_var_np(
#     pred: np.ndarray,
#     actual: np.ndarray,
# ) -> float:
#     pred = np.asarray(
#         pred,
#         dtype=np.float64,
#     ).reshape(-1)

#     actual = np.asarray(
#         actual,
#         dtype=np.float64,
#     ).reshape(-1)

#     if len(pred) != len(actual):
#         raise ValueError(
#             "MAE arrays must have equal length."
#         )

#     mask = (
#         np.isfinite(pred)
#         & np.isfinite(actual)
#     )

#     if not np.any(mask):
#         return np.nan

#     return float(
#         np.mean(
#             np.abs(
#                 actual[
#                     mask
#                 ]
#                 - pred[
#                     mask
#                 ]
#             )
#         )
#     )


# def mse_vol_np(
#     pred: np.ndarray,
#     actual: np.ndarray,
#     *,
#     eps: float = EPS,
# ) -> float:
#     pred = np.maximum(
#         np.asarray(
#             pred,
#             dtype=np.float64,
#         ).reshape(-1),
#         eps,
#     )

#     actual = np.maximum(
#         np.asarray(
#             actual,
#             dtype=np.float64,
#         ).reshape(-1),
#         eps,
#     )

#     if len(pred) != len(actual):
#         raise ValueError(
#             "Volatility-MSE arrays must have equal length."
#         )

#     mask = (
#         np.isfinite(pred)
#         & np.isfinite(actual)
#     )

#     if not np.any(mask):
#         return np.nan

#     return float(
#         np.mean(
#             (
#                 np.sqrt(
#                     actual[
#                         mask
#                     ]
#                 )
#                 - np.sqrt(
#                     pred[
#                         mask
#                     ]
#                 )
#             ) ** 2
#         )
#     )


# def mae_vol_np(
#     pred: np.ndarray,
#     actual: np.ndarray,
#     *,
#     eps: float = EPS,
# ) -> float:
#     pred = np.maximum(
#         np.asarray(
#             pred,
#             dtype=np.float64,
#         ).reshape(-1),
#         eps,
#     )

#     actual = np.maximum(
#         np.asarray(
#             actual,
#             dtype=np.float64,
#         ).reshape(-1),
#         eps,
#     )

#     if len(pred) != len(actual):
#         raise ValueError(
#             "Volatility-MAE arrays must have equal length."
#         )

#     mask = (
#         np.isfinite(pred)
#         & np.isfinite(actual)
#     )

#     if not np.any(mask):
#         return np.nan

#     return float(
#         np.mean(
#             np.abs(
#                 np.sqrt(
#                     actual[
#                         mask
#                     ]
#                 )
#                 - np.sqrt(
#                     pred[
#                         mask
#                     ]
#                 )
#             )
#         )
#     )


# def return_nll_np(
#     pred_var: np.ndarray,
#     actual_return: np.ndarray,
#     *,
#     eps: float = EPS,
# ) -> float:
#     """
#     Gaussian NLL for the return on the SAME future day as the point variance
#     forecast.
#     """
#     pred_var = np.maximum(
#         np.asarray(
#             pred_var,
#             dtype=np.float64,
#         ).reshape(-1),
#         eps,
#     )

#     actual_return = np.asarray(
#         actual_return,
#         dtype=np.float64,
#     ).reshape(-1)

#     if len(pred_var) != len(
#         actual_return
#     ):
#         raise ValueError(
#             "Return-NLL arrays must have equal length."
#         )

#     mask = (
#         np.isfinite(pred_var)
#         & np.isfinite(
#             actual_return
#         )
#     )

#     if not np.any(mask):
#         return np.nan

#     h = pred_var[
#         mask
#     ]

#     r = actual_return[
#         mask
#     ]

#     return float(
#         np.mean(
#             0.5
#             * (
#                 np.log(
#                     2.0
#                     * np.pi
#                 )
#                 + np.log(h)
#                 + r ** 2
#                 / h
#             )
#         )
#     )

