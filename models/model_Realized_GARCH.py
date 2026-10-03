import numpy as np
from scipy.optimize import minimize


def get_rg_arrays_from_data_dict(
    data_dict,
    ticker,
):
    """
    Construct Realized-GARCH return/RV arrays from utils1 data.

    utils1 stores:
        y_t = log price
        RV_t = realized variance in percentage-return-squared units

    Realized-GARCH requires daily returns:
        r_t = 100 * (log P_t - log P_{t-1})

    Alignment:
        r_t corresponds to RV_t

    Therefore:
        returns_full = 100 * diff(log_price_full)
        rv_full      = rv_full_original[1:]

    No additional scaling factor c is applied.
    """

    data = data_dict[ticker]["train"]


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
            f"{ticker}: Realized-GARCH return/RV "
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
            f"{ticker}: Realized-GARCH train "
            "return/RV length mismatch."
        )

    if len(y_test) != len(rv_test):
        raise RuntimeError(
            f"{ticker}: Realized-GARCH test "
            "return/RV length mismatch."
        )

    if len(y_train) < 10:
        raise ValueError(
            f"{ticker}: insufficient Realized-GARCH "
            "training observations."
        )

    return (
        y_train,
        rv_train,
        y_test,
        rv_test,
    )

def realized_garch11_filter(
    y, rv, omega, gamma, beta, xi, phi, tau1, tau2,
    sigma2_0=None, rv_0=None
):
    y = np.asarray(y, dtype=np.float64)
    rv = np.asarray(rv, dtype=np.float64)

    T = len(y)
    assert len(rv) == T

    if sigma2_0 is None:
        sigma2_0 = max(1e-12, rv[0])

    if rv_0 is None:
        rv_0 = rv[0]

    sigma2 = np.empty(T, dtype=np.float64)
    z = np.empty(T, dtype=np.float64)

    prev_sigma2 = float(sigma2_0)
    prev_rv = float(rv_0)

    for t in range(T):
        h_t = omega + gamma * prev_rv + beta * prev_sigma2
        h_t = max(1e-12, h_t)

        sigma2[t] = h_t
        z[t] = y[t] / np.sqrt(h_t)

        prev_sigma2 = h_t
        prev_rv = rv[t]

    return sigma2, z


def realized_garch11_negloglik(y, rv, theta):
    omega, gamma, beta, xi, phi, tau1, tau2, sigma_u2 = theta

    # Basic constraints
    if omega <= 0:
        return 1e50
    if gamma < 0:
        return 1e50
    if beta < 0 or beta >= 0.999:
        return 1e50
    if phi < 0:
        return 1e50
    if sigma_u2 <= 1e-12:
        return 1e50

    # Stationarity-like Realized GARCH restriction
    if beta + gamma * phi >= 0.999:
        return 1e50

    sigma2, z = realized_garch11_filter(
        y, rv, omega, gamma, beta, xi, phi, tau1, tau2
    )

    if np.any(~np.isfinite(sigma2)) or np.any(sigma2 <= 0):
        return 1e50

    log2pi = np.log(2.0 * np.pi)

    # Return likelihood
    nll_y = 0.5 * np.sum(
        log2pi + np.log(sigma2) + (y ** 2) / sigma2
    )

    # Measurement likelihood
    rv_mean = xi + phi * sigma2 + tau1 * z + tau2 * (z ** 2 - 1.0)
    u = rv - rv_mean

    if np.any(~np.isfinite(rv_mean)) or np.any(~np.isfinite(u)):
        return 1e50

    nll_rv = 0.5 * np.sum(
        log2pi + np.log(sigma_u2) + (u ** 2) / sigma_u2
    )

    return float(nll_y + nll_rv)


def fit_realized_garch11_twostage(
    returns_train,
    rv_train,
    initial_theta=None,
):
    returns_train = np.asarray(
        returns_train,
        dtype=np.float64,
    ).reshape(-1)

    rv_train = np.maximum(
        np.asarray(
            rv_train,
            dtype=np.float64,
        ).reshape(-1),
        1e-12,
    )

    if len(returns_train) != len(rv_train):
        raise ValueError(
            "returns_train and rv_train must have equal length."
        )

    rv_mean = float(np.mean(rv_train))
    rv_var = float(np.var(rv_train))

    def obj(theta):
        return realized_garch11_negloglik(
            returns_train,
            rv_train,
            theta,
        )

    fresh_start = np.array(
        [
            0.05 * rv_mean,
            0.05,
            0.90,
            0.05 * rv_mean,
            0.80,
            -0.05,
            0.10,
            max(1e-4, 0.10 * rv_var),
        ],
        dtype=np.float64,
    )

    starts = [fresh_start]

    if initial_theta is not None:
        previous_start = np.asarray(
            initial_theta,
            dtype=np.float64,
        ).reshape(-1)

        if (
            len(previous_start) == 8
            and np.all(np.isfinite(previous_start))
        ):
            starts.insert(0, previous_start)

    bounds = [
        (1e-12, None),  # omega
        (0.0, None),    # gamma
        (0.0, 0.999),   # beta
        (None, None),   # xi
        (0.0, None),    # phi
        (None, None),   # tau1
        (None, None),   # tau2
        (1e-12, None),  # sigma_u2
    ]

    candidate_results = []

    for start_number, start in enumerate(starts, start=1):
        # Stage 1: derivative-free search for every refit.
        result_nm = minimize(
            obj,
            start,
            method="Nelder-Mead",
            options={
                "maxiter": 10000,
                "xatol": 1e-7,
                "fatol": 1e-7,
            },
        )

        nm_theta = np.asarray(
            result_nm.x,
            dtype=np.float64,
        )

        nm_nll = obj(nm_theta)

        # If Nelder-Mead generated an invalid point, retain start.
        if not np.isfinite(nm_nll) or nm_nll >= 1e49:
            nm_theta = start.copy()

        # Stage 2: bounded refinement.
        result_lbfgs = minimize(
            obj,
            nm_theta,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 30000,
                "ftol": 1e-12,
                "gtol": 1e-8,
                "maxls": 50,
            },
        )

        candidate_theta = np.asarray(
            result_lbfgs.x,
            dtype=np.float64,
        )

        candidate_nll = obj(candidate_theta)

        if (
            np.all(np.isfinite(candidate_theta))
            and np.isfinite(candidate_nll)
            and candidate_nll < 1e49
        ):
            candidate_results.append(
                {
                    "start_number": start_number,
                    "stage1": result_nm,
                    "stage2": result_lbfgs,
                    "theta_hat": candidate_theta,
                    "nll": float(candidate_nll),
                }
            )

    if not candidate_results:
        raise RuntimeError(
            "No valid Realized GARCH optimization result."
        )

    best = min(
        candidate_results,
        key=lambda item: item["nll"],
    )

    result_lbfgs = best["stage2"]

    return {
        "stage1": best["stage1"],
        "stage2": result_lbfgs,
        "theta_hat": best["theta_hat"],
        "nll": best["nll"],
        "success": bool(result_lbfgs.success),
        "message": str(result_lbfgs.message),
        "nit": int(getattr(result_lbfgs, "nit", -1)),
        "nfev": int(getattr(result_lbfgs, "nfev", -1)),
        "selected_start": int(best["start_number"]),
    }


def realized_garch11_forecast_from_window(
    returns_window: np.ndarray,
    rv_window: np.ndarray,
    theta_hat: np.ndarray,
):
    """
    Produce one-step forecasts of:

        h_{t+1|t}
        RV_{t+1|t} = xi + phi * h_{t+1|t}
    """
    (
        omega,
        gamma,
        beta,
        xi,
        phi,
        tau1,
        tau2,
        sigma_u2,
    ) = map(float, theta_hat)

    sigma2_window, _ = realized_garch11_filter(
        y=returns_window,
        rv=rv_window,
        omega=omega,
        gamma=gamma,
        beta=beta,
        xi=xi,
        phi=phi,
        tau1=tau1,
        tau2=tau2,
    )

    previous_h = float(sigma2_window[-1])
    previous_rv = float(rv_window[-1])

    h_next = (
        omega
        + gamma * previous_rv
        + beta * previous_h
    )
    h_next = max(1e-12, float(h_next))

    # Conditional measurement forecast:
    rv_next = xi + phi * h_next
    rv_next = max(1e-12, float(rv_next))

    return h_next, rv_next

def fixed_half_sample_rolling_realized_garch(
    ticker: str,
    y_full: np.ndarray,
    rv_full: np.ndarray,
    refit_every: int = 22,
    warm_start: bool = True,
):
    """
    Fixed 50:50 block rolling-window Realized GARCH evaluation.

    Design
    ------
    1. The estimation-window length is floor(N / 2).
    2. The estimation window advances by `refit_every` observations.
    3. Parameters are re-estimated every `refit_every` trading days.
    4. Daily one-step-ahead forecasts are generated between refits.
    5. Between refits, parameters remain fixed, but the conditional
       variance state is updated daily using newly observed RV.

    Example with window_size=50 and refit_every=22:

        Estimate on [0, 49]
        Forecast targets 50, ..., 71

        Estimate on [22, 71]
        Forecast targets 72, ..., 93

        Estimate on [44, 93]
        Forecast targets 94, ..., 115
    """
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

    if len(y_full) < 6:
        raise ValueError(
            "The sample is too short."
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
    predicted_rv = []
    actual_rv = []
    actual_returns = []
    parameter_history = []

    fitted_theta = None

    active_window_start = None
    active_window_end = None

    sigma2_prev = None
    rv_prev = None

    print("\n" + "=" * 90)
    print(f"{ticker}: block rolling Realized GARCH(1,1)")
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
        candidate_persistence = np.nan

        if should_refit:
     
            active_window_end = target_idx
            active_window_start = (
                active_window_end - window_size
            )

            if active_window_start < 0:
                raise RuntimeError(
                    "Active estimation window begins before "
                    "the start of the sample."
                )

            returns_window = y_full[
                active_window_start:
                active_window_end
            ]

            rv_window = rv_full[
                active_window_start:
                active_window_end
            ]

            if len(returns_window) != window_size:
                raise RuntimeError(
                    "Unexpected active-window length: "
                    f"{len(returns_window)} instead of "
                    f"{window_size}."
                )

            if len(returns_window) != len(rv_window):
                raise RuntimeError(
                    "Return and RV windows are not aligned."
                )

            previous_theta = (
                None
                if fitted_theta is None
                else fitted_theta.copy()
            )

            fit_result = fit_realized_garch11_twostage(
                returns_train=returns_window,
                rv_train=rv_window,
                initial_theta=(
                    previous_theta
                    if warm_start
                    else None
                ),
            )

            candidate_theta = np.asarray(
                fit_result["theta_hat"],
                dtype=np.float64,
            )

            candidate_nll = realized_garch11_negloglik(
                returns_window,
                rv_window,
                candidate_theta,
            )

            (
                omega_candidate,
                gamma_candidate,
                beta_candidate,
                xi_candidate,
                phi_candidate,
                tau1_candidate,
                tau2_candidate,
                sigma_u2_candidate,
            ) = candidate_theta

            candidate_persistence = (
                beta_candidate
                + gamma_candidate * phi_candidate
            )

            optimizer_success = bool(
                fit_result.get("success", False)
            )

            optimizer_message = str(
                fit_result.get("message", "")
            )

            optimizer_nit = int(
                fit_result.get("nit", -1)
            )

            optimizer_nfev = int(
                fit_result.get("nfev", -1)
            )

            selected_start = int(
                fit_result.get("selected_start", -1)
            )

            candidate_valid = (
                optimizer_success
                and candidate_theta.shape == (8,)
                and np.all(np.isfinite(candidate_theta))
                and np.isfinite(candidate_nll)
                and candidate_nll < 1e49
                and omega_candidate > 0.0
                and gamma_candidate >= 0.0
                and 0.0 <= beta_candidate < 0.999
                and phi_candidate >= 0.0
                and sigma_u2_candidate > 1e-12
                and candidate_persistence < 0.999
            )

            if candidate_valid:
                fitted_theta = candidate_theta
                refit_accepted = True

            elif previous_theta is not None:
                fitted_theta = previous_theta
                refit_accepted = False

                print(
                    f"Rejected refit at forecast "
                    f"{forecast_number}; retained previous "
                    "parameters."
                )

            else:
                raise RuntimeError(
                    f"Initial Realized GARCH fit failed for "
                    f"{ticker}."
                )

            # Refilter the active estimation window using the
            # accepted parameters. This initializes the state
            # at the end of the new monthly window.
            (
                omega,
                gamma,
                beta,
                xi,
                phi,
                tau1,
                tau2,
                sigma_u2,
            ) = map(float, fitted_theta)

            sigma2_window, _ = realized_garch11_filter(
                y=returns_window,
                rv=rv_window,
                omega=omega,
                gamma=gamma,
                beta=beta,
                xi=xi,
                phi=phi,
                tau1=tau1,
                tau2=tau2,
            )

            sigma2_prev = float(
                sigma2_window[-1]
            )

            rv_prev = float(
                rv_window[-1]
            )

            print(
                f"\nRefit {forecast_number} | "
                f"window=[{active_window_start}, "
                f"{active_window_end - 1}] | "
                f"success={optimizer_success} | "
                f"accepted={refit_accepted} | "
                f"nit={optimizer_nit} | "
                f"start={selected_start} | "
                f"joint_train_NLL={candidate_nll:.6f} | "
                f"persistence="
                f"{candidate_persistence:.6f}"
            )

        if (
            fitted_theta is None
            or sigma2_prev is None
            or rv_prev is None
            or active_window_start is None
            or active_window_end is None
        ):
            raise RuntimeError(
                "Rolling model state has not been initialized."
            )

        (
            omega,
            gamma,
            beta,
            xi,
            phi,
            tau1,
            tau2,
            sigma_u2,
        ) = map(float, fitted_theta)

        # -------------------------------------------------
        # Daily one-step-ahead forecast
        # -------------------------------------------------
        h_next = (
            omega
            + gamma * rv_prev
            + beta * sigma2_prev
        )

        h_next = max(
            1e-12,
            float(h_next),
        )

        rv_next = (
            xi + phi * h_next
        )

        rv_next = max(
            1e-12,
            float(rv_next),
        )

        predicted_variance.append(
            h_next
        )

        predicted_rv.append(
            rv_next
        )

        actual_rv_value = float(
            rv_full[target_idx]
        )

        actual_return_value = float(
            y_full[target_idx]
        )

        actual_rv.append(
            actual_rv_value
        )

        actual_returns.append(
            actual_return_value
        )

        parameter_history.append(
            {
                "forecast_number": forecast_number,
                "target_index": target_idx,

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
                "candidate_persistence":
                    candidate_persistence,

                "omega": omega,
                "gamma": gamma,
                "beta": beta,
                "xi": xi,
                "phi": phi,
                "tau1": tau1,
                "tau2": tau2,
                "sigma_u2": sigma_u2,

                "persistence": (
                    beta + gamma * phi
                ),

                "sigma2_origin": sigma2_prev,
                "rv_origin": rv_prev,
                "predicted_variance": h_next,
                "predicted_rv": rv_next,
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
            f"h={h_next:.6f} | "
            f"rv={rv_next:.6f}"
        )

        sigma2_prev = h_next
        rv_prev = actual_rv_value

    return {
        "predicted_variance": np.asarray(
            predicted_variance,
            dtype=np.float64,
        ),
        "predicted_rv": np.asarray(
            predicted_rv,
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

def qlike_variance(
    h_pred,
    rv_true,
    eps=1e-8,
):
    h = np.maximum(
        np.asarray(h_pred, dtype=np.float64).reshape(-1),
        eps,
    )

    rv = np.maximum(
        np.asarray(rv_true, dtype=np.float64).reshape(-1),
        eps,
    )

    if len(h) != len(rv):
        raise ValueError(
            "h_pred and rv_true must have equal length."
        )

    mask = np.isfinite(h) & np.isfinite(rv)

    if not np.any(mask):
        raise ValueError(
            "No finite observations available for QLIKE."
        )

    h = h[mask]
    rv = rv[mask]

    return float(
        np.mean(
            np.log(h) + rv / h
        )
    )

def mae_volatility(
    h_pred,
    rv_true,
    eps=1e-8,
):
    h = np.maximum(
        np.asarray(h_pred, dtype=np.float64).reshape(-1),
        eps,
    )

    rv = np.maximum(
        np.asarray(rv_true, dtype=np.float64).reshape(-1),
        eps,
    )

    if len(h) != len(rv):
        raise ValueError(
            "h_pred and rv_true must have equal length."
        )

    mask = np.isfinite(h) & np.isfinite(rv)
    h = h[mask]
    rv = rv[mask]

    return float(
        np.mean(
            np.abs(
                np.sqrt(rv) - np.sqrt(h)
            )
        )
    )


def mae_variance(
    h_pred,
    rv_true,
    eps=1e-8,
):
    h = np.maximum(
        np.asarray(h_pred, dtype=np.float64).reshape(-1),
        eps,
    )

    rv = np.maximum(
        np.asarray(rv_true, dtype=np.float64).reshape(-1),
        eps,
    )

    if len(h) != len(rv):
        raise ValueError(
            "h_pred and rv_true must have equal length."
        )

    mask = np.isfinite(h) & np.isfinite(rv)
    h = h[mask]
    rv = rv[mask]

    return float(
        np.mean(
            np.abs(rv - h)
        )
    )

def mse_volatility(h_pred, rv_true, eps=1e-8):
    h_pred = np.asarray(h_pred, dtype=np.float64)
    rv_true = np.asarray(rv_true, dtype=np.float64)

    mask = np.isfinite(h_pred) & np.isfinite(rv_true)
    h = np.maximum(h_pred[mask], eps)
    rv = np.maximum(rv_true[mask], eps)

    return float(np.mean((np.sqrt(rv) - np.sqrt(h)) ** 2))


def mse_variance(h_pred, rv_true, eps=1e-8):
    h_pred = np.asarray(h_pred, dtype=np.float64)
    rv_true = np.asarray(rv_true, dtype=np.float64)

    mask = np.isfinite(h_pred) & np.isfinite(rv_true)
    h = np.maximum(h_pred[mask], eps)
    rv = np.maximum(rv_true[mask], eps)

    return float(np.mean((rv - h) ** 2))


def return_nll_np(
    predicted_variance: np.ndarray,
    returns_true: np.ndarray,
    eps: float = 1e-8,
) -> float:
    h = np.maximum(
        np.asarray(
            predicted_variance,
            dtype=np.float64,
        ).reshape(-1),
        eps,
    )

    r = np.asarray(
        returns_true,
        dtype=np.float64,
    ).reshape(-1)

    if len(h) != len(r):
        raise ValueError(
            "predicted_variance and returns_true "
            "must have equal length."
        )

    valid = np.isfinite(h) & np.isfinite(r)

    if not np.any(valid):
        raise ValueError(
            "No finite observations available."
        )

    h = h[valid]
    r = r[valid]

    nll = 0.5 * (
        np.log(2.0 * np.pi)
        + np.log(h)
        + r**2 / h
    )

    return float(np.mean(nll))


def summarize_rolling_realized_garch(
    rolling_result,
):
    h_pred = np.asarray(
        rolling_result["predicted_variance"],
        dtype=np.float64,
    ).reshape(-1)

    rv_pred = np.asarray(
        rolling_result["predicted_rv"],
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

    lengths = {
        len(h_pred),
        len(rv_pred),
        len(rv_true),
        len(returns_true),
    }

    if len(lengths) != 1:
        raise ValueError(
            "Forecast and target arrays are not aligned: "
            f"{sorted(lengths)}"
        )

    return {
        "n_total": int(
            rolling_result["n_total"]
        ),
        "rolling_window_size": int(
            rolling_result["rolling_window_size"]
        ),
        "n_oos": int(
            rolling_result["n_oos"]
        ),
        "refit_every": int(
            rolling_result["refit_every"]
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
        "Return_NLL": return_nll_np(
            h_pred,
            returns_true,
        ),

        "QLIKE_rv": qlike_variance(
            rv_pred,
            rv_true,
        ),
        "MSE_vol_rv": mse_volatility(
            rv_pred,
            rv_true,
        ),
        "MSE_var_rv": mse_variance(
            rv_pred,
            rv_true,
        ),
        "MAE_vol_rv": mae_volatility(
            rv_pred,
            rv_true,
        ),
        "MAE_var_rv": mae_variance(
            rv_pred,
            rv_true,
        ),
    }