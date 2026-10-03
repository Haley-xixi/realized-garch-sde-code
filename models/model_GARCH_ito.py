import numpy as np
from scipy.optimize import minimize

def get_garch_ito_arrays_from_data_dict(
    data_dict,
    ticker,
    eps=1e-12,
):

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

    if len(log_price_train) < 2:
        raise ValueError(
            f"{ticker}: at least two training "
            "log-price observations are required."
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

    rv_full_raw = np.asarray(
        rv_full_original[1:],
        dtype=np.float64,
    ).copy()

    if len(returns_full) != len(rv_full_raw):
        raise RuntimeError(
            f"{ticker}: GARCH-Itô full return/RV "
            f"alignment failed: "
            f"{len(returns_full)} vs "
            f"{len(rv_full_raw)}."
        )


    valid = (
        np.isfinite(returns_full)
        & np.isfinite(rv_full_raw)
    )

    n_dropped = int(
        np.sum(~valid)
    )

    if n_dropped > 0:
        print(
            f"{ticker}: GARCH-Itô preprocessing | "
            f"dropped={n_dropped}"
        )

    returns_full = (
        returns_full[
            valid
        ]
    )

    rv_full_raw = (
        rv_full_raw[
            valid
        ]
    )


    rv_full_raw = np.maximum(
        rv_full_raw,
        eps,
    )


    original_train_price_n = len(
        log_price_train
    )

    return_split = (
        original_train_price_n
        - 1
    )

    if return_split <= 0:
        raise ValueError(
            f"{ticker}: invalid training return split."
        )

    if return_split >= len(returns_full):
        raise ValueError(
            f"{ticker}: invalid train/test split: "
            f"return_split={return_split}, "
            f"N_returns={len(returns_full)}."
        )


    y_train = (
        returns_full[
            :return_split
        ]
        .copy()
    )

    y_test = (
        returns_full[
            return_split:
        ]
        .copy()
    )

    rv_train_raw = (
        rv_full_raw[
            :return_split
        ]
        .copy()
    )

    rv_test_raw = (
        rv_full_raw[
            return_split:
        ]
        .copy()
    )

    if len(y_train) != len(rv_train_raw):
        raise RuntimeError(
            f"{ticker}: pre-scaling training return/RV "
            "length mismatch."
        )

    if len(y_test) != len(rv_test_raw):
        raise RuntimeError(
            f"{ticker}: pre-scaling test return/RV "
            "length mismatch."
        )

    if len(y_train) < 3:
        raise ValueError(
            f"{ticker}: insufficient GARCH-Itô "
            "training observations."
        )

    mean_return_squared_train = float(
        np.mean(
            y_train ** 2
        )
    )

    mean_rv_train_raw = float(
        np.mean(
            rv_train_raw
        )
    )

    if (
        not np.isfinite(mean_return_squared_train)
        or mean_return_squared_train <= 0
    ):
        raise ValueError(
            f"{ticker}: invalid mean training "
            f"return squared: "
            f"{mean_return_squared_train}."
        )

    if (
        not np.isfinite(mean_rv_train_raw)
        or mean_rv_train_raw <= eps
    ):
        raise ValueError(
            f"{ticker}: invalid mean raw "
            f"training RV: "
            f"{mean_rv_train_raw}."
        )

    rv_scale_factor = (
        mean_return_squared_train
        / mean_rv_train_raw
    )

    if (
        not np.isfinite(rv_scale_factor)
        or rv_scale_factor <= 0
    ):
        raise ValueError(
            f"{ticker}: invalid RV scaling factor: "
            f"{rv_scale_factor}."
        )


    rv_train = (
        rv_scale_factor
        * rv_train_raw
    )

    rv_test = (
        rv_scale_factor
        * rv_test_raw
    )

    rv_train = np.maximum(
        rv_train,
        eps,
    )

    rv_test = np.maximum(
        rv_test,
        eps,
    )


    if len(y_train) != len(rv_train):
        raise RuntimeError(
            f"{ticker}: GARCH-Itô training return/RV "
            "length mismatch."
        )

    if len(y_test) != len(rv_test):
        raise RuntimeError(
            f"{ticker}: GARCH-Itô test return/RV "
            "length mismatch."
        )

    if len(y_train) < 10:
        raise ValueError(
            f"{ticker}: insufficient GARCH-Itô "
            "training observations."
        )

    if len(y_test) == 0:
        raise ValueError(
            f"{ticker}: no GARCH-Itô test "
            "observations."
        )

    # =====================================================
    # 14. Diagnostics
    #
    # Designed to make comparison with HEAVY easy.
    # =====================================================

    print(
        f"\n{ticker}: GARCH-Itô data construction"
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
        f"Training N           : "
        f"{len(y_train)}"
    )

    print(
        f"Test N               : "
        f"{len(y_test)}"
    )

    return (
        y_train,
        rv_train,
        y_test,
        rv_test,
    )

# =========================================================
# Kim-Wang unified GARCH-Ito model
#
# Structural parameters:
#
#   theta = (omega, beta_ct, gamma)
#
# Implied daily recursion:
#
#   h_t
#     = omega_g
#     + gamma h_{t-1}
#     + beta_g y_{t-1}^2
#
# RV-QMLE:
#
#   0.5 * sum[
#       log(h_t)
#       + RV_t / h_t
#   ]
# =========================================================

def _stable_exponential_terms(
    beta_ct,
):
    beta_ct = float(
        beta_ct
    )

    if abs(beta_ct) < 1e-6:

        b = beta_ct
        b2 = b * b
        b3 = b2 * b
        b4 = b3 * b

        a1 = (
            b
            + 0.5 * b2
            + b3 / 6.0
            + b4 / 24.0
        )

        a2 = (
            0.5 * b2
            + b3 / 6.0
            + b4 / 24.0
        )

        a1_over_beta = (
            1.0
            + 0.5 * b
            + b2 / 6.0
            + b3 / 24.0
        )

        a2_over_beta = (
            0.5 * b
            + b2 / 6.0
            + b3 / 24.0
        )

        a2_over_beta_sq = (
            0.5
            + b / 6.0
            + b2 / 24.0
        )

    else:

        a1 = np.expm1(
            beta_ct
        )

        a2 = (
            a1
            - beta_ct
        )

        a1_over_beta = (
            a1
            / beta_ct
        )

        a2_over_beta = (
            a2
            / beta_ct
        )

        a2_over_beta_sq = (
            a2
            / beta_ct ** 2
        )

    return (
        float(a1),
        float(a2),
        float(a1_over_beta),
        float(a2_over_beta),
        float(a2_over_beta_sq),
    )


def garch_ito_daily_parameters(
    omega,
    beta_ct,
    gamma,
):
    omega = float(
        omega
    )

    beta_ct = float(
        beta_ct
    )

    gamma = float(
        gamma
    )

    (
        a1,
        _,
        a1_over_beta,
        a2_over_beta,
        _,
    ) = _stable_exponential_terms(
        beta_ct
    )

    omega_g = (
        omega
        * a1_over_beta
    )

    beta_g = (
        (gamma - 1.0)
        * a2_over_beta
        + a1
    )

    return (
        float(omega_g),
        float(beta_g),
        float(gamma),
    )


# =========================================================
# Paper initialization
# =========================================================

def garch_ito_initial_h(
    y_first,
    omega,
    beta_ct,
    gamma,
    eps=1e-12,
):
    sigma0_sq = max(
        float(y_first) ** 2,
        eps,
    )

    (
        _,
        _,
        a1_over_beta,
        _,
        a2_over_beta_sq,
    ) = _stable_exponential_terms(
        beta_ct
    )

    h1 = (
        float(omega)
        * a2_over_beta_sq
        +
        (
            (float(gamma) - 1.0)
            * a2_over_beta_sq
            + a1_over_beta
        )
        * sigma0_sq
    )

    return max(
        float(h1),
        eps,
    )


# =========================================================
# Filter
# =========================================================

def garch_ito_filter(
    y,
    omega,
    beta_ct,
    gamma,
    h0=None,
    initialization="paper",
    eps=1e-12,
):
    """
    h[t] is aligned with y[t] and RV[t].

    For t >= 1:

        h[t]
          = omega_g
          + gamma * h[t-1]
          + beta_g * y[t-1]^2
    """

    y = np.asarray(
        y,
        dtype=np.float64,
    ).reshape(-1)

    T = len(
        y
    )

    if T == 0:
        return np.empty(
            0,
            dtype=np.float64,
        )

    (
        omega_g,
        beta_g,
        gamma_daily,
    ) = garch_ito_daily_parameters(
        omega,
        beta_ct,
        gamma,
    )

    h = np.empty(
        T,
        dtype=np.float64,
    )

    if h0 is not None:

        h[0] = max(
            float(h0),
            eps,
        )

    elif initialization == "paper":

        h[0] = garch_ito_initial_h(
            y_first=y[0],
            omega=omega,
            beta_ct=beta_ct,
            gamma=gamma,
            eps=eps,
        )

    elif initialization == "sample_variance":

        h[0] = max(
            float(
                np.var(y)
            ),
            eps,
        )

    elif initialization == "mean_square":

        h[0] = max(
            float(
                np.mean(
                    y ** 2
                )
            ),
            eps,
        )

    else:

        raise ValueError(
            "initialization must be one of "
            "{'paper', 'sample_variance', "
            "'mean_square'}."
        )

    for t in range(
        1,
        T,
    ):

        h_t = (
            omega_g
            + gamma_daily
            * h[t - 1]
            + beta_g
            * y[t - 1] ** 2
        )

        if not np.isfinite(
            h_t
        ):
            h[t:] = np.nan
            break

        h[t] = max(
            float(h_t),
            eps,
        )

    return h


# =========================================================
# Parameter restrictions
# =========================================================

def garch_ito_parameter_valid(
    theta,
    persistence_limit=0.999,
    require_positive_beta_g=True,
):
    omega, beta_ct, gamma = map(
        float,
        theta,
    )

    if not np.all(
        np.isfinite(
            [
                omega,
                beta_ct,
                gamma,
            ]
        )
    ):
        return False

    if omega <= 0.0:
        return False

    if (
        beta_ct <= 0.0
        or beta_ct >= 1.0
    ):
        return False

    if (
        gamma <= 0.0
        or gamma >= 1.0
    ):
        return False

    (
        omega_g,
        beta_g,
        _,
    ) = garch_ito_daily_parameters(
        omega,
        beta_ct,
        gamma,
    )

    if (
        not np.isfinite(omega_g)
        or not np.isfinite(beta_g)
    ):
        return False

    if omega_g <= 0.0:
        return False

    if (
        require_positive_beta_g
        and beta_g < 0.0
    ):
        return False

    if (
        gamma
        + beta_g
        >= persistence_limit
    ):
        return False

    return True


# =========================================================
# RV-QMLE
# =========================================================

def garch_ito_rv_negloglik(
    y,
    rv,
    theta,
    initialization="paper",
    drop_initial=False,
    persistence_limit=0.999,
    eps=1e-12,
):
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
            "y and rv must have the same length."
        )

    if not garch_ito_parameter_valid(
        theta,
        persistence_limit=persistence_limit,
        require_positive_beta_g=True,
    ):
        return 1e50

    omega, beta_ct, gamma = map(
        float,
        theta,
    )

    h = garch_ito_filter(
        y=y,
        omega=omega,
        beta_ct=beta_ct,
        gamma=gamma,
        initialization=initialization,
        eps=eps,
    )

    if (
        np.any(
            ~np.isfinite(h)
        )
        or np.any(
            h <= 0.0
        )
    ):
        return 1e50

    rv_safe = np.maximum(
        rv,
        eps,
    )

    start = (
        1
        if (
            drop_initial
            and len(y) > 1
        )
        else 0
    )

    h_eval = h[
        start:
    ]

    rv_eval = rv_safe[
        start:
    ]

    if len(h_eval) == 0:
        return 1e50

    loss_terms = (
        np.log(h_eval)
        + rv_eval / h_eval
    )

    if np.any(
        ~np.isfinite(
            loss_terms
        )
    ):
        return 1e50

    return float(
        0.5
        * np.sum(
            loss_terms
        )
    )



def _make_garch_ito_starting_values(
    y_train,
    rv_train,
):
    """
    Single predetermined starting value for GARCH-Ito QMLE.
    """

    rv_train = np.asarray(
        rv_train,
        dtype=np.float64,
    ).reshape(-1)

    rv_level = max(
        float(np.mean(rv_train)),
        1e-8,
    )

    x0 = np.array(
        [
            0.02 * rv_level,  # omega
            0.20,             # beta_ct
            0.50,             # gamma
        ],
        dtype=np.float64,
    )

    return [x0]


# =========================================================
# Fit
# =========================================================

def fit_garch_ito_rv(
    y_train,
    rv_train,
    initial_theta=None,
    initialization="paper",
    drop_initial=False,
    persistence_limit=0.999,
    maxiter=20000,
    verbose=False,
):
    y_train = np.asarray(
        y_train,
        dtype=np.float64,
    ).reshape(-1)

    rv_train = np.asarray(
        rv_train,
        dtype=np.float64,
    ).reshape(-1)

    if len(y_train) != len(rv_train):
        raise ValueError(
            "y_train and rv_train must "
            "have the same length."
        )

    mask = (
        np.isfinite(y_train)
        & np.isfinite(rv_train)
    )

    y_train = y_train[
        mask
    ]

    rv_train = np.maximum(
        rv_train[
            mask
        ],
        1e-12,
    )

    if len(y_train) < 3:
        raise ValueError(
            "At least three training "
            "observations are required."
        )

    rv_level = max(
        float(
            np.mean(
                rv_train
            )
        ),
        1e-8,
    )

    bounds = [
        (
            1e-12,
            max(
                100.0
                * rv_level,
                1.0,
            ),
        ),
        (
            1e-6,
            0.999,
        ),
        (
            1e-6,
            0.999,
        ),
    ]

    def objective(
        theta,
    ):
        return garch_ito_rv_negloglik(
            y=y_train,
            rv=rv_train,
            theta=theta,
            initialization=initialization,
            drop_initial=drop_initial,
            persistence_limit=(
                persistence_limit
            ),
        )

    def stationarity_constraint(
        theta,
    ):
        _, beta_g, gamma = (
            garch_ito_daily_parameters(
                *theta
            )
        )

        return (
            persistence_limit
            - gamma
            - beta_g
        )

    def positive_beta_g_constraint(
        theta,
    ):
        _, beta_g, _ = (
            garch_ito_daily_parameters(
                *theta
            )
        )

        return beta_g

    constraints = [
        {
            "type": "ineq",
            "fun":
                stationarity_constraint,
        },
        {
            "type": "ineq",
            "fun":
                positive_beta_g_constraint,
        },
    ]

    starts = (
        _make_garch_ito_starting_values(
            y_train,
            rv_train,
        )
    )

    if initial_theta is not None:

        previous_start = np.asarray(
            initial_theta,
            dtype=np.float64,
        ).reshape(-1)

        if (
            previous_start.shape
            == (3,)
            and np.all(
                np.isfinite(
                    previous_start
                )
            )
            and garch_ito_parameter_valid(
                previous_start,
                persistence_limit=(
                    persistence_limit
                ),
            )
        ):
            starts.insert(
                0,
                previous_start.copy(),
            )

    best_result = None

    for start_idx, x0 in enumerate(
        starts
    ):

        result = minimize(
            objective,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter":
                    maxiter,
                "ftol":
                    1e-12,
                "disp":
                    False,
            },
        )

        valid_solution = (
            np.isfinite(
                result.fun
            )
            and garch_ito_parameter_valid(
                result.x,
                persistence_limit=(
                    persistence_limit
                ),
            )
        )

        if verbose:

            print(
                f"start={start_idx:03d} "
                f"success={result.success} "
                f"valid={valid_solution} "
                f"loss={result.fun:.8f} "
                f"theta={result.x}"
            )

        if valid_solution:

            if (
                best_result is None
                or result.fun
                < best_result.fun
            ):
                best_result = result

    if best_result is None:

        raise RuntimeError(
            "GARCH-Ito RV-QMLE failed to "
            "find a valid parameter estimate."
        )

    theta_hat = np.asarray(
        best_result.x,
        dtype=np.float64,
    )

    (
        omega_g_hat,
        beta_g_hat,
        gamma_hat,
    ) = garch_ito_daily_parameters(
        *theta_hat
    )

    h_train = garch_ito_filter(
        y=y_train,
        omega=theta_hat[0],
        beta_ct=theta_hat[1],
        gamma=theta_hat[2],
        initialization=initialization,
    )

    return {
        "result":
            best_result,

        "theta_hat":
            theta_hat,

        "omega_hat":
            float(
                theta_hat[0]
            ),

        "beta_ct_hat":
            float(
                theta_hat[1]
            ),

        "gamma_hat":
            float(
                theta_hat[2]
            ),

        "tau_hat":
            np.array(
                [
                    omega_g_hat,
                    beta_g_hat,
                    gamma_hat,
                ],
                dtype=np.float64,
            ),

        "omega_g_hat":
            float(
                omega_g_hat
            ),

        "beta_g_hat":
            float(
                beta_g_hat
            ),

        "persistence_hat":
            float(
                beta_g_hat
                + gamma_hat
            ),

        "h_train":
            h_train,

        "loss":
            float(
                best_result.fun
            ),

        "success":
            bool(
                best_result.success
            ),

        "message":
            str(
                best_result.message
            ),
    }


# =========================================================
# Fixed 50:50 rolling GARCH-Ito
# =========================================================

def fixed_half_sample_rolling_garch_ito(
    ticker: str,
    y_full: np.ndarray,
    rv_full: np.ndarray,
    refit_every: int = 21,
    warm_start: bool = False,
    initialization: str = "paper",
    drop_initial: bool = False,
    persistence_limit: float = 0.999,
    maxiter: int = 20000,
    verbose: bool = False,
):
    """
    SAME rolling design as standard GARCH(1,1).

    For every target t:

        forecast h_t using information through t-1

    and evaluate h_t against:

        y_t
        RV_t

    The estimation window moves only on refit dates.
    """

    y_full = np.asarray(
        y_full,
        dtype=np.float64,
    ).reshape(-1)

    rv_full = np.asarray(
        rv_full,
        dtype=np.float64,
    ).reshape(-1)

    if len(y_full) != len(rv_full):

        raise ValueError(
            "y_full and rv_full must "
            "have equal length."
        )


    valid = (
        np.isfinite(y_full)
        & np.isfinite(rv_full)
    )

    if not np.all(valid):

        print(
            f"{ticker}: WARNING - "
            f"fixed_half_sample_rolling_garch_ito "
            f"received {np.sum(~valid)} invalid "
            "return/RV pairs."
        )

    y_full = y_full[
        valid
    ]

    rv_full = np.maximum(
        rv_full[
            valid
        ],
        1e-12,
    )

    if len(y_full) < 6:

        raise ValueError(
            "The full sample is too short."
        )

    if refit_every < 1:

        raise ValueError(
            "refit_every must be at least 1."
        )

    # -----------------------------------------------------
    # EXACT same 50:50 convention as GARCH.
    # -----------------------------------------------------
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
    target_indices = []
    parameter_history = []

    fitted_theta = None

    active_window_start = None
    active_window_end = None

    h_prev = None
    y_prev_sq = None

    refit_count = 0

    print(
        "\n"
        + "=" * 90
    )

    print(
        f"{ticker}: block rolling GARCH-Ito"
    )

    print(
        "=" * 90
    )

    print(
        f"Total observations       : "
        f"{n_total}"
    )

    print(
        f"Estimation-window size   : "
        f"{window_size}"
    )

    print(
        f"Out-of-sample forecasts  : "
        f"{n_oos}"
    )

    print(
        f"Window/parameter update : "
        f"every {refit_every} trading days"
    )

    print(
        "Forecast frequency       : daily"
    )

    for (
        forecast_number,
        target_idx,
    ) in enumerate(
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
            )
            % refit_every
            == 0
        )

        refit_accepted = False
        optimizer_success = False
        optimizer_message = ""
        optimizer_nit = -1
        optimizer_nfev = -1

        candidate_loss = np.nan
        candidate_persistence = np.nan

        if should_refit:

            refit_count += 1

            # Current target excluded.
            active_window_end = (
                target_idx
            )

            active_window_start = (
                active_window_end
                - window_size
            )

            if active_window_start < 0:

                raise RuntimeError(
                    "The active estimation "
                    "window begins before "
                    "the sample."
                )

            y_window = y_full[
                active_window_start:
                active_window_end
            ]

            rv_window = rv_full[
                active_window_start:
                active_window_end
            ]

            if (
                len(y_window)
                != window_size
            ):
                raise RuntimeError(
                    "Unexpected GARCH-Ito "
                    "estimation-window length: "
                    f"{len(y_window)} rather than "
                    f"{window_size}."
                )

            if (
                len(y_window)
                != len(rv_window)
            ):
                raise RuntimeError(
                    "GARCH-Ito return and RV "
                    "windows are not aligned."
                )

            previous_theta = (
                None
                if fitted_theta is None
                else fitted_theta.copy()
            )

            try:

                fit_result = (
                    fit_garch_ito_rv(
                        y_train=y_window,
                        rv_train=rv_window,
                        initial_theta=(
                            previous_theta
                            if warm_start
                            else None
                        ),
                        initialization=(
                            initialization
                        ),
                        drop_initial=(
                            drop_initial
                        ),
                        persistence_limit=(
                            persistence_limit
                        ),
                        maxiter=maxiter,
                        verbose=verbose,
                    )
                )

                candidate_theta = (
                    np.asarray(
                        fit_result[
                            "theta_hat"
                        ],
                        dtype=np.float64,
                    ).reshape(-1)
                )

                candidate_loss = (
                    garch_ito_rv_negloglik(
                        y=y_window,
                        rv=rv_window,
                        theta=candidate_theta,
                        initialization=(
                            initialization
                        ),
                        drop_initial=(
                            drop_initial
                        ),
                        persistence_limit=(
                            persistence_limit
                        ),
                    )
                )

                (
                    _,
                    candidate_beta_g,
                    candidate_gamma,
                ) = (
                    garch_ito_daily_parameters(
                        *candidate_theta
                    )
                )

                candidate_persistence = (
                    candidate_beta_g
                    + candidate_gamma
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

                result_object = (
                    fit_result.get(
                        "result",
                        None,
                    )
                )

                if result_object is not None:

                    optimizer_nit = int(
                        getattr(
                            result_object,
                            "nit",
                            -1,
                        )
                    )

                    optimizer_nfev = int(
                        getattr(
                            result_object,
                            "nfev",
                            -1,
                        )
                    )

                candidate_valid = (
                    candidate_theta.shape
                    == (3,)
                    and np.all(
                        np.isfinite(
                            candidate_theta
                        )
                    )
                    and np.isfinite(
                        candidate_loss
                    )
                    and candidate_loss
                    < 1e49
                    and garch_ito_parameter_valid(
                        candidate_theta,
                        persistence_limit=(
                            persistence_limit
                        ),
                    )
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

                print(
                    f"Rejected GARCH-Ito refit "
                    f"at forecast "
                    f"{forecast_number}; "
                    "retained previous parameters."
                )

            else:

                raise RuntimeError(
                    f"Initial GARCH-Ito fit "
                    f"failed for {ticker}: "
                    f"{optimizer_message}"
                )

            (
                omega,
                beta_ct,
                gamma,
            ) = map(
                float,
                fitted_theta,
            )

            # -------------------------------------------------
            # Refilter exactly the active estimation window.
            # -------------------------------------------------
            h_window = garch_ito_filter(
                y=y_window,
                omega=omega,
                beta_ct=beta_ct,
                gamma=gamma,
                initialization=(
                    initialization
                ),
            )

            if (
                len(h_window) == 0
                or not np.isfinite(
                    h_window[-1]
                )
                or h_window[-1]
                <= 0.0
            ):
                raise RuntimeError(
                    "Invalid filtered GARCH-Ito "
                    "variance at the end of "
                    "the active window."
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
                f"window=["
                f"{active_window_start}, "
                f"{active_window_end - 1}] | "
                f"success="
                f"{optimizer_success} | "
                f"accepted="
                f"{refit_accepted} | "
                f"nit="
                f"{optimizer_nit} | "
                f"RV_QMLE="
                f"{candidate_loss:.6f} | "
                f"persistence="
                f"{candidate_persistence:.6f}"
            )

        if (
            fitted_theta is None
            or h_prev is None
            or y_prev_sq is None
            or active_window_start
            is None
            or active_window_end
            is None
        ):

            raise RuntimeError(
                "GARCH-Ito rolling state "
                "has not been initialized."
            )

        (
            omega,
            beta_ct,
            gamma,
        ) = map(
            float,
            fitted_theta,
        )

        (
            omega_g,
            beta_g,
            gamma_daily,
        ) = garch_ito_daily_parameters(
            omega,
            beta_ct,
            gamma,
        )

        # =================================================
        # DAILY one-step-ahead forecast
        #
        # Information available through target_idx - 1.
        # =================================================
        h_next = (
            omega_g
            + gamma_daily
            * h_prev
            + beta_g
            * y_prev_sq
        )

        h_next = max(
            float(h_next),
            1e-12,
        )

        actual_return_value = float(
            y_full[
                target_idx
            ]
        )

        actual_rv_value = float(
            rv_full[
                target_idx
            ]
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

        target_indices.append(
            int(
                target_idx
            )
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

                "candidate_loss":
                    candidate_loss,

                "candidate_persistence":
                    candidate_persistence,

                "omega":
                    omega,

                "beta_ct":
                    beta_ct,

                "gamma":
                    gamma,

                "omega_g":
                    omega_g,

                "beta_g":
                    beta_g,

                "persistence":
                    beta_g
                    + gamma_daily,

                "h_origin":
                    h_prev,

                "return_squared_origin":
                    y_prev_sq,

                "predicted_variance":
                    h_next,

                "actual_return":
                    actual_return_value,

                "actual_rv":
                    actual_rv_value,
            }
        )

        print(
            f"{ticker} forecast "
            f"{forecast_number}/"
            f"{n_oos} | "
            f"estimation_window="
            f"[{active_window_start}, "
            f"{active_window_end - 1}] | "
            f"target={target_idx} | "
            f"refit={should_refit} | "
            f"h={h_next:.6f}"
        )

        # =================================================
        # Daily state update after observing target day.
        # =================================================
        h_prev = (
            h_next
        )

        y_prev_sq = (
            actual_return_value
            ** 2
        )

    return {
        "predicted_variance":
            np.asarray(
                predicted_variance,
                dtype=np.float64,
            ),

        "actual_rv":
            np.asarray(
                actual_rv,
                dtype=np.float64,
            ),

        "actual_returns":
            np.asarray(
                actual_returns,
                dtype=np.float64,
            ),

        "target_index":
            np.asarray(
                target_indices,
                dtype=np.int64,
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
            "h_pred and rv_true must "
            "have equal length."
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


def mse_volatility(
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
            "h_pred and rv_true must "
            "have equal length."
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
            "h_pred and rv_true must "
            "have equal length."
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
            (
                rv - h
            ) ** 2
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
            "h_pred and rv_true must "
            "have equal length."
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
            "h_pred and rv_true must "
            "have equal length."
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
            "h_pred and y_true must "
            "have equal length."
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

    return float(
        np.mean(
            0.5
            * (
                np.log(
                    2.0
                    * np.pi
                )
                + np.log(h)
                + y ** 2 / h
            )
        )
    )


# =========================================================
# Summary
# =========================================================

def summarize_rolling_garch_ito(
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
            "GARCH-Ito rolling forecasts "
            "and targets are not aligned."
        )

    return {
        "n_total":
            int(
                rolling_result[
                    "n_total"
                ]
            ),

        "rolling_window_size":
            int(
                rolling_result[
                    "rolling_window_size"
                ]
            ),

        "n_oos":
            int(
                rolling_result[
                    "n_oos"
                ]
            ),

        "refit_every":
            int(
                rolling_result[
                    "refit_every"
                ]
            ),

        "window_step":
            int(
                rolling_result[
                    "window_step"
                ]
            ),

        "QLIKE_raw":
            qlike_variance(
                h_pred,
                rv_true,
            ),

        "MSE_vol_raw":
            mse_volatility(
                h_pred,
                rv_true,
            ),

        "MSE_var_raw":
            mse_variance(
                h_pred,
                rv_true,
            ),

        "MAE_vol_raw":
            mae_volatility(
                h_pred,
                rv_true,
            ),

        "MAE_var_raw":
            mae_variance(
                h_pred,
                rv_true,
            ),

        "Return_NLL":
            return_nll(
                h_pred,
                returns_true,
            ),
    }



def run_rolling_garch_ito_for_ticker(
    data_dict,
    ticker,
    refit_every=21,
    warm_start=False,
    initialization="paper",
    verbose=False,
):
    """
    Run GARCH-Ito using EXACTLY the same raw return/RV
    arrays and 50:50 convention as standard GARCH.
    """

    (
        y_train_old,
        rv_train_old,
        y_test_old,
        rv_test_old,
    ) = get_garch_ito_arrays_from_data_dict(
        data_dict=data_dict,
        ticker=ticker,
    )

    print(
        f"\n{ticker}: "
        "GARCH-Ito data construction"
    )

    print(
        f"train n = "
        f"{len(y_train_old)}"
    )

    print(
        f"test n  = "
        f"{len(y_test_old)}"
    )

    print(
        f"mean train RV = "
        f"{np.mean(rv_train_old):.8f}"
    )

    print(
        f"mean test RV  = "
        f"{np.mean(rv_test_old):.8f}"
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

    if len(y_full) != len(rv_full):

        raise RuntimeError(
            f"{ticker}: reconstructed "
            "GARCH-Ito return/RV lengths differ."
        )

    rolling_result = (
        fixed_half_sample_rolling_garch_ito(
            ticker=ticker,
            y_full=y_full,
            rv_full=rv_full,
            refit_every=refit_every,
            warm_start=warm_start,
            initialization=initialization,
            verbose=verbose,
        )
    )

    metrics = {
        "ticker":
            ticker,

        **summarize_rolling_garch_ito(
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

        "y_oos":
            rolling_result[
                "actual_returns"
            ],

        "rv_oos":
            rolling_result[
                "actual_rv"
            ],

        "target_index":
            rolling_result[
                "target_index"
            ],

        "parameter_history":
            rolling_result[
                "parameter_history"
            ],

        "metrics":
            metrics,
    }