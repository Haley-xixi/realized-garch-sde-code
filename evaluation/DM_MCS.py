# -*- coding: utf-8 -*-
"""
Diebold-Mariano / Harvey-Leybourne-Newbold tests
and Model Confidence Set (MCS)

Designed for forecast-loss comparisons such as:
    - QLIKE
    - MSE
    - MAE
    - NLL

Loss convention:
    lower loss = better forecast

For DM:
    d_t = loss_model_0 - loss_model_1

Therefore:
    DM < 0  -> model 0 has lower average loss
    DM > 0  -> model 1 has lower average loss

Author: corrected implementation
"""

import numpy as np
import pandas as pd

from scipy.stats import norm, t
from collections import namedtuple
from numpy import ix_


# ============================================================
# 1. Loss functions
# ============================================================

def qlike_loss(realized_var, forecast_var, eps=1e-12):
    """
    QLIKE loss.

    Parameters
    ----------
    realized_var : array-like
        Observed realized variance.
    forecast_var : array-like
        Forecast variance.
    eps : float
        Numerical lower bound.

    Returns
    -------
    loss : np.ndarray
        Daily QLIKE losses.

    Definition
    ----------
    QLIKE_t = RV_t / h_t - log(RV_t / h_t) - 1

    Lower is better.
    """
    realized_var = np.asarray(realized_var, dtype=float)
    forecast_var = np.asarray(forecast_var, dtype=float)

    realized_var = np.maximum(realized_var, eps)
    forecast_var = np.maximum(forecast_var, eps)

    ratio = realized_var / forecast_var

    return ratio - np.log(ratio) - 1.0


def mse_loss(actual, forecast):
    """
    Daily squared-error loss.
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    return (actual - forecast) ** 2


def mae_loss(actual, forecast):
    """
    Daily absolute-error loss.
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    return np.abs(actual - forecast)


# ============================================================
# 2. Newey-West long-run variance
# ============================================================

def newey_west_long_run_variance(x, max_lag=None):
    """
    Estimate the long-run variance of a stationary series
    using the Bartlett/Newey-West estimator.

    Parameters
    ----------
    x : array-like
        Input series.
    max_lag : int or None
        HAC truncation lag.

        If None:
            floor(4 * (T / 100)^(2/9))

        This is a common automatic Newey-West rule.

    Returns
    -------
    lrv : float
        Estimated long-run variance of x_t, not of its mean.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    T = len(x)

    if T < 2:
        raise ValueError("At least 2 observations are required.")

    if max_lag is None:
        max_lag = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))

    max_lag = min(max_lag, T - 1)

    x_centered = x - np.mean(x)

    # Lag-zero autocovariance
    gamma0 = np.dot(x_centered, x_centered) / T

    lrv = gamma0

    # Bartlett weights
    for lag in range(1, max_lag + 1):

        gamma_lag = np.dot(
            x_centered[lag:],
            x_centered[:-lag]
        ) / T

        weight = 1.0 - lag / (max_lag + 1.0)

        lrv += 2.0 * weight * gamma_lag

    return lrv


# ============================================================
# 3. Diebold-Mariano test
# ============================================================

DMResult = namedtuple(
    "DMResult",
    [
        "statistic",
        "p_value",
        "mean_loss_diff",
        "mean_loss_0",
        "mean_loss_1",
        "n_obs",
        "alternative",
        "test_type",
        "hac_lags"
    ]
)


def dm_test(
    loss_0,
    loss_1,
    h=1,
    alternative="two-sided",
    test_type="HLN",
    hac_lags=None
):
    """
    Diebold-Mariano test based directly on forecast losses.

    Parameters
    ----------
    loss_0 : array-like
        Loss series from model 0.
    loss_1 : array-like
        Loss series from model 1.

    h : int, default=1
        Forecast horizon.

    alternative : {"two-sided", "less", "greater"}
        Because

            d_t = loss_0 - loss_1

        alternatives mean:

        "two-sided":
            E[d_t] != 0

        "less":
            E[d_t] < 0
            model 0 is better than model 1

        "greater":
            E[d_t] > 0
            model 1 is better than model 0

    test_type : {"DM", "HLN"}
        "DM":
            original asymptotic DM statistic

        "HLN":
            Harvey-Leybourne-Newbold small-sample correction

    hac_lags : int or None
        Number of Newey-West lags.

        Recommended for rolling volatility forecasts:
            hac_lags=None

        which uses an automatic NW bandwidth.

        If you want the classical horizon-based DM variance,
        use:
            hac_lags=h-1

        For one-step forecasts:
            hac_lags=0

    Returns
    -------
    DMResult
    """

    loss_0 = np.asarray(loss_0, dtype=float)
    loss_1 = np.asarray(loss_1, dtype=float)

    if loss_0.shape != loss_1.shape:
        raise ValueError("loss_0 and loss_1 must have the same shape.")

    if loss_0.ndim != 1:
        raise ValueError("loss arrays must be one-dimensional.")

    if not isinstance(h, int) or h < 1:
        raise ValueError("h must be a positive integer.")

    if alternative not in ["two-sided", "less", "greater"]:
        raise ValueError(
            "alternative must be 'two-sided', 'less', or 'greater'."
        )

    if test_type not in ["DM", "HLN"]:
        raise ValueError("test_type must be 'DM' or 'HLN'.")

    # Remove any pair where either observation is non-finite
    valid = np.isfinite(loss_0) & np.isfinite(loss_1)

    loss_0 = loss_0[valid]
    loss_1 = loss_1[valid]

    T = len(loss_0)

    if T <= h:
        raise ValueError(
            "Number of observations must be larger than forecast horizon."
        )

    # --------------------------------------------------------
    # Loss differential
    # --------------------------------------------------------
    # Negative d -> model 0 better
    # Positive d -> model 1 better
    d = loss_0 - loss_1

    mean_d = np.mean(d)

    # --------------------------------------------------------
    # HAC variance
    # --------------------------------------------------------
    if hac_lags is None:
        used_hac_lags = int(
            np.floor(4 * (T / 100.0) ** (2.0 / 9.0))
        )
    else:
        if not isinstance(hac_lags, int) or hac_lags < 0:
            raise ValueError("hac_lags must be a non-negative integer.")
        used_hac_lags = hac_lags

    used_hac_lags = min(used_hac_lags, T - 1)

    long_run_var = newey_west_long_run_variance(
        d,
        max_lag=used_hac_lags
    )

    if long_run_var <= 0 or not np.isfinite(long_run_var):
        raise ValueError(
            "Estimated long-run variance is non-positive. "
            "Check the loss series or HAC specification."
        )

    # Variance of sample mean
    var_mean_d = long_run_var / T

    DM_stat = mean_d / np.sqrt(var_mean_d)

    # --------------------------------------------------------
    # HLN finite-sample correction
    # --------------------------------------------------------
    if test_type == "HLN":

        correction_term = (
            T
            + 1
            - 2 * h
            + h * (h - 1) / T
        ) / T

        if correction_term <= 0:
            raise ValueError(
                "HLN correction is undefined for this T and h."
            )

        harvey_adjustment = np.sqrt(correction_term)

        statistic = harvey_adjustment * DM_stat

        distribution = "t"

    else:

        statistic = DM_stat

        distribution = "normal"

    # --------------------------------------------------------
    # p-value
    # --------------------------------------------------------
    if distribution == "t":

        if alternative == "two-sided":
            p_value = 2.0 * (
                1.0 - t.cdf(abs(statistic), df=T - 1)
            )

        elif alternative == "less":
            p_value = t.cdf(
                statistic,
                df=T - 1
            )

        else:  # greater
            p_value = 1.0 - t.cdf(
                statistic,
                df=T - 1
            )

    else:

        if alternative == "two-sided":
            p_value = 2.0 * (
                1.0 - norm.cdf(abs(statistic))
            )

        elif alternative == "less":
            p_value = norm.cdf(statistic)

        else:
            p_value = 1.0 - norm.cdf(statistic)

    return DMResult(
        statistic=float(statistic),
        p_value=float(p_value),
        mean_loss_diff=float(mean_d),
        mean_loss_0=float(np.mean(loss_0)),
        mean_loss_1=float(np.mean(loss_1)),
        n_obs=T,
        alternative=alternative,
        test_type=test_type,
        hac_lags=used_hac_lags
    )


# ============================================================
# 4. Stationary bootstrap for MCS
# ============================================================

def stationary_bootstrap_indices(T, B, block_length, rng=None):
    """
    Generate stationary-bootstrap indices.

    Parameters
    ----------
    T : int
        Number of time observations.
    B : int
        Number of bootstrap samples.
    block_length : float
        Expected block length.
    rng : np.random.Generator or None

    Returns
    -------
    indices : ndarray, shape (T, B)
    """

    if rng is None:
        rng = np.random.default_rng()

    if block_length <= 0:
        raise ValueError("block_length must be positive.")

    p = 1.0 / block_length

    indices = np.empty((T, B), dtype=int)

    # Initial observations
    indices[0, :] = rng.integers(
        low=0,
        high=T,
        size=B
    )

    for tt in range(1, T):

        restart = rng.random(B) < p

        # Restart block
        indices[tt, restart] = rng.integers(
            low=0,
            high=T,
            size=np.sum(restart)
        )

        # Continue previous block
        indices[tt, ~restart] = (
            indices[tt - 1, ~restart] + 1
        ) % T

    return indices


# ============================================================
# 5. Pairwise loss differences for MCS
# ============================================================

def compute_mcs_loss_differences(losses, bootstrap_indices):
    """
    Compute pairwise average loss differences.

    dij[i,j] = average(loss_i - loss_j)

    Positive dij:
        model i has larger loss than model j.
    """

    losses = np.asarray(losses, dtype=float)

    T, M = losses.shape
    B = bootstrap_indices.shape[1]

    # --------------------------------------------------------
    # Sample mean loss differences
    # --------------------------------------------------------
    mean_losses = losses.mean(axis=0)

    dij = (
        mean_losses[:, None]
        - mean_losses[None, :]
    )

    # --------------------------------------------------------
    # Bootstrap loss differences
    # --------------------------------------------------------
    dij_star = np.zeros((B, M, M))

    for b in range(B):

        boot_mean = losses[
            bootstrap_indices[:, b],
            :
        ].mean(axis=0)

        dij_star[b] = (
            boot_mean[:, None]
            - boot_mean[None, :]
        )

    # --------------------------------------------------------
    # Bootstrap variances
    # --------------------------------------------------------
    centered = dij_star - dij[None, :, :]

    vardij = np.mean(
        centered ** 2,
        axis=0
    )

    # Diagonal corresponds to d_ii = 0.
    # Set only for safe numerical division.
    np.fill_diagonal(vardij, 1.0)

    # Protect only truly degenerate pairwise variance
    offdiag = ~np.eye(M, dtype=bool)

    if np.any(vardij[offdiag] <= 0):
        raise ValueError(
            "Zero/non-positive bootstrap variance encountered "
            "for an off-diagonal model comparison."
        )

    return dij, dij_star, vardij


# ============================================================
# 6. MCS test statistics
# ============================================================

def mcs_pvalue_R(z_star, included, z_sample):
    """
    Range statistic T_R.
    """

    idx = np.asarray(included)

    z_boot = z_star[:, idx][:, :, idx]
    z_obs = z_sample[np.ix_(idx, idx)]

    # Exclude diagonal
    M = len(idx)

    mask = ~np.eye(M, dtype=bool)

    boot_stats = np.max(
        np.abs(z_boot[:, mask]),
        axis=1
    )

    observed_stat = np.max(
        np.abs(z_obs[mask])
    )

    return np.mean(
        boot_stats >= observed_stat
    )


def mcs_pvalue_SQ(z_star, included, z_sample):
    """
    Semi-quadratic statistic T_SQ.
    """

    idx = np.asarray(included)

    z_boot = z_star[:, idx][:, :, idx]
    z_obs = z_sample[np.ix_(idx, idx)]

    M = len(idx)
    mask = ~np.eye(M, dtype=bool)

    boot_stats = 0.5 * np.sum(
        z_boot[:, mask] ** 2,
        axis=1
    )

    observed_stat = 0.5 * np.sum(
        z_obs[mask] ** 2
    )

    return np.mean(
        boot_stats >= observed_stat
    )


# ============================================================
# 7. MCS elimination procedure
# ============================================================

def mcs_iterate(
    dij,
    dij_star,
    vardij,
    alpha=0.10,
    algorithm="R"
):
    """
    Sequential elimination procedure for MCS.

    alpha:
        significance level

    Example:
        alpha=0.10 -> 90% MCS
        alpha=0.05 -> 95% MCS
    """

    B, M, _ = dij_star.shape

    # Standardized observed pairwise loss differences
    z_sample = dij / np.sqrt(vardij)

    # Bootstrap-centered standardized statistics
    z_star = (
        dij_star - dij[None, :, :]
    ) / np.sqrt(vardij[None, :, :])

    remaining = list(range(M))

    elimination_order = []
    stage_pvalues = []

    while len(remaining) > 1:

        # ----------------------------------------------------
        # Test equal predictive ability
        # ----------------------------------------------------
        if algorithm == "R":

            pval = mcs_pvalue_R(
                z_star,
                remaining,
                z_sample
            )

        elif algorithm == "SQ":

            pval = mcs_pvalue_SQ(
                z_star,
                remaining,
                z_sample
            )

        else:
            raise ValueError(
                "algorithm must be 'R' or 'SQ'."
            )

        stage_pvalues.append(pval)

        # ----------------------------------------------------
        # Calculate average loss disadvantage
        #
        # d_i. = average_j (L_i - L_j)
        #
        # Largest standardized value = worst model
        # ----------------------------------------------------
        idx = np.asarray(remaining)

        m = len(idx)

        dij_subset = dij[np.ix_(idx, idx)]

        dij_star_subset = dij_star[:, idx][:, :, idx]

        # Average relative loss
        di_bar = (
            np.sum(dij_subset, axis=1)
            / (m - 1)
        )

        # Bootstrap version
        di_star = (
            np.sum(dij_star_subset, axis=2)
            / (m - 1)
        )

        centered_di_star = (
            di_star - di_bar[None, :]
        )

        var_di = np.mean(
            centered_di_star ** 2,
            axis=0
        )

        if np.any(var_di <= 0):
            raise ValueError(
                "Non-positive bootstrap variance in "
                "MCS elimination statistic."
            )

        t_i = di_bar / np.sqrt(var_di)

        # Worst model = largest relative loss
        worst_local = np.argmax(t_i)

        worst_model = remaining[worst_local]

        elimination_order.append(worst_model)

        remaining.remove(worst_model)

    # Last surviving model
    elimination_order.append(remaining[0])

    # MCS p-values must be monotonically non-decreasing
    stage_pvalues = np.asarray(stage_pvalues)

    if len(stage_pvalues) > 0:
        stage_pvalues = np.maximum.accumulate(
            stage_pvalues
        )

    # Determine confidence set
    #
    # Models eliminated while p < alpha are excluded.
    # Once EPA cannot be rejected, all remaining models
    # form the MCS.
    number_excluded = 0

    for pval in stage_pvalues:

        if pval < alpha:
            number_excluded += 1
        else:
            break

    excluded = elimination_order[:number_excluded]

    included = elimination_order[number_excluded:]

    return (
        np.asarray(included, dtype=int),
        np.asarray(excluded, dtype=int),
        stage_pvalues,
        elimination_order
    )


# ============================================================
# 8. User-facing MCS class
# ============================================================

class ModelConfidenceSet:

    def __init__(
        self,
        data,
        alpha=0.10,
        B=5000,
        w=None,
        algorithm="R",
        names=None,
        random_state=1313
    ):
        """
        Hansen-Lunde-Nason Model Confidence Set.

        Parameters
        ----------
        data : pd.DataFrame or np.ndarray
            Shape (T, M).

            Rows:
                forecast dates

            Columns:
                models

            Entries:
                forecast losses

            Lower loss is better.

        alpha : float
            Significance level.

            alpha=0.10 -> 90% Model Confidence Set
            alpha=0.05 -> 95% Model Confidence Set

        B : int
            Number of stationary-bootstrap samples.

        w : float or None
            Expected stationary-bootstrap block length.

            If None:
                1.5 * T^(1/3)

        algorithm : {"R", "SQ"}
            MCS test statistic.

            "R":
                range statistic

            "SQ":
                semi-quadratic statistic

        names : list-like or None
            Optional model names.

        random_state : int
            Bootstrap seed.
        """

        if isinstance(data, pd.DataFrame):

            self.data = data.to_numpy(
                dtype=float
            )

            if names is None:
                self.names = np.asarray(
                    data.columns,
                    dtype=object
                )
            else:
                self.names = np.asarray(
                    names,
                    dtype=object
                )

        else:

            self.data = np.asarray(
                data,
                dtype=float
            )

            if self.data.ndim != 2:
                raise ValueError(
                    "data must be a 2D array."
                )

            if names is None:
                self.names = np.asarray(
                    [
                        f"Model_{i}"
                        for i in range(
                            self.data.shape[1]
                        )
                    ],
                    dtype=object
                )
            else:
                self.names = np.asarray(
                    names,
                    dtype=object
                )

        T, M = self.data.shape

        if len(self.names) != M:
            raise ValueError(
                "Number of names must equal "
                "number of model columns."
            )

        if not (0 < alpha < 1):
            raise ValueError(
                "alpha must satisfy 0 < alpha < 1."
            )

        if not isinstance(B, int) or B < 1:
            raise ValueError(
                "B must be a positive integer."
            )

        if algorithm not in ["R", "SQ"]:
            raise ValueError(
                "algorithm must be 'R' or 'SQ'."
            )

        # Remove dates containing missing model losses.
        valid_rows = np.all(
            np.isfinite(self.data),
            axis=1
        )

        self.data = self.data[valid_rows]

        T = len(self.data)

        if T < 2:
            raise ValueError(
                "Too few valid observations."
            )

        if w is None:
            w = 1.5 * T ** (1.0 / 3.0)

        if w <= 0:
            raise ValueError(
                "w must be positive."
            )

        self.alpha = alpha
        self.B = B
        self.w = float(w)
        self.algorithm = algorithm
        self.random_state = random_state

        self.included = None
        self.excluded = None
        self.pvalues = None
        self.elimination_order = None

    def run(self):

        rng = np.random.default_rng(
            self.random_state
        )

        T = self.data.shape[0]

        bootstrap_indices = (
            stationary_bootstrap_indices(
                T=T,
                B=self.B,
                block_length=self.w,
                rng=rng
            )
        )

        dij, dij_star, vardij = (
            compute_mcs_loss_differences(
                self.data,
                bootstrap_indices
            )
        )

        (
            included,
            excluded,
            stage_pvalues,
            elimination_order
        ) = mcs_iterate(
            dij=dij,
            dij_star=dij_star,
            vardij=vardij,
            alpha=self.alpha,
            algorithm=self.algorithm
        )

        self.included = (
            self.names[included]
            .ravel()
            .tolist()
        )

        self.excluded = (
            self.names[excluded]
            .ravel()
            .tolist()
        )

        self.elimination_order = (
            self.names[
                np.asarray(
                    elimination_order
                )
            ]
            .ravel()
            .tolist()
        )

        # Store stage p-values
        stage_names = (
            self.elimination_order[:-1]
        )

        self.pvalues = pd.Series(
            stage_pvalues,
            index=stage_names,
            name="MCS_pvalue"
        )

        return self


# ============================================================
# 9. Pairwise DM table helper
# ============================================================

def pairwise_dm_table(
    losses,
    benchmark,
    test_type="HLN",
    alternative="two-sided",
    h=1,
    hac_lags=None
):
    """
    Compare one benchmark model against every other model.

    Parameters
    ----------
    losses : pd.DataFrame
        Daily loss matrix:
            rows    = dates
            columns = models

    benchmark : str
        Name of reference model.

    Interpretation
    --------------
    d_t =
        loss_benchmark - loss_competitor

    Therefore:

        DM < 0:
            benchmark has lower loss

        DM > 0:
            competitor has lower loss
    """

    if not isinstance(
        losses,
        pd.DataFrame
    ):
        raise TypeError(
            "losses must be a pandas DataFrame."
        )

    if benchmark not in losses.columns:
        raise ValueError(
            f"{benchmark} is not in losses."
        )

    results = []

    benchmark_loss = losses[
        benchmark
    ].to_numpy()

    for model in losses.columns:

        if model == benchmark:
            continue

        result = dm_test(
            benchmark_loss,
            losses[model].to_numpy(),
            h=h,
            alternative=alternative,
            test_type=test_type,
            hac_lags=hac_lags
        )

        results.append({
            "benchmark": benchmark,
            "competitor": model,
            "mean_loss_benchmark":
                result.mean_loss_0,
            "mean_loss_competitor":
                result.mean_loss_1,
            "mean_loss_diff":
                result.mean_loss_diff,
            "DM_stat":
                result.statistic,
            "p_value":
                result.p_value,
            "n_obs":
                result.n_obs,
            "hac_lags":
                result.hac_lags
        })

    return pd.DataFrame(results)