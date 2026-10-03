import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Project setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"


# ============================================================
# DM / MCS implementation
# ============================================================

from DM_MCS import (
    ModelConfidenceSet,
    pairwise_dm_table,
)


# ============================================================
# Configuration
# ============================================================

SEED = 1313
EPS = 1e-8

DM_TEST_TYPE = "DM"
DM_ALTERNATIVE = "two-sided"
FORECAST_HORIZON = 1
DM_HAC_LAGS = None

# 90% Model Confidence Set
MCS_ALPHA = 0.10
MCS_BOOTSTRAPS = 5000
MCS_ALGORITHM = "R"

TARGET_RTOL = 1e-10
TARGET_ATOL = 1e-10

METRICS = [
    "QLIKE",
    "MSE_var",
    "MAE_var",
]

MODEL_DIRS = {
    'Full-SDE': RESULTS_DIR / 'proposed_SDE',
    'Gamma-zero': RESULTS_DIR / 'ablation_gamma0'
}
MODEL_ORDER = [
    'Full-SDE',
    'Gamma-zero'
]

DM_COMPARISONS = [
    ('Full-SDE', 'Gamma-zero', 'Full_vs_GammaZero')
]

TICKERS = [
    'SPX',
    'SSMI',
    'STI',
    'OMXSPI',
    'KS11',
    'OMXHPI',
    'SSEC',
    'KSE',
    'MXX',
    'IXIC',
    'OMXC20',
    'STOXX50E',
    'SMSI',
    'N225',
    'OSEAX',
    'RUT',
    'NSEI'
]


# ============================================================
# Output directories
# ============================================================

OUTPUT_DIR = RESULTS_DIR / 'statistical_test_ablation_gamma'
LOSS_DIR = OUTPUT_DIR / "daily_losses"
DM_DIR = OUTPUT_DIR / "dm"
MCS_DIR = OUTPUT_DIR / "mcs"
DIAGNOSTIC_DIR = OUTPUT_DIR / "diagnostics"

for directory in [
    OUTPUT_DIR,
    LOSS_DIR,
    DM_DIR,
    MCS_DIR,
    DIAGNOSTIC_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# Utilities
# ============================================================

def ensure_1d_float(values, *, name="array"):
    """Convert input to a finite float64 one-dimensional array."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)

    if len(x) == 0:
        raise ValueError(f"{name} is empty.")

    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values.")

    return x


def load_array(path, *, name=None):
    """Load and validate one .npy array."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return ensure_1d_float(
        np.load(path, allow_pickle=False),
        name=str(path) if name is None else name,
    )


# ============================================================
# Daily loss functions
# ============================================================

def qlike_loss(realized_var, forecast_var, eps=EPS):
    """
    Daily QLIKE:
        L_t = log(h_t) + RV_t / h_t
    """
    rv = ensure_1d_float(realized_var, name="realized variance")
    h = ensure_1d_float(forecast_var, name="forecast variance")

    if len(rv) != len(h):
        raise ValueError(
            "realized_var and forecast_var must have equal length."
        )

    rv = np.maximum(rv, eps)
    h = np.maximum(h, eps)
    return np.log(h) + rv / h


def mse_var_loss(realized_var, forecast_var):
    """Daily variance squared-error loss."""
    rv = ensure_1d_float(realized_var, name="realized variance")
    h = ensure_1d_float(forecast_var, name="forecast variance")

    if len(rv) != len(h):
        raise ValueError(
            "realized_var and forecast_var must have equal length."
        )

    return (rv - h) ** 2


def mae_var_loss(realized_var, forecast_var):
    """Daily variance absolute-error loss."""
    rv = ensure_1d_float(realized_var, name="realized variance")
    h = ensure_1d_float(forecast_var, name="forecast variance")

    if len(rv) != len(h):
        raise ValueError(
            "realized_var and forecast_var must have equal length."
        )

    return np.abs(rv - h)


# ============================================================
# Load forecasts
# ============================================================

def load_model_forecasts(ticker, model_name):
    """
    Load only the arrays required for variance-forecast evaluation.

    Required files:
        {ticker}_rolling_variance_pred.npy
        {ticker}_rolling_rv_target.npy
    """
    model_dir = MODEL_DIRS[model_name]

    return {
        "pred_var": load_array(
            model_dir / f"{ticker}_rolling_variance_pred.npy",
            name=f"{ticker} {model_name} predicted variance",
        ),
        "rv_true": load_array(
            model_dir / f"{ticker}_rolling_rv_target.npy",
            name=f"{ticker} {model_name} RV target",
        ),
    }


def load_all_models(ticker):
    return {
        model_name: load_model_forecasts(ticker, model_name)
        for model_name in MODEL_ORDER
    }


# ============================================================
# Strict controlled-ablation alignment
# ============================================================

def load_and_validate_models(ticker):
    """Require identical OOS length and RV targets for all models."""
    models = load_all_models(ticker)

    # Internal consistency within each model.
    for model_name, model in models.items():
        lengths = {
            len(model["pred_var"]),
            len(model["rv_true"]),
        }

        if len(lengths) != 1:
            raise ValueError(
                f"{ticker}: {model_name} arrays have different lengths: "
                f"{sorted(lengths)}."
            )

        if np.any(model["pred_var"] <= 0):
            raise ValueError(
                f"{ticker}: {model_name} contains non-positive "
                "variance forecasts."
            )

    reference_name = MODEL_ORDER[0]
    reference = models[reference_name]
    common_rv = reference["rv_true"]
    n = len(reference["pred_var"])

    # All models must have exactly the same OOS target sequence.
    for model_name, model in models.items():
        if len(model["pred_var"]) != n:
            raise ValueError(
                f"{ticker}: OOS length mismatch. "
                f"{reference_name}={n}, "
                f"{model_name}={len(model['pred_var'])}."
            )

        rv_equal = np.allclose(
            common_rv,
            model["rv_true"],
            rtol=TARGET_RTOL,
            atol=TARGET_ATOL,
        )

        if not rv_equal:
            rv_max_diff = float(
                np.max(np.abs(common_rv - model["rv_true"]))
            )
            raise ValueError(
                f"{ticker}: {reference_name} and {model_name} "
                f"RV targets differ. max_abs_diff={rv_max_diff:.6e}."
            )

    # Diagnostics.
    diagnostic_rows = []
    for model_name, model in models.items():
        diagnostic_rows.append(
            {
                "ticker": ticker,
                "model": model_name,
                "n_oos": n,
                "mean_RV": float(np.mean(common_rv)),
                "mean_predicted_variance": float(
                    np.mean(model["pred_var"])
                ),
            }
        )

    # Add pairwise forecast-difference diagnostics for every DM pair.
    for model_a, model_b, comparison_name in DM_COMPARISONS:
        diff = models[model_a]["pred_var"] - models[model_b]["pred_var"]
        diagnostic_rows.append(
            {
                "ticker": ticker,
                "model": comparison_name,
                "n_oos": n,
                "mean_RV": float(np.mean(common_rv)),
                "mean_predicted_variance": np.nan,
                "mean_abs_forecast_difference": float(
                    np.mean(np.abs(diff))
                ),
                "rmse_forecast_difference": float(
                    np.sqrt(np.mean(diff ** 2))
                ),
                "max_abs_forecast_difference": float(
                    np.max(np.abs(diff))
                ),
                "forecast_correlation": float(
                    np.corrcoef(
                        models[model_a]["pred_var"],
                        models[model_b]["pred_var"],
                    )[0, 1]
                ),
            }
        )

    diagnostics = pd.DataFrame(diagnostic_rows)

    print("\n" + "=" * 100)
    print(f"{ticker}: CONTROLLED ABLATION CHECK")
    print("=" * 100)
    print(f"OOS observations : {n}")
    print(f"Mean RV          : {np.mean(common_rv):.8f}")
    for model_name, model in models.items():
        print(
            f"Mean variance forecast [{model_name:24s}]: "
            f"{np.mean(model['pred_var']):.8f}"
        )

    return models, diagnostics


# ============================================================
# Construct daily loss matrices
# ============================================================

def construct_loss_dfs(ticker):
    models, diagnostics = load_and_validate_models(ticker)
    common_rv = models[MODEL_ORDER[0]]["rv_true"]

    qlike_df = pd.DataFrame(
        {
            model_name: qlike_loss(
                common_rv,
                models[model_name]["pred_var"],
            )
            for model_name in MODEL_ORDER
        }
    )

    mse_var_df = pd.DataFrame(
        {
            model_name: mse_var_loss(
                common_rv,
                models[model_name]["pred_var"],
            )
            for model_name in MODEL_ORDER
        }
    )

    mae_var_df = pd.DataFrame(
        {
            model_name: mae_var_loss(
                common_rv,
                models[model_name]["pred_var"],
            )
            for model_name in MODEL_ORDER
        }
    )

    metric_dfs = {
        "QLIKE": qlike_df,
        "MSE_var": mse_var_df,
        "MAE_var": mae_var_df,
    }

    n = len(common_rv)
    for metric_name, loss_df in metric_dfs.items():
        if len(loss_df) != n:
            raise RuntimeError(
                f"{ticker}: {metric_name} loss length "
                f"{len(loss_df)} != {n}."
            )

        if list(loss_df.columns) != MODEL_ORDER:
            raise RuntimeError(
                f"{ticker}: {metric_name} model ordering is incorrect."
            )

        if not np.all(np.isfinite(loss_df.to_numpy())):
            raise ValueError(
                f"{ticker}: {metric_name} contains non-finite losses."
            )

        print(f"\nMean {metric_name}:")
        print(loss_df.mean())

    metric_dfs["diagnostics"] = diagnostics
    return metric_dfs


# ============================================================
# Pairwise DM
# ============================================================

def run_dm_pair(
    ticker,
    metric_name,
    loss_df,
    model_a,
    model_b,
    comparison_name,
):
    pair_losses = loss_df[[model_a, model_b]].copy()

    dm_df = pairwise_dm_table(
        losses=pair_losses,
        benchmark=model_a,
        test_type=DM_TEST_TYPE,
        alternative=DM_ALTERNATIVE,
        h=FORECAST_HORIZON,
        hac_lags=DM_HAC_LAGS,
    )

    if len(dm_df) != 1:
        raise RuntimeError(
            f"{ticker}: expected one DM row for {metric_name}, "
            f"{comparison_name}; received {len(dm_df)}."
        )

    dm_df = dm_df.copy()
    dm_df.insert(0, "comparison", comparison_name)
    dm_df.insert(0, "metric", metric_name)
    dm_df.insert(0, "ticker", ticker)

    dm_df["model_A"] = model_a
    dm_df["model_B"] = model_b
    dm_df["mean_loss_A"] = dm_df["mean_loss_benchmark"]
    dm_df["mean_loss_B"] = dm_df["mean_loss_competitor"]
    dm_df["mean_loss_diff_A_minus_B"] = dm_df["mean_loss_diff"]

    diff = dm_df["mean_loss_diff_A_minus_B"]

    dm_df["better_mean_loss"] = np.where(
        diff < 0,
        model_a,
        np.where(diff > 0, model_b, "Equal"),
    )

    dm_df["A_significantly_better_5pct"] = (
        (dm_df["p_value"] < 0.05) & (diff < 0)
    )
    dm_df["B_significantly_better_5pct"] = (
        (dm_df["p_value"] < 0.05) & (diff > 0)
    )
    dm_df["A_significantly_better_10pct"] = (
        (dm_df["p_value"] < 0.10) & (diff < 0)
    )
    dm_df["B_significantly_better_10pct"] = (
        (dm_df["p_value"] < 0.10) & (diff > 0)
    )

    return dm_df


# ============================================================
# MCS
# ============================================================

def run_mcs(loss_df):
    ordered_losses = loss_df[MODEL_ORDER].copy()

    mcs = ModelConfidenceSet(
        data=ordered_losses,
        alpha=MCS_ALPHA,
        B=MCS_BOOTSTRAPS,
        w=None,
        algorithm=MCS_ALGORITHM,
        random_state=SEED,
    )
    mcs.run()
    return mcs


def make_mcs_summary(ticker, metric, loss_df, mcs):
    included_models = set(mcs.included)

    rows = []
    for model_name in MODEL_ORDER:
        rows.append(
            {
                "ticker": ticker,
                "metric": metric,
                "model": model_name,
                "mean_loss": float(loss_df[model_name].mean()),
                "included_90pct_MCS": model_name in included_models,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():
    np.random.seed(SEED)

    all_dm_results = []
    all_mcs_results = []
    all_diagnostics = []
    successful_tickers = []
    failed_tickers = []

    print("=" * 100)
    print('REALIZED-MEASURE FEEDBACK ABLATION: DM + MCS')
    print("=" * 100)
    print("Metrics:", ", ".join(METRICS))
    print("Return NLL: excluded")
    print("MCS confidence level: 90%")

    for ticker in TICKERS:
        print("\n" + "#" * 100)
        print(f"Processing: {ticker}")
        print("#" * 100)

        try:
            result = construct_loss_dfs(ticker)
            all_diagnostics.append(result["diagnostics"])

            metric_dfs = {
                metric_name: result[metric_name]
                for metric_name in METRICS
            }

            for metric_name, loss_df in metric_dfs.items():
                print("\n" + "=" * 100)
                print(f"{ticker}: {metric_name}")
                print("=" * 100)

                loss_df.to_csv(
                    LOSS_DIR
                    / f"{ticker}_{metric_name.lower()}_losses.csv",
                    index=False,
                )

                for model_a, model_b, comparison_name in DM_COMPARISONS:
                    dm_df = run_dm_pair(
                        ticker=ticker,
                        metric_name=metric_name,
                        loss_df=loss_df,
                        model_a=model_a,
                        model_b=model_b,
                        comparison_name=comparison_name,
                    )
                    all_dm_results.append(dm_df)

                    dm_df.to_csv(
                        DM_DIR
                        / (
                            f"{ticker}_{metric_name.lower()}_"
                            f"{comparison_name}.csv"
                        ),
                        index=False,
                    )

                    print("\nDM:")
                    print(
                        dm_df[
                            [
                                "model_A",
                                "model_B",
                                "mean_loss_A",
                                "mean_loss_B",
                                "mean_loss_diff_A_minus_B",
                                "DM_stat",
                                "p_value",
                                "hac_lags",
                                "better_mean_loss",
                                "A_significantly_better_5pct",
                                "B_significantly_better_5pct",
                            ]
                        ].to_string(index=False)
                    )

                mcs = run_mcs(loss_df)
                mcs_summary = make_mcs_summary(
                    ticker=ticker,
                    metric=metric_name,
                    loss_df=loss_df,
                    mcs=mcs,
                )
                all_mcs_results.append(mcs_summary)

                mcs_summary.to_csv(
                    MCS_DIR
                    / f"{ticker}_{metric_name.lower()}_mcs.csv",
                    index=False,
                )

                print("\n90% MCS:")
                print(mcs_summary.to_string(index=False))

            successful_tickers.append(ticker)

        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            failed_tickers.append(
                {"ticker": ticker, "error": error_message}
            )
            print(f"\nFAILED: {ticker}")
            print(error_message)

    # --------------------------------------------------------
    # Combined DM outputs
    # --------------------------------------------------------
    if all_dm_results:
        combined_dm = pd.concat(all_dm_results, ignore_index=True)
        combined_dm.to_csv(
            OUTPUT_DIR / "all_tickers_dm.csv",
            index=False,
        )

        dm_summary_columns = [
            "ticker",
            "metric",
            "comparison",
            "model_A",
            "model_B",
            "mean_loss_A",
            "mean_loss_B",
            "mean_loss_diff_A_minus_B",
            "DM_stat",
            "p_value",
            "n_obs",
            "hac_lags",
            "better_mean_loss",
            "A_significantly_better_5pct",
            "B_significantly_better_5pct",
            "A_significantly_better_10pct",
            "B_significantly_better_10pct",
        ]
        available_columns = [
            c for c in dm_summary_columns
            if c in combined_dm.columns
        ]
        dm_summary = combined_dm[available_columns].copy()
        dm_summary.to_csv(
            OUTPUT_DIR / "dm_summary.csv",
            index=False,
        )

        for metric_name in METRICS:
            metric_dm = combined_dm[
                combined_dm["metric"] == metric_name
            ].copy()
            metric_dm.to_csv(
                OUTPUT_DIR / f"all_tickers_dm_{metric_name.lower()}.csv",
                index=False,
            )

        count_rows = []
        for metric_name in METRICS:
            metric_subset = dm_summary[
                dm_summary["metric"] == metric_name
            ]

            for model_a, model_b, comparison_name in DM_COMPARISONS:
                subset = metric_subset[
                    metric_subset["comparison"] == comparison_name
                ]

                count_rows.append(
                    {
                        "metric": metric_name,
                        "comparison": comparison_name,
                        "model_A": model_a,
                        "model_B": model_b,
                        "n_tickers": len(subset),
                        "A_lower_mean_loss": int(
                            (subset["mean_loss_diff_A_minus_B"] < 0).sum()
                        ),
                        "B_lower_mean_loss": int(
                            (subset["mean_loss_diff_A_minus_B"] > 0).sum()
                        ),
                        "A_significant_5pct": int(
                            subset["A_significantly_better_5pct"].sum()
                        ),
                        "B_significant_5pct": int(
                            subset["B_significantly_better_5pct"].sum()
                        ),
                        "A_significant_10pct": int(
                            subset["A_significantly_better_10pct"].sum()
                        ),
                        "B_significant_10pct": int(
                            subset["B_significantly_better_10pct"].sum()
                        ),
                    }
                )

        count_df = pd.DataFrame(count_rows)
        count_df.to_csv(
            OUTPUT_DIR / "dm_counts.csv",
            index=False,
        )

        print("\n" + "=" * 100)
        print("AGGREGATE DM SUMMARY")
        print("=" * 100)
        print(count_df.to_string(index=False))

    # --------------------------------------------------------
    # Combined MCS outputs
    # --------------------------------------------------------
    if all_mcs_results:
        combined_mcs = pd.concat(all_mcs_results, ignore_index=True)
        combined_mcs.to_csv(
            OUTPUT_DIR / "all_tickers_mcs.csv",
            index=False,
        )

        for metric_name in METRICS:
            metric_mcs = combined_mcs[
                combined_mcs["metric"] == metric_name
            ].copy()
            metric_mcs.to_csv(
                OUTPUT_DIR / f"all_tickers_mcs_{metric_name.lower()}.csv",
                index=False,
            )

        mcs_counts = (
            combined_mcs
            .groupby(["metric", "model"], as_index=False)
            .agg(
                n_tickers=("ticker", "nunique"),
                n_included_90pct_MCS=("included_90pct_MCS", "sum"),
                mean_loss_across_tickers=("mean_loss", "mean"),
            )
        )
        mcs_counts["inclusion_rate"] = (
            mcs_counts["n_included_90pct_MCS"]
            / mcs_counts["n_tickers"]
        )
        mcs_counts.to_csv(
            OUTPUT_DIR / "mcs_counts.csv",
            index=False,
        )

        print("\n" + "=" * 100)
        print("AGGREGATE MCS SUMMARY")
        print("=" * 100)
        print(mcs_counts.to_string(index=False))

    # --------------------------------------------------------
    # Diagnostics and failure log
    # --------------------------------------------------------
    if all_diagnostics:
        pd.concat(all_diagnostics, ignore_index=True).to_csv(
            DIAGNOSTIC_DIR / "all_tickers_diagnostics.csv",
            index=False,
        )

    if failed_tickers:
        pd.DataFrame(failed_tickers).to_csv(
            OUTPUT_DIR / "failed_tickers.csv",
            index=False,
        )

    print("\n" + "=" * 100)
    print("ABLATION DM/MCS COMPLETE")
    print("=" * 100)
    print(f"Successful tickers: {len(successful_tickers)}")
    print(successful_tickers)
    print(f"Failed tickers: {len(failed_tickers)}")
    if failed_tickers:
        print(pd.DataFrame(failed_tickers).to_string(index=False))
    print("\nResults saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
