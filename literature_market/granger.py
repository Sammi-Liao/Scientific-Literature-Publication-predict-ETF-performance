from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import grangercausalitytests

from literature_market.config import HORIZON_WEEKS, TRADING_DAYS_PER_WEEK


def run_granger_tests(
    modeling_df: pd.DataFrame,
    literature_feature_cols: list[str],
    tickers: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_lag = max(HORIZON_WEEKS) * TRADING_DAYS_PER_WEEK
    weekly_lags = [week * TRADING_DAYS_PER_WEEK for week in HORIZON_WEEKS]
    aggregate_features = [
        col for col in literature_feature_cols if not col.startswith(("kw_count__", "sf_count__"))
    ]
    rows = []
    for ticker in tickers:
        ticker_df = modeling_df[modeling_df["ticker"] == ticker].sort_values("date").copy()
        for feature in aggregate_features:
            test_df = (
                ticker_df[["daily_return", feature]]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .copy()
            )
            test_df[feature] = test_df[feature].diff()
            test_df = test_df.replace([np.inf, -np.inf], np.nan).dropna()
            if len(test_df) <= max_lag + 10 or test_df[feature].nunique() < 3:
                continue
            try:
                result = grangercausalitytests(
                    test_df[["daily_return", feature]],
                    maxlag=max_lag,
                    verbose=False,
                )
            except Exception as exc:
                rows.append(
                    {
                        "ticker": ticker,
                        "feature": feature,
                        "n_obs": len(test_df),
                        "best_lag": np.nan,
                        "min_pvalue": np.nan,
                        "best_weekly_lag": np.nan,
                        "min_weekly_pvalue": np.nan,
                        "error": str(exc),
                    }
                )
                continue

            lag_pvalues = {
                lag: float(result[lag][0]["ssr_ftest"][1])
                for lag in range(1, max_lag + 1)
            }
            weekly_lag_pvalues = {lag: lag_pvalues[lag] for lag in weekly_lags}
            best_lag = min(lag_pvalues, key=lag_pvalues.get)
            best_weekly_lag = min(weekly_lag_pvalues, key=weekly_lag_pvalues.get)
            rows.append(
                {
                    "ticker": ticker,
                    "feature": feature,
                    "n_obs": len(test_df),
                    "best_lag": best_lag,
                    "min_pvalue": lag_pvalues[best_lag],
                    "best_weekly_lag": best_weekly_lag,
                    "min_weekly_pvalue": weekly_lag_pvalues[best_weekly_lag],
                    **{f"p_lag_{lag}": pvalue for lag, pvalue in lag_pvalues.items()},
                }
            )

    granger_results = pd.DataFrame(rows)
    if granger_results.empty:
        return granger_results, granger_results.copy()

    granger_results["fdr_pvalue"] = np.nan
    valid_pvalues = granger_results["min_pvalue"].notna()
    if valid_pvalues.any():
        granger_results.loc[valid_pvalues, "fdr_pvalue"] = multipletests(
            granger_results.loc[valid_pvalues, "min_pvalue"],
            method="fdr_bh",
        )[1]

    granger_results["weekly_fdr_pvalue"] = np.nan
    valid_weekly_pvalues = granger_results["min_weekly_pvalue"].notna()
    if valid_weekly_pvalues.any():
        granger_results.loc[valid_weekly_pvalues, "weekly_fdr_pvalue"] = multipletests(
            granger_results.loc[valid_weekly_pvalues, "min_weekly_pvalue"],
            method="fdr_bh",
        )[1]

    granger_results = granger_results.sort_values(
        ["weekly_fdr_pvalue", "min_weekly_pvalue", "fdr_pvalue", "min_pvalue", "ticker", "feature"]
    ).reset_index(drop=True)
    granger_significant = granger_results[
        (granger_results["min_weekly_pvalue"] < 0.05)
        & (granger_results["weekly_fdr_pvalue"] < 0.05)
    ].copy()
    return granger_results, granger_significant

