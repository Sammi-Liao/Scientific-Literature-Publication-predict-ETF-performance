from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from literature_market.config import ARTIFACTS_DIR, INDUSTRIES, IndustryConfig
from literature_market.data_loading import load_market_data, load_publications
from literature_market.features import (
    build_daily_literature_features,
    build_daily_market_features,
    build_targets,
    downsample_jan1_publications,
)
from literature_market.granger import run_granger_tests
from literature_market.modeling import compare_feature_sets, run_experiments, select_best


def build_modeling_dataset(config: IndustryConfig) -> tuple[pd.DataFrame, list[str], list[str], pd.DataFrame, pd.DataFrame]:
    publications = load_publications(config)
    publications, jan1_log = downsample_jan1_publications(publications)
    daily_lit, literature_cols = build_daily_literature_features(publications)
    market_daily, benchmark_daily = load_market_data(config.tickers, config.benchmark)
    market_features, market_cols = build_daily_market_features(market_daily, benchmark_daily)
    modeling_df = market_features.merge(daily_lit, on="date", how="left")
    for col in literature_cols:
        modeling_df[col] = modeling_df[col].fillna(0.0)
    modeling_df = build_targets(modeling_df)
    return modeling_df, market_cols, literature_cols, jan1_log, daily_lit


def summarize_results(
    config: IndustryConfig,
    comparison: pd.DataFrame,
    granger_significant: pd.DataFrame,
) -> dict:
    overall_rows = []
    for task, sub in comparison.groupby("task"):
        n_configs = len(sub)
        n_lit_better = int((sub["lit_helps"] > 0).sum())
        overall_rows.append(
            {
                "task": task,
                "n_configs": n_configs,
                "n_lit_better": n_lit_better,
                "avg_lit_help": float(sub["lit_helps"].mean()) if n_configs else None,
                "median_lit_help": float(sub["lit_helps"].median()) if n_configs else None,
                "pct_lit_better": float(n_lit_better / n_configs * 100) if n_configs else None,
            }
        )

    by_horizon = (
        comparison.groupby(["task", "horizon_weeks"])["lit_helps"]
        .agg(avg_lit_help="mean", median_lit_help="median", n_configs="size")
        .reset_index()
    )
    return {
        "industry": config.key,
        "label": config.label,
        "tickers": config.tickers,
        "benchmark": config.benchmark,
        "overall": overall_rows,
        "by_horizon": by_horizon.to_dict(orient="records"),
        "n_granger_significant": int(len(granger_significant)),
        "top_granger": granger_significant.head(10).to_dict(orient="records"),
    }


def run_industry_pipeline(
    industry: str,
    artifacts_dir: Path = ARTIFACTS_DIR,
    print_progress: bool = False,
) -> Path:
    config = INDUSTRIES[industry]
    output_dir = artifacts_dir / industry
    output_dir.mkdir(parents=True, exist_ok=True)

    modeling_df, market_cols, literature_cols, jan1_log, daily_lit = build_modeling_dataset(config)
    feature_sets = {
        "A_market_only": market_cols,
        "B_market_plus_literature": market_cols + literature_cols,
    }
    experiment_results, fold_counts = run_experiments(
        modeling_df=modeling_df,
        feature_sets=feature_sets,
        tickers=config.tickers,
        print_progress=print_progress,
    )
    best_models = select_best(experiment_results)
    comparison = compare_feature_sets(best_models)
    granger_results, granger_significant = run_granger_tests(modeling_df, literature_cols, config.tickers)
    summary = summarize_results(config, comparison, granger_significant)

    modeling_df.to_csv(output_dir / "modeling_dataset_sample.csv", index=False)
    daily_lit.to_csv(output_dir / "daily_literature_features.csv", index=False)
    jan1_log.to_csv(output_dir / "jan1_downsampling.csv", index=False)
    experiment_results.to_csv(output_dir / "experiment_results.csv", index=False)
    fold_counts.to_csv(output_dir / "fold_counts.csv", index=False)
    best_models.to_csv(output_dir / "best_models.csv", index=False)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    granger_results.to_csv(output_dir / "granger_results.csv", index=False)
    granger_significant.to_csv(output_dir / "granger_significant_weekly.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return output_dir

