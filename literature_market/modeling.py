from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from literature_market.config import (
    HORIZON_WEEKS,
    MIN_TEST_ROWS,
    MIN_TRAIN_ROWS,
    RANDOM_STATE,
    TEST_PERIOD_MONTHS,
    TRAIN_WINDOW_MONTHS,
)


TASKS = ["regression", "classification"]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if not np.isfinite(y_pred).all():
        raise ValueError("Predictions contain non-finite values.")
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def get_regression_models() -> dict:
    return {
        "lasso": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=0.001, max_iter=20000, random_state=RANDOM_STATE)),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=80,
            max_depth=4,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
    }


def get_classification_models() -> dict:
    return {
        "lasso": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="l1",
                        C=1.0,
                        solver="liblinear",
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=80,
            max_depth=4,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
    }


def get_models(task: str) -> dict:
    return get_regression_models() if task == "regression" else get_classification_models()


def score(task: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if task == "regression":
        return rmse(y_true, y_pred)
    return float(accuracy_score(y_true, y_pred))


def rolling_monthly_folds(
    ticker_df: pd.DataFrame,
    horizon_weeks: int,
    train_months: int,
    test_months: int = TEST_PERIOD_MONTHS,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    dates = ticker_df["date"].dropna()
    first_month = dates.min().to_period("M") + train_months
    last_month = dates.max().to_period("M")
    folds = []
    for month in pd.period_range(first_month, last_month, freq="M"):
        test_start = month.to_timestamp()
        test_end = (month + test_months).to_timestamp() - pd.Timedelta(days=1)
        train_end = test_start - pd.Timedelta(days=horizon_weeks * 7)
        train_start = train_end - pd.DateOffset(months=train_months) + pd.Timedelta(days=1)
        folds.append((train_start, train_end, test_start, test_end))
    return folds


def fit_predict(model, x_train, y_train, x_test, task: str):
    if task == "classification" and pd.Series(y_train).nunique() < 2:
        majority = int(pd.Series(y_train).iloc[0])
        return np.full(len(x_test), majority, dtype=int)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    if not np.isfinite(np.asarray(preds, dtype=float)).all():
        raise ValueError("Predictions contain non-finite values.")
    return preds.astype(int) if task == "classification" else preds


def run_experiments(
    modeling_df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    tickers: list[str],
    print_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    experiment_rows = []
    fold_count_rows = []

    for ticker in tickers:
        ticker_df = modeling_df[modeling_df["ticker"] == ticker].sort_values("date").reset_index(drop=True).copy()
        for horizon in HORIZON_WEEKS:
            target_reg = f"future_{horizon}w_return"
            target_cls = f"future_{horizon}w_up"
            for train_months in TRAIN_WINDOW_MONTHS:
                folds = rolling_monthly_folds(ticker_df, horizon, train_months)
                for task in TASKS:
                    target_col = target_reg if task == "regression" else target_cls
                    models_template = get_models(task)
                    for feature_set, feature_cols in feature_sets.items():
                        per_model_scores: dict[str, list[float]] = {model: [] for model in models_template}
                        n_valid_folds = 0
                        for train_start, train_end, test_start, test_end in folds:
                            train_mask = (
                                (ticker_df["date"] >= train_start)
                                & (ticker_df["date"] <= train_end)
                                & ticker_df[target_col].notna()
                            )
                            test_mask = (
                                (ticker_df["date"] >= test_start)
                                & (ticker_df["date"] <= test_end)
                                & ticker_df[target_col].notna()
                            )
                            train_part = ticker_df.loc[train_mask]
                            test_part = ticker_df.loc[test_mask]
                            if len(train_part) < MIN_TRAIN_ROWS or len(test_part) < MIN_TEST_ROWS:
                                continue

                            x_train = train_part[feature_cols].replace([np.inf, -np.inf], np.nan)
                            x_test = test_part[feature_cols].replace([np.inf, -np.inf], np.nan)
                            medians = x_train.median(numeric_only=True).fillna(0.0)
                            x_train = x_train.fillna(medians)
                            x_test = x_test.fillna(medians)
                            y_train = train_part[target_col].astype(float).values
                            y_test = test_part[target_col].astype(float).values
                            if task == "classification":
                                y_train = y_train.astype(int)
                                y_test = y_test.astype(int)

                            n_valid_folds += 1
                            for model_name, estimator in models_template.items():
                                try:
                                    model = clone(estimator)
                                    preds = fit_predict(model, x_train, y_train, x_test, task)
                                    model_score = score(task, y_test, preds)
                                    per_model_scores[model_name].append(model_score)
                                    if print_progress:
                                        print(
                                            f"Ran {model_name} | ticker={ticker} | horizon={horizon}w | "
                                            f"train={train_months}m | task={task} | "
                                            f"features={feature_set} | fold={n_valid_folds} | "
                                            f"score={model_score:.6f}"
                                        )
                                except Exception as exc:
                                    if print_progress:
                                        print(f"Skipped {model_name}: {exc}")

                        fold_count_rows.append(
                            {
                                "ticker": ticker,
                                "horizon_weeks": horizon,
                                "train_months": train_months,
                                "task": task,
                                "feature_set": feature_set,
                                "n_folds": n_valid_folds,
                            }
                        )
                        for model_name, scores in per_model_scores.items():
                            if not scores:
                                continue
                            experiment_rows.append(
                                {
                                    "ticker": ticker,
                                    "horizon_weeks": horizon,
                                    "train_months": train_months,
                                    "task": task,
                                    "feature_set": feature_set,
                                    "model": model_name,
                                    "metric": "rmse" if task == "regression" else "accuracy",
                                    "mean_score": float(np.mean(scores)),
                                    "std_score": float(np.std(scores)),
                                    "n_folds": len(scores),
                                }
                            )

    return pd.DataFrame(experiment_rows), pd.DataFrame(fold_count_rows)


def select_best(experiment_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["ticker", "horizon_weeks", "train_months", "task", "feature_set"]
    for _, sub in experiment_results.groupby(keys):
        ascending = sub["task"].iloc[0] == "regression"
        rows.append(sub.sort_values("mean_score", ascending=ascending).iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def compare_feature_sets(best_models: pd.DataFrame) -> pd.DataFrame:
    pivot_keys = ["ticker", "horizon_weeks", "train_months", "task"]
    pivoted = best_models.pivot_table(
        index=pivot_keys,
        columns="feature_set",
        values=["mean_score", "model"],
        aggfunc="first",
    )
    pivoted.columns = [f"{left}__{right}" for left, right in pivoted.columns]
    comparison = pivoted.reset_index()
    comparison["lit_helps"] = np.where(
        comparison["task"] == "regression",
        comparison["mean_score__A_market_only"] - comparison["mean_score__B_market_plus_literature"],
        comparison["mean_score__B_market_plus_literature"] - comparison["mean_score__A_market_only"],
    )
    return comparison.rename(
        columns={
            "mean_score__A_market_only": "baseline_score",
            "mean_score__B_market_plus_literature": "lit_score",
            "model__A_market_only": "baseline_best_model",
            "model__B_market_plus_literature": "lit_best_model",
        }
    )[
        pivot_keys
        + ["baseline_best_model", "baseline_score", "lit_best_model", "lit_score", "lit_helps"]
    ]

