from __future__ import annotations

import numpy as np
import pandas as pd

from literature_market.config import (
    HORIZON_WEEKS,
    RANDOM_STATE,
    TOP_N_KEYWORDS,
    TOP_N_SUBFIELDS,
    TRADING_DAYS_PER_WEEK,
)
from literature_market.data_loading import (
    AWARDS_COL,
    DATE_COL,
    FUNDERS_COL,
    IS_CORE_COL,
    KEYWORD_COL,
    SUBFIELD_COL,
    WORK_ID_COL,
)


def downsample_jan1_publications(
    publications: pd.DataFrame,
    date_col: str = DATE_COL,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    is_jan1 = (publications[date_col].dt.month == 1) & (publications[date_col].dt.day == 1)
    pub_year = publications[date_col].dt.year
    parts = [publications[~is_jan1]]
    log_rows = []
    for year, year_jan1 in publications[is_jan1].groupby(pub_year):
        non_jan1_year = publications[(pub_year == year) & ~is_jan1]
        target = 0 if non_jan1_year.empty else int(non_jan1_year.groupby(date_col).size().median())
        if len(year_jan1) > target > 0:
            keep_idx = rng.choice(year_jan1.index.values, size=target, replace=False)
            kept = year_jan1.loc[keep_idx]
        elif target == 0:
            kept = year_jan1.iloc[:0]
        else:
            kept = year_jan1
        parts.append(kept)
        log_rows.append(
            {
                "year": int(year),
                "jan1_original": len(year_jan1),
                "year_median_daily": target,
                "jan1_kept": len(kept),
            }
        )
    out = pd.concat(parts, ignore_index=True).sort_values(date_col).reset_index(drop=True)
    log = pd.DataFrame(log_rows).sort_values("year").reset_index(drop=True)
    return out, log


def build_daily_literature_features(
    publications: pd.DataFrame,
    top_n_keywords: int = TOP_N_KEYWORDS,
    top_n_subfields: int = TOP_N_SUBFIELDS,
) -> tuple[pd.DataFrame, list[str]]:
    df = publications.copy()
    df["date"] = df[DATE_COL].dt.floor("D")
    df["is_core_bool"] = (
        df[IS_CORE_COL]
        .astype("string")
        .str.lower()
        .map({"true": True, "false": False})
        .fillna(False)
        .astype(bool)
    )
    df["has_award"] = df[AWARDS_COL].notna()
    df["has_funder"] = df[FUNDERS_COL].notna()
    df["keyword_list"] = (
        df[KEYWORD_COL]
        .fillna("")
        .astype(str)
        .str.split("|")
        .apply(lambda keywords: [kw.strip() for kw in keywords if kw.strip()])
    )
    df["num_keywords"] = df["keyword_list"].apply(len)
    df[SUBFIELD_COL] = df[SUBFIELD_COL].astype("string").fillna("unknown")

    daily = (
        df.groupby("date")
        .agg(
            daily_pubs=(WORK_ID_COL, "nunique"),
            core_count=("is_core_bool", "sum"),
            award_count=("has_award", "sum"),
            funder_count=("has_funder", "sum"),
            avg_num_keywords=("num_keywords", "mean"),
        )
        .reset_index()
        .sort_values("date")
    )
    daily["core_share"] = daily["core_count"] / daily["daily_pubs"].replace(0, np.nan)
    daily["award_share"] = daily["award_count"] / daily["daily_pubs"].replace(0, np.nan)
    daily["funder_share"] = daily["funder_count"] / daily["daily_pubs"].replace(0, np.nan)

    for col in ["daily_pubs", "core_count", "award_count", "funder_count", "avg_num_keywords"]:
        for window in [7, 14, 30]:
            daily[f"{col}_{window}d_avg"] = daily[col].rolling(window, min_periods=1).mean()
    for window in [1, 7, 14]:
        daily[f"daily_pubs_growth_{window}d"] = daily["daily_pubs"].pct_change(window)
    daily["daily_pubs_zscore_30d"] = (
        (daily["daily_pubs"] - daily["daily_pubs"].rolling(30, min_periods=5).mean())
        / daily["daily_pubs"].rolling(30, min_periods=5).std(ddof=0)
    )

    kw_long = df[["date", "keyword_list"]].explode("keyword_list").rename(columns={"keyword_list": "keyword"})
    kw_long = kw_long[kw_long["keyword"].notna() & (kw_long["keyword"] != "")]
    kw_daily = kw_long.groupby(["date", "keyword"]).size().rename("kw_count").reset_index()
    kw_day_totals = kw_daily.groupby("date")["kw_count"].sum().rename("total_keyword_mentions")
    kw_daily = kw_daily.merge(kw_day_totals, on="date", how="left")
    kw_daily["kw_prob"] = kw_daily["kw_count"] / kw_daily["total_keyword_mentions"]
    kw_daily["kw_entropy_component"] = -kw_daily["kw_prob"] * np.log(kw_daily["kw_prob"])

    diversity = (
        kw_daily.groupby("date")
        .agg(
            unique_keywords=("keyword", "nunique"),
            keyword_entropy=("kw_entropy_component", "sum"),
            total_keyword_mentions=("total_keyword_mentions", "first"),
        )
        .reset_index()
    )
    top10_keywords = kw_long["keyword"].value_counts().head(10).index
    top10_per_day = (
        kw_daily[kw_daily["keyword"].isin(top10_keywords)]
        .groupby("date")["kw_count"]
        .sum()
        .rename("top10_keyword_mentions")
        .reset_index()
    )
    diversity = diversity.merge(top10_per_day, on="date", how="left")
    diversity["top10_keyword_concentration"] = (
        diversity["top10_keyword_mentions"] / diversity["total_keyword_mentions"].replace(0, np.nan)
    )
    diversity["unique_keywords_7d_avg"] = diversity["unique_keywords"].rolling(7, min_periods=1).mean()
    diversity["keyword_entropy_7d_avg"] = diversity["keyword_entropy"].rolling(7, min_periods=1).mean()

    first_seen = kw_long.groupby("keyword")["date"].min().reset_index(name="first_date")
    novelty = (
        first_seen.groupby("first_date").size().rename("new_keyword_count").reset_index()
        .rename(columns={"first_date": "date"})
    )
    novelty = novelty.merge(diversity[["date", "unique_keywords"]], on="date", how="right")
    novelty["new_keyword_count"] = novelty["new_keyword_count"].fillna(0)
    novelty["new_keyword_share"] = novelty["new_keyword_count"] / novelty["unique_keywords"].replace(0, np.nan)
    novelty["new_keyword_count_14d_avg"] = novelty["new_keyword_count"].rolling(14, min_periods=1).mean()
    novelty = novelty.drop(columns=["unique_keywords"])

    sf_daily = df.groupby(["date", SUBFIELD_COL]).size().rename("sf_count").reset_index()
    sf_day_totals = sf_daily.groupby("date")["sf_count"].sum().rename("total_sf_count")
    sf_daily = sf_daily.merge(sf_day_totals, on="date", how="left")
    sf_daily["sf_prob"] = sf_daily["sf_count"] / sf_daily["total_sf_count"]
    sf_daily["sf_entropy_component"] = -sf_daily["sf_prob"] * np.log(sf_daily["sf_prob"])
    sf_summary = (
        sf_daily.groupby("date")
        .agg(unique_subfields=(SUBFIELD_COL, "nunique"), subfield_entropy=("sf_entropy_component", "sum"))
        .reset_index()
    )
    dominant_sf = (
        sf_daily.sort_values(["date", "sf_count"], ascending=[True, False])
        .groupby("date").head(1)[["date", SUBFIELD_COL, "sf_prob"]]
        .rename(columns={SUBFIELD_COL: "dominant_subfield", "sf_prob": "dominant_subfield_share"})
    )
    sf_summary = sf_summary.merge(dominant_sf, on="date", how="left")
    dominant_subfield = sf_summary["dominant_subfield"]
    previous_dominant_subfield = dominant_subfield.shift(1)
    sf_summary["dominant_subfield_changed"] = (
        dominant_subfield.notna()
        & previous_dominant_subfield.notna()
        & (dominant_subfield != previous_dominant_subfield)
    ).astype(int)
    sf_summary = sf_summary.drop(columns=["dominant_subfield"])

    top_keywords = kw_long["keyword"].value_counts().head(top_n_keywords).index.tolist()
    kw_matrix = (
        kw_daily[kw_daily["keyword"].isin(top_keywords)]
        .pivot_table(index="date", columns="keyword", values="kw_count", aggfunc="sum", fill_value=0)
        .add_prefix("kw_count__")
        .reset_index()
    )
    top_subfields = df[SUBFIELD_COL].value_counts().head(top_n_subfields).index.tolist()
    sf_matrix = (
        sf_daily[sf_daily[SUBFIELD_COL].isin(top_subfields)]
        .pivot_table(index="date", columns=SUBFIELD_COL, values="sf_count", aggfunc="sum", fill_value=0)
        .add_prefix("sf_count__")
        .reset_index()
    )

    daily_lit = (
        daily.merge(diversity, on="date", how="left")
        .merge(novelty, on="date", how="left")
        .merge(sf_summary, on="date", how="left")
        .merge(kw_matrix, on="date", how="left")
        .merge(sf_matrix, on="date", how="left")
    )
    daily_lit = daily_lit.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    daily_lit = daily_lit.sort_values("date").reset_index(drop=True)
    return daily_lit, [column for column in daily_lit.columns if column != "date"]


def build_daily_market_features(
    market: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    df = market.sort_values(["ticker", "date"]).copy()
    df["daily_return"] = df.groupby("ticker")["adj_close"].pct_change()
    df["log_volume"] = np.log1p(df["volume"].astype(float))
    group = df.groupby("ticker", group_keys=False)
    for lag in [1, 2, 3, 5, 10]:
        df[f"return_lag_{lag}"] = group["daily_return"].shift(lag)
    for window in [5, 10, 20]:
        df[f"return_{window}d_avg"] = group["daily_return"].transform(
            lambda series, w=window: series.rolling(w, min_periods=2).mean()
        )
        df[f"return_{window}d_vol"] = group["daily_return"].transform(
            lambda series, w=window: series.rolling(w, min_periods=2).std(ddof=0)
        )
        df[f"log_volume_{window}d_avg"] = group["log_volume"].transform(
            lambda series, w=window: series.rolling(w, min_periods=2).mean()
        )

    df = df.merge(benchmark[["date", "benchmark_daily_return"]], on="date", how="left")
    df["benchmark_return_5d_avg"] = (
        df.groupby("ticker", group_keys=False)["benchmark_daily_return"]
        .transform(lambda series: series.rolling(5, min_periods=2).mean())
    )
    df["benchmark_return_20d_avg"] = (
        df.groupby("ticker", group_keys=False)["benchmark_daily_return"]
        .transform(lambda series: series.rolling(20, min_periods=2).mean())
    )
    feature_cols = [
        "daily_return",
        "return_lag_1",
        "return_lag_2",
        "return_lag_3",
        "return_lag_5",
        "return_lag_10",
        "return_5d_avg",
        "return_10d_avg",
        "return_20d_avg",
        "return_5d_vol",
        "return_10d_vol",
        "return_20d_vol",
        "log_volume",
        "log_volume_5d_avg",
        "log_volume_10d_avg",
        "log_volume_20d_avg",
        "benchmark_daily_return",
        "benchmark_return_5d_avg",
        "benchmark_return_20d_avg",
    ]
    return df, feature_cols


def build_targets(df: pd.DataFrame, horizon_weeks: list[int] = HORIZON_WEEKS) -> pd.DataFrame:
    out = df.sort_values(["ticker", "date"]).copy()
    grouped_prices = out.groupby("ticker", group_keys=False)["adj_close"]
    for horizon in horizon_weeks:
        days = TRADING_DAYS_PER_WEEK * horizon
        out[f"future_{horizon}w_return"] = grouped_prices.shift(-days) / out["adj_close"] - 1
        out[f"future_{horizon}w_up"] = (out[f"future_{horizon}w_return"] > 0).astype("Int64")
    return out

