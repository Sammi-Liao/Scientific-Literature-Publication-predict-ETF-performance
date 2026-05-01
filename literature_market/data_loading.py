from __future__ import annotations

from pathlib import Path

import pandas as pd

from literature_market.config import CLIP_END, IndustryConfig, MARKET_DAILY_PATH


WORK_ID_COL = "id (start with W)"
DATE_COL = "publication_date"
KEYWORD_COL = "keywords id (after https://openalex.org/keywords/)"
TOPIC_COL = "primary_topic id (start with T)"
SUBFIELD_COL = "primary_topic subfield id (four digit num)"
FIELD_COL = "primary_topic field id (two digit num)"
IS_CORE_COL = "is_core"
AWARDS_COL = "awards"
FUNDERS_COL = "funders"


def load_publications(config: IndustryConfig) -> pd.DataFrame:
    frames = [pd.read_csv(config.publication_dir / filename) for filename in config.publication_files]
    publications = pd.concat(frames, ignore_index=True)
    publications[DATE_COL] = pd.to_datetime(publications[DATE_COL], errors="coerce")
    clip_end = pd.Timestamp(CLIP_END)
    return (
        publications.dropna(subset=[DATE_COL])
        .loc[lambda df: df[DATE_COL] <= clip_end]
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )


def load_market_data(
    tickers: list[str],
    benchmark: str,
    path: Path = MARKET_DAILY_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path, header=[0, 1])
    raw.columns = [
        "date" if (lvl0 == "Ticker" and lvl1 == "Price") else f"{lvl0}_{lvl1}"
        for lvl0, lvl1 in raw.columns
    ]
    raw = raw[raw["date"] != "Date"].copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"])
    raw = raw[raw["date"] <= pd.Timestamp(CLIP_END)].sort_values("date").reset_index(drop=True)

    for ticker in tickers + [benchmark]:
        raw[f"{ticker}_Adj Close"] = pd.to_numeric(raw[f"{ticker}_Adj Close"], errors="coerce")
        raw[f"{ticker}_Volume"] = pd.to_numeric(raw[f"{ticker}_Volume"], errors="coerce")

    market_daily = pd.concat(
        [
            raw[["date", f"{ticker}_Adj Close", f"{ticker}_Volume"]]
            .rename(columns={f"{ticker}_Adj Close": "adj_close", f"{ticker}_Volume": "volume"})
            .assign(ticker=ticker)
            for ticker in tickers
        ],
        ignore_index=True,
    ).sort_values(["ticker", "date"]).reset_index(drop=True)

    benchmark_daily = (
        raw[["date", f"{benchmark}_Adj Close"]]
        .rename(columns={f"{benchmark}_Adj Close": "benchmark_adj_close"})
        .copy()
    )
    benchmark_daily["benchmark_daily_return"] = benchmark_daily["benchmark_adj_close"].pct_change()
    return market_daily, benchmark_daily

