from pathlib import Path

import pandas as pd
import yfinance as yf


TICKERS = ["IHI", "XHE", "XBI", "IBB"]
BENCHMARK_TICKER = "XLV"
RETURN_HORIZONS_WEEKS = [1, 2, 4, 8, 12]
START_DATE = "2020-01-01"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "market"


def download_market_data(tickers: list[str], start_date: str) -> pd.DataFrame:
    """Download daily OHLCV market data from Yahoo Finance."""
    data = yf.download(
        tickers=tickers,
        start=start_date,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
    )

    if data.empty:
        raise RuntimeError("No market data returned from Yahoo Finance.")

    return data


def build_weekly_returns(
    daily_data: pd.DataFrame,
    tickers: list[str],
    benchmark_ticker: str,
    horizons: list[int],
) -> pd.DataFrame:
    """Convert daily adjusted close prices into weekly returns."""
    weekly_frames = []
    benchmark_weekly_price = daily_data[benchmark_ticker]["Adj Close"].dropna().resample("W-FRI").last()

    for ticker in tickers:
        adj_close = daily_data[ticker]["Adj Close"].dropna()
        weekly_price = adj_close.resample("W-FRI").last()
        weekly_return = weekly_price.pct_change()
        weekly_frame = pd.DataFrame(
            {
                "week": weekly_price.index,
                "ticker": ticker,
                "adj_close": weekly_price.values,
                "weekly_return": weekly_return.values,
            }
        )

        for horizon in horizons:
            future_return = weekly_price.shift(-horizon) / weekly_price - 1
            benchmark_future_return = (
                benchmark_weekly_price.shift(-horizon) / benchmark_weekly_price - 1
            )
            weekly_frame[f"future_{horizon}w_return"] = future_return.values
            weekly_frame[f"future_{horizon}w_excess_return"] = (
                future_return - benchmark_future_return.reindex(weekly_price.index)
            ).values

        weekly_frames.append(weekly_frame)

    return pd.concat(weekly_frames, ignore_index=True)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    download_tickers = TICKERS + [BENCHMARK_TICKER]
    daily_data = download_market_data(download_tickers, START_DATE)
    weekly_returns = build_weekly_returns(
        daily_data,
        TICKERS,
        BENCHMARK_TICKER,
        RETURN_HORIZONS_WEEKS,
    )

    daily_data.to_csv(OUTPUT_DIR / "ihi_xhe_xbi_ibb_daily.csv")
    weekly_returns.to_csv(OUTPUT_DIR / "ihi_xhe_xbi_ibb_weekly_returns.csv", index=False)

    print(f"Saved daily data to {OUTPUT_DIR / 'ihi_xhe_xbi_ibb_daily.csv'}")
    print(f"Saved weekly returns to {OUTPUT_DIR / 'ihi_xhe_xbi_ibb_weekly_returns.csv'}")
    print(weekly_returns.tail())


if __name__ == "__main__":
    main()
