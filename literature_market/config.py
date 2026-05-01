from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLIP_END = "2025-12-31"
RANDOM_STATE = 42
TRADING_DAYS_PER_WEEK = 5
HORIZON_WEEKS = list(range(1, 9))
TRAIN_WINDOW_MONTHS = [1, 2, 3]
TEST_PERIOD_MONTHS = 1
MIN_TRAIN_ROWS = 15
MIN_TEST_ROWS = 5
TOP_N_KEYWORDS = 10
TOP_N_SUBFIELDS = 8


@dataclass(frozen=True)
class IndustryConfig:
    key: str
    label: str
    publication_dir: Path
    publication_files: list[str]
    tickers: list[str]
    benchmark: str = "XLV"


INDUSTRIES = {
    "medical_devices": IndustryConfig(
        key="medical_devices",
        label="Medical Devices",
        publication_dir=RAW_DATA_DIR / "medical_devices",
        publication_files=["T13690.csv", "2741.csv", "2204.csv"],
        tickers=["IHI", "XHE"],
    ),
    "biotech": IndustryConfig(
        key="biotech",
        label="Biotech",
        publication_dir=RAW_DATA_DIR / "biotech",
        publication_files=["13.csv", "30.csv"],
        tickers=["XBI", "IBB"],
    ),
}

MARKET_DAILY_PATH = DATA_DIR / "market" / "ihi_xhe_xbi_ibb_daily.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

