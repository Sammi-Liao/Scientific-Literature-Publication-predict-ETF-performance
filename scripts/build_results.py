from __future__ import annotations

import argparse

from literature_market.config import INDUSTRIES
from literature_market.pipeline import run_industry_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cached literature-market analytics artifacts.")
    parser.add_argument(
        "--industry",
        choices=["all", *INDUSTRIES.keys()],
        default="all",
        help="Industry to build. Use 'all' to build every configured industry.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print each successful model-fold run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    industries = INDUSTRIES.keys() if args.industry == "all" else [args.industry]
    for industry in industries:
        print(f"Building artifacts for {industry}...")
        output_dir = run_industry_pipeline(industry, print_progress=args.progress)
        print(f"Saved artifacts to {output_dir}")


if __name__ == "__main__":
    main()

