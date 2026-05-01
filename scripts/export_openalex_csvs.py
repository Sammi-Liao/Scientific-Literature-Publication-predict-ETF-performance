#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


COLUMNS = [
    "id (start with W)",
    "title",
    "publication_year",
    "publication_date",
    "language",
    "is_core",
    "cited_by_count",
    "citation_normalized_percentile value",
    "is_in_top_1_percent",
    "is_in_top_10_percent",
    "primary_topic id (start with T)",
    "primary_topic subfield id (four digit num)",
    "primary_topic field id (two digit num)",
    "keywords id (after https://openalex.org/keywords/)",
    "awards",
    "funders",
]


def suffix_from_url(value: str) -> str:
    if not value:
        return ""
    return str(value).rstrip("/").split("/")[-1]


def normalize_list(values):
    if not values:
        return ""
    return "|".join(str(v) for v in values if v)


def parse_awards(awards):
    if not isinstance(awards, list):
        return ""
    out = []
    for award in awards:
        if not isinstance(award, dict):
            continue
        if award.get("funder_award_id"):
            out.append(award["funder_award_id"])
        elif award.get("id"):
            out.append(suffix_from_url(award["id"]))
    return normalize_list(out)


def parse_funders(funders):
    if not isinstance(funders, list):
        return ""
    ids = []
    for funder in funders:
        if isinstance(funder, dict) and funder.get("id"):
            ids.append(suffix_from_url(funder["id"]))
    return normalize_list(ids)


def parse_keywords(keywords):
    if not isinstance(keywords, list):
        return ""
    ids = []
    for kw in keywords:
        if isinstance(kw, dict) and kw.get("id"):
            ids.append(suffix_from_url(kw["id"]))
    return normalize_list(ids)


def row_from_json(data):
    cnp = data.get("citation_normalized_percentile") or {}
    primary_topic = data.get("primary_topic") or {}
    subfield = primary_topic.get("subfield") or {}
    field = primary_topic.get("field") or {}

    source = ((data.get("primary_location") or {}).get("source")) or {}
    is_core = source.get("is_core", "")

    return {
        "id (start with W)": suffix_from_url(data.get("id", "")),
        "title": data.get("title", ""),
        "publication_year": data.get("publication_year", ""),
        "publication_date": data.get("publication_date", ""),
        "language": data.get("language", ""),
        "is_core": is_core,
        "cited_by_count": data.get("cited_by_count", ""),
        "citation_normalized_percentile value": cnp.get("value", ""),
        "is_in_top_1_percent": cnp.get("is_in_top_1_percent", ""),
        "is_in_top_10_percent": cnp.get("is_in_top_10_percent", ""),
        "primary_topic id (start with T)": suffix_from_url(primary_topic.get("id", "")),
        "primary_topic subfield id (four digit num)": suffix_from_url(subfield.get("id", "")),
        "primary_topic field id (two digit num)": suffix_from_url(field.get("id", "")),
        "keywords id (after https://openalex.org/keywords/)": parse_keywords(data.get("keywords")),
        "awards": parse_awards(data.get("awards")),
        "funders": parse_funders(data.get("funders")),
    }


def export_folder(folder_path: Path, output_csv: Path):
    json_files = sorted(folder_path.glob("*.json"))
    with output_csv.open("w", newline="", encoding="utf-8") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=COLUMNS)
        writer.writeheader()
        for json_file in json_files:
            try:
                payload = json.loads(json_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            writer.writerow(row_from_json(payload))
    return len(json_files)


def main():
    parser = argparse.ArgumentParser(
        description="Export OpenAlex JSON folders into per-folder CSV files."
    )
    parser.add_argument(
        "--base-dir",
        default="data/raw/medical_devices",
        help="Base directory containing folders of OpenAlex JSON files (default: data/raw/medical_devices).",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=["2204", "2741", "T13690"],
        help="Folder names inside --base-dir to export (default: 2204 2741 T13690).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    base_dir = (project_root / args.base_dir).resolve()

    if not base_dir.exists():
        raise SystemExit(f"Base directory does not exist: {base_dir}")

    for folder in args.folders:
        folder_path = base_dir / folder
        if not folder_path.exists():
            print(f"Skipping missing folder: {folder_path}")
            continue
        output_csv = base_dir / f"{folder}.csv"
        count = export_folder(folder_path, output_csv)
        print(f"Wrote {output_csv} ({count} JSON files scanned)")


if __name__ == "__main__":
    main()
