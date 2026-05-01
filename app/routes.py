from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from flask import Blueprint, jsonify, render_template, request

from literature_market.config import ARTIFACTS_DIR, INDUSTRIES


bp = Blueprint("main", __name__)


def artifact_path(industry: str, filename: str) -> Path:
    if industry not in INDUSTRIES:
        raise ValueError(f"Unknown industry: {industry}")
    return ARTIFACTS_DIR / industry / filename


def read_csv_artifact(industry: str, filename: str) -> list[dict]:
    path = artifact_path(industry, filename)
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return df.where(pd.notna(df), None).to_dict(orient="records")


@bp.get("/")
def index():
    return render_template("index.html", industries=INDUSTRIES)


@bp.get("/api/industries")
def industries():
    return jsonify(
        [
            {"key": config.key, "label": config.label, "tickers": config.tickers}
            for config in INDUSTRIES.values()
        ]
    )


@bp.get("/api/summary")
def summary():
    industry = request.args.get("industry", "medical_devices")
    path = artifact_path(industry, "summary.json")
    if not path.exists():
        return jsonify({"error": f"No cached artifacts found for {industry}. Run scripts/build_results.py first."}), 404
    return jsonify(json.loads(path.read_text(encoding="utf-8")))


@bp.get("/api/comparison")
def comparison():
    industry = request.args.get("industry", "medical_devices")
    return jsonify(read_csv_artifact(industry, "comparison.csv"))


@bp.get("/api/best-models")
def best_models():
    industry = request.args.get("industry", "medical_devices")
    return jsonify(read_csv_artifact(industry, "best_models.csv"))


@bp.get("/api/granger")
def granger():
    industry = request.args.get("industry", "medical_devices")
    significant_only = request.args.get("significant", "true").lower() == "true"
    filename = "granger_significant_weekly.csv" if significant_only else "granger_results.csv"
    return jsonify(read_csv_artifact(industry, filename))


@bp.get("/api/literature")
def literature():
    industry = request.args.get("industry", "medical_devices")
    rows = read_csv_artifact(industry, "daily_literature_features.csv")
    return jsonify(rows)

