import json
import re
from pathlib import Path
import numpy as np
import pandas as pd


JSON_FOLDER = Path(__file__).parent / "json_files"
REQUIRED_COLUMNS = ["company", "date", "total", "address"]
FEATURE_COLUMNS = ["total_log", "company_deviation_log", "days_since_latest"]
VALID_YEAR_RANGE = (2020, 2030)


def load_receipts(folder: Path) -> pd.DataFrame:
    rows = []

    for path in sorted(folder.rglob("*.json")):
        with path.open(encoding="utf-8") as file:
            receipt = json.load(file)

        rows.append({**receipt, "file_name": path.name})

    receipts = pd.DataFrame(rows)

    for column in REQUIRED_COLUMNS:
        if column not in receipts:
            receipts[column] = pd.NA

    return receipts


def parse_total(value: object) -> float | None:
    if pd.isna(value):
        return None

    text = str(value).replace(" ", "").strip()
    if not text:
        return None

    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")
    elif text.count(".") > 1:
        integer, decimal = text.rsplit(".", 1)
        text = f"{integer.replace('.', '')}.{decimal}"

    try:
        return float(text)
    except ValueError:
        return None


def normalize_company(value: object) -> str | None:
    if pd.isna(value):
        return None

    company = re.sub(r"[İIıi]", "I", str(value).strip()).upper()
    company = re.sub(r"[.,]", "", company)
    company = re.sub(r"\s+", " ", company).strip()
    return company or None


df = load_receipts(JSON_FOLDER)

df["total_original"] = df["total"]
df["total"] = df["total"].apply(parse_total)

df["date_original"] = df["date"]
df["date"] = pd.to_datetime(
    df["date"],
    dayfirst=True,
    errors="coerce",
    format="mixed",
)
df["date"] = df["date"].where(
    df["date"].dt.year.between(*VALID_YEAR_RANGE)
)

df["days_since_latest"] = (df["date"].max() - df["date"]).dt.days
df["company_normalized"] = df["company"].apply(normalize_company)

company_stats = df.groupby("company_normalized")["total"].agg(["median", "size"])
company_stats.loc[company_stats["size"] < 2, "median"] = df["total"].median()
company_reference_total = df["company_normalized"].map(company_stats["median"])

df["total_log"] = np.log1p(df["total"])
df["company_deviation_log"] = np.log(
    (df["total"] + 1) / (company_reference_total + 1)
)

missing_required_data = df[REQUIRED_COLUMNS].isna().any(axis=1)
invalid_feature_data = ~np.isfinite(df[FEATURE_COLUMNS]).all(axis=1)

df["data_quality_anomaly"] = (
    missing_required_data
    | df["company_normalized"].isna()
    | invalid_feature_data
)

valid_model_mask = ~df["data_quality_anomaly"]
X = df.loc[valid_model_mask, FEATURE_COLUMNS]
