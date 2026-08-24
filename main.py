import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


JSON_FOLDER = Path(__file__).parent / "json_files"
REQUIRED_COLUMNS = ["fatura_no", "cift_grup_id", "aciklama_kategorisi", "onay_durumu"]
FEATURE_COLUMNS = ["grup_buyuklugu", "aciklama_risk", "onay_risk"]

# Sıralamalar hem anlamsal olarak hem de gözlemlenen anomali oranlarıyla tutarlı:
# aciklama_kategorisi -> yeterli %5.9, yetersiz %31.6, ai_uretimi/manipulatif ~%67
# onay_durumu -> onaylandi %1.9, gozden_gecirilecek %12.7, onaylanmadi %100
ACIKLAMA_RISK_SIRASI = {"yeterli": 0, "yetersiz": 1, "ai_uretimi": 2, "manipulatif": 3}
ONAY_RISK_SIRASI = {"onaylandi": 0, "gozden_gecirilecek": 1, "onaylanmadi": 2}


def load_kayitlar(folder: Path) -> pd.DataFrame:
    rows = []

    for path in sorted(folder.rglob("*.json")):
        with path.open(encoding="utf-8") as file:
            kayit = json.load(file)

        rows.append({**kayit, "file_name": path.name})

    kayitlar = pd.DataFrame(rows)

    for column in REQUIRED_COLUMNS:
        if column not in kayitlar:
            kayitlar[column] = pd.NA

    return kayitlar


def evaluate_against_ground_truth(sonuc: pd.DataFrame, model_adi: str) -> None:
    gercek = sonuc["is_anomali"]
    tahmin = sonuc["is_anomaly"]

    tn, fp, fn, tp = confusion_matrix(gercek, tahmin).ravel()
    precision = precision_score(gercek, tahmin, zero_division=0)
    recall = recall_score(gercek, tahmin, zero_division=0)
    f1 = f1_score(gercek, tahmin, zero_division=0)

    print(f"\n[{model_adi}] gercek etikete gore degerlendirme")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")


df = load_kayitlar(JSON_FOLDER)

if df.empty:
    raise FileNotFoundError(
        f"'{JSON_FOLDER}' klasorunde hic kayit json dosyasi bulunamadi."
    )

df["grup_no"] = df["cift_grup_id"].astype(str).str.split(":").str[0]
df["grup_buyuklugu"] = df.groupby("grup_no")["grup_no"].transform("size")
df.loc[df["cift_grup_id"].isna(), "grup_buyuklugu"] = np.nan

df["aciklama_risk"] = df["aciklama_kategorisi"].map(ACIKLAMA_RISK_SIRASI)
df["onay_risk"] = df["onay_durumu"].map(ONAY_RISK_SIRASI)

missing_required_data = df[REQUIRED_COLUMNS].isna().any(axis=1)
invalid_feature_data = ~np.isfinite(df[FEATURE_COLUMNS].astype(float)).all(axis=1)

df["data_quality_anomaly"] = missing_required_data | invalid_feature_data

valid_model_mask = ~df["data_quality_anomaly"]
X = df.loc[valid_model_mask, FEATURE_COLUMNS]
