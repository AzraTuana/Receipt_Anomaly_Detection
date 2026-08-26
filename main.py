import json
from pathlib import Path
import numpy as np
import pandas as pd

JSON_ROOT = Path(__file__).parent / "json_files"
VALIDATION_LABEL_PATH = Path(__file__).parent / "text_files_doğrulama" / "dogrulama_etiket.json"
SPLIT_KLASORLERI = ["egitim", "dogrulama", "test"]
FORBIDDEN_KATEGORILER = {"kumar", "alkol", "tutun_urunleri"}
BEKLENEN_KATEGORI_ESIGI = 0.01
PARASAL_TOLERANS = 5.00
ALT_TUTAR_QUANTILE = 0.05
UST_TUTAR_QUANTILE = 0.95

CSV_HARIC_KOLONLAR = ["kalemler", "kalem_kategorileri"]

FEATURE_COLUMNS = [
    "genel_toplam_log",
    "is_kolu_sapma_log",
    "is_kolu_ust_limit_sapmasi_log",
    "is_kolu_alt_limit_sapmasi_log",
    "birim_fiyat_yuksek_sapma_log",
    "birim_fiyat_dusuk_sapma_log",
    "toplam_tutarsizligi_log",
    "satir_toplam_tutarsizligi_log",
    "ara_toplam_tutarsizligi_log",
    "kdv_tutarsizligi_log",
    "satir_matematik_tutarsizligi_log",
    "gelecek_tarihli",
    "yasakli_kategori_var",
    "kategori_uyumsuzlugu_var",
    "vkn_format_anomalisi",
    "satici_kimlik_referans_uyumsuzlugu",
    "mukerrer_fatura_var",
]

RULE_FLAG_COLUMNS = {
    "mukerrer_fatura": "mukerrer_fatura_var",
    "gelecek_tarih": "gelecek_tarihli",
    "yasakli_kategori": "yasakli_kategori_var",
    "kimlik_formati": "vkn_format_anomalisi",
    "genel_toplam_matematigi": "toplam_tutarsizligi_var",
    "kalemler_genel_toplam": "satir_toplam_tutarsizligi_var",
    "ara_toplam_matematigi": "ara_toplam_tutarsizligi_var",
    "kdv_matematigi": "kdv_tutarsizligi_var",
    "satir_matematigi": "satir_matematik_tutarsizligi_var",
    "veri_kalitesi": "data_quality_anomaly",
}

def load_split(folder: Path, split_adi: str) -> pd.DataFrame:
    rows = []

    for path in sorted(folder.rglob("*.json")):
        with path.open(encoding="utf-8") as file:
            kayit = json.load(file)

        rows.append({**kayit, "file_name": path.name, "split": split_adi})

    return pd.DataFrame(rows)

def _sayiya_cevir(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan

def fatura_matematik_hatalari(satir: pd.Series) -> pd.Series:
    kalemler = satir.get("kalemler")
    if not isinstance(kalemler, list) or not kalemler:
        return pd.Series(
            {
                "toplam_tutarsizligi": np.inf,
                "satir_toplam_tutarsizligi": np.inf,
                "ara_toplam_tutarsizligi": np.inf,
                "kdv_tutarsizligi": np.inf,
                "satir_matematik_tutarsizligi": np.inf,
            }
        )

    ara_toplamlar = []
    kdv_tutarlari = []
    satir_toplamlari = []
    ara_hatalari = []
    kdv_hatalari = []
    satir_hatalari = []

    for kalem in kalemler:
        miktar = _sayiya_cevir(kalem.get("miktar"))
        birim_fiyat = _sayiya_cevir(kalem.get("birim_fiyat"))
        iskonto_orani = _sayiya_cevir(kalem.get("iskonto_orani", 0))
        kdv_orani = _sayiya_cevir(kalem.get("kdv_orani"))
        ara_toplam = _sayiya_cevir(kalem.get("ara_toplam"))
        kdv_tutari = _sayiya_cevir(kalem.get("kdv_tutari"))
        satir_toplam = _sayiya_cevir(kalem.get("satir_toplam"))

        beklenen_ara_toplam = miktar * birim_fiyat * (1 - iskonto_orani / 100)
        beklenen_kdv = ara_toplam * kdv_orani / 100
        beklenen_satir_toplam = ara_toplam + kdv_tutari

        ara_toplamlar.append(ara_toplam)
        kdv_tutarlari.append(kdv_tutari)
        satir_toplamlari.append(satir_toplam)
        ara_hatalari.append(abs(ara_toplam - beklenen_ara_toplam))
        kdv_hatalari.append(abs(kdv_tutari - beklenen_kdv))
        satir_hatalari.append(abs(satir_toplam - beklenen_satir_toplam))

    toplam_vergisiz = _sayiya_cevir(satir.get("toplam_vergisiz_tutar"))
    toplam_kdv = _sayiya_cevir(satir.get("toplam_kdv_tutari"))
    genel_toplam = _sayiya_cevir(satir.get("genel_toplam"))

    return pd.Series(
        {
            "toplam_tutarsizligi": abs(genel_toplam - (toplam_vergisiz + toplam_kdv)),
            "satir_toplam_tutarsizligi": abs(sum(satir_toplamlari) - genel_toplam),
            "ara_toplam_tutarsizligi": max(
                max(ara_hatalari),
                abs(sum(ara_toplamlar) - toplam_vergisiz),
            ),
            "kdv_tutarsizligi": max(
                max(kdv_hatalari),
                abs(sum(kdv_tutarlari) - toplam_kdv),
            ),
            "satir_matematik_tutarsizligi": max(
                max(satir_hatalari),
                abs(sum(satir_toplamlari) - genel_toplam),
            ),
        }
    )

df = pd.concat(
    [load_split(JSON_ROOT / ad, ad) for ad in SPLIT_KLASORLERI],
    ignore_index=True,
)

if df.empty:
    raise FileNotFoundError(
        f"'{JSON_ROOT}' altinda hic kayit json dosyasi bulunamadi."
    )

egitim_mask = df["split"] == "egitim"

df["genel_toplam_log"] = np.log1p(df["genel_toplam"])

matematik_hatalari = df.apply(fatura_matematik_hatalari, axis=1)
for hata_adi in matematik_hatalari.columns:
    df[hata_adi] = matematik_hatalari[hata_adi]
    df[f"{hata_adi}_log"] = np.log1p(df[hata_adi])
    df[f"{hata_adi}_var"] = (df[hata_adi] > PARASAL_TOLERANS).astype(int)

df["gelecek_tarihli"] = (
    df["fatura_tarihi"] > df["yukleme_zamani"].str[:10]
).astype(int)

kimlik_no = df["satici_vkn"].astype(str).str.strip()
df["vkn_format_anomalisi"] = (
    ~kimlik_no.str.fullmatch(r"(?:\d{10}|\d{11})", na=False)
).astype(int)

df["mukerrer_fatura_var"] = df.duplicated(
    subset=["satici_vkn", "fatura_no"],
    keep=False,
).astype(int)

df["kalem_kategorileri"] = df["kalemler"].apply(
    lambda kalemler: [kalem["harcama_kategorisi"] for kalem in kalemler]
)

df["yasakli_kategori_var"] = df["kalem_kategorileri"].apply(
    lambda kategoriler: int(any(k in FORBIDDEN_KATEGORILER for k in kategoriler))
)

egitim_kategori_sayimi = (
    df.loc[egitim_mask, ["is_kolu", "kalem_kategorileri"]]
    .explode("kalem_kategorileri")
    .groupby(["is_kolu", "kalem_kategorileri"])
    .size()
)
is_kolu_toplam = egitim_kategori_sayimi.groupby(level=0).sum()
kategori_payi = egitim_kategori_sayimi / is_kolu_toplam

beklenen_kategoriler = {
    is_kolu: (
        set(
            pay[pay >= BEKLENEN_KATEGORI_ESIGI]
            .index
            .get_level_values("kalem_kategorileri")
        )
        - FORBIDDEN_KATEGORILER
    )
    for is_kolu, pay in kategori_payi.groupby(level=0)
}

def kategori_uyumsuz_mu(is_kolu: str, kategoriler: list) -> int:
    beklenen = beklenen_kategoriler.get(is_kolu, set())
    return int(any(k not in beklenen and k not in FORBIDDEN_KATEGORILER for k in kategoriler))

df["kategori_uyumsuzlugu_var"] = df.apply(
    lambda satir: kategori_uyumsuz_mu(satir["is_kolu"], satir["kalem_kategorileri"]),
    axis=1,
)

egitim_tutarlari = df.loc[egitim_mask].groupby("is_kolu")["genel_toplam"]
is_kolu_medyan = egitim_tutarlari.median()
is_kolu_alt_limit = egitim_tutarlari.quantile(ALT_TUTAR_QUANTILE)
is_kolu_ust_limit = egitim_tutarlari.quantile(UST_TUTAR_QUANTILE)
genel_medyan = df.loc[egitim_mask, "genel_toplam"].median()
genel_alt_limit = df.loc[egitim_mask, "genel_toplam"].quantile(ALT_TUTAR_QUANTILE)
genel_ust_limit = df.loc[egitim_mask, "genel_toplam"].quantile(UST_TUTAR_QUANTILE)
referans_tutar = df["is_kolu"].map(is_kolu_medyan).fillna(genel_medyan)
referans_alt_limit = df["is_kolu"].map(is_kolu_alt_limit).fillna(genel_alt_limit)
referans_ust_limit = df["is_kolu"].map(is_kolu_ust_limit).fillna(genel_ust_limit)

df["is_kolu_sapma_log"] = np.log((df["genel_toplam"] + 1) / (referans_tutar + 1))
df["is_kolu_ust_limit_sapmasi_log"] = np.maximum(
    np.log((df["genel_toplam"] + 1) / (referans_ust_limit + 1)),
    0,
)
df["is_kolu_alt_limit_sapmasi_log"] = np.maximum(
    np.log((referans_alt_limit + 1) / (df["genel_toplam"] + 1)),
    0,
)

satici_kimlik_referansi = (
    df.loc[egitim_mask]
    .groupby("satici_unvan")["satici_vkn"]
    .agg(lambda values: values.value_counts().index[0])
)
beklenen_satici_kimligi = df["satici_unvan"].map(satici_kimlik_referansi)
df["satici_kimlik_referans_uyumsuzlugu"] = (
    beklenen_satici_kimligi.notna()
    & (df["satici_vkn"].astype(str) != beklenen_satici_kimligi.astype(str))
).astype(int)

egitim_birim_fiyatlari = pd.DataFrame(
    [
        {
            "harcama_kategorisi": kalem.get("harcama_kategorisi"),
            "birim_fiyat": _sayiya_cevir(kalem.get("birim_fiyat")),
        }
        for kalemler in df.loc[egitim_mask, "kalemler"]
        if isinstance(kalemler, list)
        for kalem in kalemler
    ]
)
birim_fiyat_medyanlari = egitim_birim_fiyatlari.groupby("harcama_kategorisi")[
    "birim_fiyat"
].median()
genel_birim_fiyat_medyan = egitim_birim_fiyatlari["birim_fiyat"].median()

def birim_fiyat_sapmalari(kalemler: list) -> pd.Series:
    oranlar = []
    for kalem in kalemler if isinstance(kalemler, list) else []:
        kategori = kalem.get("harcama_kategorisi")
        fiyat = _sayiya_cevir(kalem.get("birim_fiyat"))
        referans = birim_fiyat_medyanlari.get(kategori, genel_birim_fiyat_medyan)
        oranlar.append(np.log((max(fiyat, 0) + 1) / (max(referans, 0) + 1)))

    if not oranlar:
        return pd.Series(
            {
                "birim_fiyat_yuksek_sapma_log": np.inf,
                "birim_fiyat_dusuk_sapma_log": np.inf,
            }
        )

    return pd.Series(
        {
            "birim_fiyat_yuksek_sapma_log": max(0, max(oranlar)),
            "birim_fiyat_dusuk_sapma_log": max(0, -min(oranlar)),
        }
    )

df[["birim_fiyat_yuksek_sapma_log", "birim_fiyat_dusuk_sapma_log"]] = (
    df["kalemler"].apply(birim_fiyat_sapmalari)
)

invalid_feature_data = ~np.isfinite(df[FEATURE_COLUMNS].astype(float)).all(axis=1)
df["data_quality_anomaly"] = invalid_feature_data

df["rule_anomaly_reasons"] = df.apply(
    lambda satir: "|".join(
        neden
        for neden, kolon in RULE_FLAG_COLUMNS.items()
        if bool(satir[kolon])
    ),
    axis=1,
)
df["rule_based_anomaly"] = df["rule_anomaly_reasons"].str.len() > 0

valid_mask = ~df["data_quality_anomaly"]
X_egitim = df.loc[egitim_mask & valid_mask, FEATURE_COLUMNS]
X = df.loc[valid_mask, FEATURE_COLUMNS]

MODEL_REPORT_COLUMNS = [
    "kayit_id",
    "fatura_no",
    "split",
    "is_kolu",
    "satici_unvan",
    "genel_toplam",
    *FEATURE_COLUMNS,
    "toplam_tutarsizligi_var",
    "satir_toplam_tutarsizligi_var",
    "ara_toplam_tutarsizligi_var",
    "kdv_tutarsizligi_var",
    "satir_matematik_tutarsizligi_var",
    "data_quality_anomaly",
    "rule_based_anomaly",
    "rule_anomaly_reasons",
    "model_is_anomaly",
    "model_anomaly_level",
    "decision_threshold",
    "threshold_source",
    "calibration_validation_f1",
    "anomaly_level",
    "is_anomaly",
]

def _f1_hesapla(true_positive: int, false_positive: int, false_negative: int) -> float:
    payda = 2 * true_positive + false_positive + false_negative
    return 0.0 if payda == 0 else (2 * true_positive) / payda

def load_validation_ground_truth() -> pd.Series:
    if not VALIDATION_LABEL_PATH.exists():
        raise FileNotFoundError(f"Dogrulama etiketi bulunamadi: {VALIDATION_LABEL_PATH}")

    with VALIDATION_LABEL_PATH.open(encoding="utf-8") as file:
        records = json.load(file)

    labels = pd.DataFrame(records)
    required = {"kayit_id", "is_anomali"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(
            f"{VALIDATION_LABEL_PATH.name} eksik kolonlar: {sorted(missing)}"
        )
    if labels["kayit_id"].duplicated().any():
        raise ValueError("Dogrulama etiketinde tekrar eden kayit_id bulundu.")

    return labels.set_index("kayit_id")["is_anomali"].astype(bool)

def kalibre_edilmis_esik(
    score_column: str,
    lower_scores_more_anomalous: bool,
    default_threshold: float,
) -> tuple[float, float, str]:
    try:
        ground_truth = load_validation_ground_truth()
    except FileNotFoundError:
        return default_threshold, np.nan, "model_default"

    validation = df.loc[
        (df["split"] == "dogrulama") & df[score_column].notna(),
        ["kayit_id", score_column, "rule_based_anomaly"],
    ].copy()
    validation["ground_truth"] = validation["kayit_id"].map(ground_truth)
    if validation["ground_truth"].isna().any():
        missing_ids = validation.loc[
            validation["ground_truth"].isna(),
            "kayit_id",
        ].head(10)
        raise ValueError(
            "Dogrulama tahmini icin gercek etiket bulunamadi: "
            f"{missing_ids.tolist()}"
        )

    truth = validation["ground_truth"].astype(bool)
    rule_prediction = validation["rule_based_anomaly"].astype(bool)
    true_positive = int((rule_prediction & truth).sum())
    false_positive = int((rule_prediction & ~truth).sum())
    false_negative = int((~rule_prediction & truth).sum())

    best_f1 = _f1_hesapla(true_positive, false_positive, false_negative)
    best_k = 0

    candidates = validation.loc[~rule_prediction].sort_values(
        score_column,
        ascending=lower_scores_more_anomalous,
    )
    for k, is_true_anomaly in enumerate(candidates["ground_truth"].astype(bool), start=1):
        if is_true_anomaly:
            true_positive += 1
            false_negative -= 1
        else:
            false_positive += 1

        current_f1 = _f1_hesapla(true_positive, false_positive, false_negative)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_k = k

    if best_k == 0:
        threshold = -np.inf if lower_scores_more_anomalous else np.inf
    else:
        threshold = float(candidates.iloc[best_k - 1][score_column])

    return threshold, best_f1, "dogrulama_f1"

def hibrit_karari_uygula(
    score_column: str,
    lower_scores_more_anomalous: bool,
    default_threshold: float,
) -> None:
    threshold, validation_f1, threshold_source = kalibre_edilmis_esik(
        score_column,
        lower_scores_more_anomalous,
        default_threshold,
    )
    score_is_valid = df[score_column].notna()
    if lower_scores_more_anomalous:
        model_prediction = df[score_column] <= threshold
    else:
        model_prediction = df[score_column] >= threshold

    df["model_is_anomaly"] = score_is_valid & model_prediction
    df["model_anomaly_level"] = df["anomaly_level"]
    df["is_anomaly"] = df["rule_based_anomaly"] | df["model_is_anomaly"]
    df["prediction"] = np.where(df["is_anomaly"], -1, 1)

    df["anomaly_level"] = np.where(
        df["rule_based_anomaly"],
        100.0,
        df["model_anomaly_level"],
    )
    df["decision_threshold"] = threshold
    df["threshold_source"] = threshold_source
    df["calibration_validation_f1"] = validation_f1

    print(
        f"Kalibre esik ({score_column}): {threshold:.8g} | "
        f"kaynak={threshold_source} | validation_hibrit_f1={validation_f1:.4f}"
    )

def anomali_oranini_raporla(df: pd.DataFrame, skorlanan_index: pd.Index, model_adi: str) -> None:
    skorlanan = df.loc[skorlanan_index]
    toplam_oran = skorlanan["is_anomaly"].mean() * 100

    print(f"\n[{model_adi}] anomali orani: %{toplam_oran:.2f} ({skorlanan['is_anomaly'].sum()} / {len(skorlanan)} kayit)")
    for split_adi, grup in skorlanan.groupby("split"):
        oran = grup["is_anomaly"].mean() * 100
        print(f"  {split_adi}: %{oran:.2f} ({grup['is_anomaly'].sum()} / {len(grup)} kayit)")
