import json
from pathlib import Path
import numpy as np
import pandas as pd


JSON_ROOT = Path(__file__).parent / "json_files"
SPLIT_KLASORLERI = ["egitim", "dogrulama", "test"]
FORBIDDEN_KATEGORILER = {"kumar", "alkol", "tutun_urunleri"}
BEKLENEN_KATEGORI_ESIGI = 0.01

CSV_HARIC_KOLONLAR = ["kalemler", "kalem_kategorileri"]

FEATURE_COLUMNS = [
    "genel_toplam_log",
    "is_kolu_sapma_log",
    "toplam_tutarsizligi_log",
    "satir_toplam_tutarsizligi_log",
    "gelecek_tarihli",
    "yasakli_kategori_var",
    "kategori_uyumsuzlugu_var",
    "vkn_format_anomalisi",
]


def load_split(folder: Path, split_adi: str) -> pd.DataFrame:
    rows = []

    for path in sorted(folder.rglob("*.json")):
        with path.open(encoding="utf-8") as file:
            kayit = json.load(file)

        rows.append({**kayit, "file_name": path.name, "split": split_adi})

    return pd.DataFrame(rows)


df = pd.concat(
    [load_split(JSON_ROOT / ad, ad) for ad in SPLIT_KLASORLERI],
    ignore_index=True,
)

if df.empty:
    raise FileNotFoundError(
        f"'{JSON_ROOT}' altinda hic kayit json dosyasi bulunamadi."
    )

egitim_mask = df["split"] == "egitim"

# --- tutar/tarih tutarlilik kontrolleri ---
df["genel_toplam_log"] = np.log1p(df["genel_toplam"])

df["toplam_tutarsizligi_log"] = np.log1p(
    (df["genel_toplam"] - (df["toplam_vergisiz_tutar"] + df["toplam_kdv_tutari"])).abs()
)

df["satir_toplam_tutarsizligi_log"] = np.log1p(
    df.apply(
        lambda satir: abs(
            sum(kalem["satir_toplam"] for kalem in satir["kalemler"])
            - satir["genel_toplam"]
        ),
        axis=1,
    )
)

# yukleme_zamani ISO formatinda ("YYYY-MM-DDTHH:MM:SS+03:00") basladigi icin
# ilk 10 karakteri fatura_tarihi ("YYYY-MM-DD") ile dogrudan karsilastirilabilir.
df["gelecek_tarihli"] = (
    df["fatura_tarihi"] > df["yukleme_zamani"].str[:10]
).astype(int)

df["vkn_format_anomalisi"] = (
    df["satici_vkn"].astype(str).str.len() != 10
).astype(int)

# --- kategori tabanli kontroller (referans egitimden cikarilir) ---
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
    is_kolu: set(pay[pay >= BEKLENEN_KATEGORI_ESIGI].index) - FORBIDDEN_KATEGORILER
    for is_kolu, pay in kategori_payi.groupby(level=0)
}


def kategori_uyumsuz_mu(is_kolu: str, kategoriler: list) -> int:
    beklenen = beklenen_kategoriler.get(is_kolu, set())
    return int(any(k not in beklenen and k not in FORBIDDEN_KATEGORILER for k in kategoriler))


df["kategori_uyumsuzlugu_var"] = df.apply(
    lambda satir: kategori_uyumsuz_mu(satir["is_kolu"], satir["kalem_kategorileri"]),
    axis=1,
)

# --- is_kolu bazinda tutar sapmasi (referans medyan egitimden) ---
is_kolu_medyan = df.loc[egitim_mask].groupby("is_kolu")["genel_toplam"].median()
genel_medyan = df.loc[egitim_mask, "genel_toplam"].median()
referans_tutar = df["is_kolu"].map(is_kolu_medyan).fillna(genel_medyan)

df["is_kolu_sapma_log"] = np.log((df["genel_toplam"] + 1) / (referans_tutar + 1))

# --- veri kalitesi / model girdisi ---
invalid_feature_data = ~np.isfinite(df[FEATURE_COLUMNS].astype(float)).all(axis=1)
df["data_quality_anomaly"] = invalid_feature_data

valid_mask = ~df["data_quality_anomaly"]
X_egitim = df.loc[egitim_mask & valid_mask, FEATURE_COLUMNS]
X = df.loc[valid_mask, FEATURE_COLUMNS]


def anomali_oranini_raporla(df: pd.DataFrame, skorlanan_index: pd.Index, model_adi: str) -> None:
    skorlanan = df.loc[skorlanan_index]
    toplam_oran = skorlanan["is_anomaly"].mean() * 100

    print(f"\n[{model_adi}] anomali orani: %{toplam_oran:.2f} ({skorlanan['is_anomaly'].sum()} / {len(skorlanan)} kayit)")
    for split_adi, grup in skorlanan.groupby("split"):
        oran = grup["is_anomaly"].mean() * 100
        print(f"  {split_adi}: %{oran:.2f} ({grup['is_anomaly'].sum()} / {len(grup)} kayit)")
