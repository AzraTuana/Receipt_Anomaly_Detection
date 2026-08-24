
from main import *
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

#------------------------------------------------
#LOCAL OUTLIER FACTOR (LOF) MODELI
#------------------------------------------------

contamination = 0.28

scaler = StandardScaler()

# novelty=True: egitimde fit edilip baska bir kayit setini (dogrulama/test)
# skorlayabilmek icin gerekli; varsayilan novelty=False sadece fit_predict
# (tek veri seti) destekler.
model = LocalOutlierFactor(
    n_neighbors=20,
    contamination=contamination,
    novelty=True,
    n_jobs=-1
)

# Eksik/gecersiz verili satirlar modele gonderilmeden dogrudan anomali
# olarak atanir. lof_score bos kalir; cunku bu satirlar model tarafindan
# skorlanmamistir.
df["prediction"] = -1
df["is_anomaly"] = True
df["lof_score"] = np.nan
df["anomaly_level"] = 100.0

if not X_egitim.empty:
    X_egitim_scaled = scaler.fit_transform(X_egitim)
    X_scaled = scaler.transform(X)

    model.fit(X_egitim_scaled)

    tahmin = model.predict(X_scaled)
    # score_samples: normal kayitlarda ~-1'e yakin, anomalilerde
    # daha kucuk (daha negatif) degerler alir (negative_outlier_factor_ ile ayni yon).
    skor = model.score_samples(X_scaled)

    df.loc[X.index, "prediction"] = tahmin
    df.loc[X.index, "is_anomaly"] = tahmin == -1
    df.loc[X.index, "lof_score"] = skor

    raw_anomaly = -skor
    minimum = raw_anomaly.min()
    maximum = raw_anomaly.max()

    if maximum != minimum:
        anomaly_level = (
            (raw_anomaly - minimum)
            / (maximum - minimum)
        ) * 100
    else:
        anomaly_level = np.zeros(len(X))

    df.loc[X.index, "anomaly_level"] = anomaly_level

df["anomaly_level"] = (
    df["anomaly_level"]
    .round(2)
)

result = df.sort_values(
    by="anomaly_level",
    ascending=False
)

print(
    result[
        [
            "kayit_id",
            "fatura_no",
            "split",
            "is_kolu",
            "satici_unvan",
            "genel_toplam",
            "genel_toplam_log",
            "is_kolu_sapma_log",
            "toplam_tutarsizligi_log",
            "satir_toplam_tutarsizligi_log",
            "gelecek_tarihli",
            "yasakli_kategori_var",
            "kategori_uyumsuzlugu_var",
            "vkn_format_anomalisi",
            "data_quality_anomaly",
            "anomaly_level",
            "lof_score",
            "is_anomaly"
        ]
    ].head(30)
)

anomali_oranini_raporla(df, X.index, "Local Outlier Factor")

result.to_csv(
    "lof_result.csv",
    index=False,
    encoding="utf-8-sig"
)
