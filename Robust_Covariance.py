
from main import *
from sklearn.covariance import EllipticEnvelope

#------------------------------------------------
#ROBUST COVARIANCE (ELLIPTIC ENVELOPE) MODELI
#------------------------------------------------

contamination = 0.02

model = EllipticEnvelope(
    contamination=contamination,
    random_state=42
)

# Eksik/gecersiz verili satirlar modele gonderilmeden dogrudan anomali
# olarak atanir. cov_score bos kalir; cunku bu satirlar model tarafindan
# skorlanmamistir.
df["prediction"] = -1
df["is_anomaly"] = True
df["cov_score"] = np.nan
df["anomaly_level"] = 100.0

if not X_egitim.empty:
    model.fit(X_egitim)

    tahmin = model.predict(X)
    # decision_function: pozitif -> normal, negatif -> anomali
    # (Mahalanobis mesafesinden turetilmis, isareti IF ile ayni yonde).
    skor = model.decision_function(X)

    df.loc[X.index, "prediction"] = tahmin
    df.loc[X.index, "is_anomaly"] = tahmin == -1
    df.loc[X.index, "cov_score"] = skor

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
            "cov_score",
            "is_anomaly"
        ]
    ].head(30)
)

result.to_csv(
    "robust_covariance_result.csv",
    index=False,
    encoding="utf-8-sig"
)
