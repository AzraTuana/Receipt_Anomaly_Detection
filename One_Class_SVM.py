
from main import *
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

#------------------------------------------------
#ONE-CLASS SVM MODELI
#------------------------------------------------

nu = 0.02

scaler = StandardScaler()

model = OneClassSVM(
    kernel="rbf",
    nu=nu,
    gamma="scale"
)

# Eksik/gecersiz verili satirlar modele gonderilmeden dogrudan anomali
# olarak atanir. svm_score bos kalir; cunku bu satirlar model tarafindan
# skorlanmamistir.
df["prediction"] = -1
df["is_anomaly"] = True
df["svm_score"] = np.nan
df["anomaly_level"] = 100.0

if not X_egitim.empty:
    X_egitim_scaled = scaler.fit_transform(X_egitim)
    X_scaled = scaler.transform(X)

    model.fit(X_egitim_scaled)

    tahmin = model.predict(X_scaled)
    # decision_function: pozitif -> sinirin icinde (normal),
    # negatif -> sinirin disinda (anomali).
    skor = model.decision_function(X_scaled)

    df.loc[X.index, "prediction"] = tahmin
    df.loc[X.index, "is_anomaly"] = tahmin == -1
    df.loc[X.index, "svm_score"] = skor

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
            "svm_score",
            "is_anomaly"
        ]
    ].head(30)
)

result.to_csv(
    "one_class_svm_result.csv",
    index=False,
    encoding="utf-8-sig"
)
