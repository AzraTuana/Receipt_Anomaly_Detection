
from main import *
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler

#------------------------------------------------
#ROBUST COVARIANCE (ELLIPTIC ENVELOPE) MODELI
#------------------------------------------------
# Calisma prensibi: Ozelliklerin cok degiskenli dagilimini, aykiri degerlerden
# daha az etkilenen saglam bir ortalama ve kovaryans ile eliptik bir zarf olarak
# tahmin eder. Bu merkeze Mahalanobis mesafesi buyuk olan kayitlar anomalidir;
# ozellikler arasindaki korelasyonlar da mesafeye dahil edilir.
# Bu projedeki kullanim alani: Birlikte hareket etmesi beklenen mali sinyallerin
# (genel toplam, vergi/kalem farklari ve sektor sapmasi gibi) alisilmis ortak
# iliskisini bozan global aykiriliklari yakalar. Yaklasik eliptik bir normal
# bulut varsayimi nedeniyle karmasik veya cok tepeli dagilimlarda daha sinirlidir.

# Karar zarfinin disinda kalmasi beklenen yaklasik egitim payi.
contamination = 0.28

# EllipticEnvelope surekli dagilim ve tam-rank kovaryans bekler. Birbirine bagli
# ikili kural bayraklari hibrit katmanda zaten ele alindigi icin bu model yalnizca
# surekli sapma/hata buyuklukleriyle calisir.
ROBUST_FEATURE_COLUMNS = [
    "genel_toplam_log",
    "is_kolu_sapma_log",
    "birim_fiyat_yuksek_sapma_log",
    "birim_fiyat_dusuk_sapma_log",
]
scaler = StandardScaler()

model = EllipticEnvelope(
    contamination=contamination,
    support_fraction=0.9,
    random_state=42
)

# Eksik/gecersiz verili satirlar modele gonderilmeden dogrudan anomali
# olarak atanir. cov_score bos kalir; cunku bu satirlar model tarafindan
# skorlanmamistir.
df["prediction"] = -1
df["is_anomaly"] = True
df["cov_score"] = np.nan
df["anomaly_level"] = 0.0

if not X_egitim.empty:
    # Saglam merkez ve kovaryans yalnizca egitim kayitlarindan ogrenilir.
    X_egitim_robust = scaler.fit_transform(df.loc[X_egitim.index, ROBUST_FEATURE_COLUMNS])
    X_robust = scaler.transform(df.loc[X.index, ROBUST_FEATURE_COLUMNS])
    model.fit(X_egitim_robust)

    # decision_function: pozitif -> normal, negatif -> anomali
    # (Mahalanobis mesafesinden turetilmis, isareti IF ile ayni yonde).
    skor = model.decision_function(X_robust)

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

hibrit_karari_uygula(
    score_column="cov_score",
    lower_scores_more_anomalous=True,
    default_threshold=0.0,
)

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
        [*MODEL_REPORT_COLUMNS, "cov_score"]
    ].head(30)
)

anomali_oranini_raporla(df, df.index, "Robust Covariance")

result.drop(columns=CSV_HARIC_KOLONLAR).to_csv(
    "robust_covariance_result.csv",
    index=False,
    encoding="utf-8-sig"
)
