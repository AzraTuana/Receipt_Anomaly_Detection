
from main import *
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler

contamination = 0.28

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

df["prediction"] = -1
df["is_anomaly"] = True
df["cov_score"] = np.nan
df["anomaly_level"] = 0.0

if not X_egitim.empty:
    X_egitim_robust = scaler.fit_transform(df.loc[X_egitim.index, ROBUST_FEATURE_COLUMNS])
    X_robust = scaler.transform(df.loc[X.index, ROBUST_FEATURE_COLUMNS])
    model.fit(X_egitim_robust)

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
