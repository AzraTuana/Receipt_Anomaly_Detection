
from main import *
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

contamination = 0.28

scaler = StandardScaler()

model = LocalOutlierFactor(
    n_neighbors=20,
    contamination=contamination,
    novelty=True,
    n_jobs=-1
)

df["prediction"] = -1
df["is_anomaly"] = True
df["lof_score"] = np.nan
df["anomaly_level"] = 0.0

if not X_egitim.empty:
    X_egitim_scaled = scaler.fit_transform(X_egitim)
    X_scaled = scaler.transform(X)

    model.fit(X_egitim_scaled)

    skor = model.score_samples(X_scaled)

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

hibrit_karari_uygula(
    score_column="lof_score",
    lower_scores_more_anomalous=True,
    default_threshold=float(model.offset_),
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
        [*MODEL_REPORT_COLUMNS, "lof_score"]
    ].head(30)
)

anomali_oranini_raporla(df, df.index, "Local Outlier Factor")

result.drop(columns=CSV_HARIC_KOLONLAR).to_csv(
    "lof_result.csv",
    index=False,
    encoding="utf-8-sig"
)
