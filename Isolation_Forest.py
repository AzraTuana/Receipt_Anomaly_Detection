
from main import *
from sklearn.ensemble import IsolationForest

model = IsolationForest(
    n_estimators=300,
    contamination=0.28,
    random_state=42,
    n_jobs=-1
)

df["prediction"] = -1
df["is_anomaly"] = True
df["if_score"] = np.nan
df["anomaly_level"] = 0.0

if not X_egitim.empty:
    model.fit(X_egitim)

    skor = model.decision_function(X)

    df.loc[X.index, "if_score"] = skor

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
    score_column="if_score",
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
        [*MODEL_REPORT_COLUMNS, "if_score"]
    ].head(30)
)

anomali_oranini_raporla(df, df.index, "Isolation Forest")

result.drop(columns=CSV_HARIC_KOLONLAR).to_csv(
    "isolation_forest_result.csv",
    index=False,
    encoding="utf-8-sig"
)
