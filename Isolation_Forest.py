
from main import *
from sklearn.ensemble import IsolationForest

#------------------------------------------------
#ISOLATION FOREST MODELI
#------------------------------------------------
# contamination=0.02

# model.score_samples(X) ile hesaplanan ham anomali skorları
# küçükten büyüğe sıralandığında, en uçtaki tek nokta ayrıldıktan sonraki
# en büyük ikinci sıçrama, sıralı skorların tam olarak 5. noktasında
# (225 kayıdın %2,2'si) görülüyor; bu noktadan sonra sıçramalar küçülüp
# gürültü seviyesine iniyor. contamination=0.02, 225*0.02 ≈ 5 kaydı
# işaretleyerek bu doğal kırılma noktasıyla örtüşüyor.

model = IsolationForest(
    n_estimators=300,
    contamination=0.02,
    random_state=42,
    n_jobs=-1
)

# Eksik/geçersiz verili satırlar modele gönderilmeden doğrudan anomali
# olarak atanır. if_score boş kalır; çünkü bu satırlar model tarafından
# skorlanmamıştır.
df["prediction"] = -1
df["is_anomaly"] = True
df["if_score"] = np.nan
df["anomaly_level"] = 100.0

if not X.empty:
    model.fit(X)

    valid_predictions = model.predict(X)
    valid_scores = model.decision_function(X)

    df.loc[valid_model_mask, "prediction"] = valid_predictions
    df.loc[valid_model_mask, "is_anomaly"] = valid_predictions == -1
    df.loc[valid_model_mask, "if_score"] = valid_scores

    raw_anomaly = -valid_scores
    minimum = raw_anomaly.min()
    maximum = raw_anomaly.max()

    if maximum != minimum:
        valid_anomaly_level = (
            (raw_anomaly - minimum)
            / (maximum - minimum)
        ) * 100
    else:
        valid_anomaly_level = np.zeros(len(X))

    df.loc[valid_model_mask, "anomaly_level"] = valid_anomaly_level

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
            "file_name",
            "company",
            "company_normalized",
            "date_original",
            "total",
            "total_log",
            "company_deviation_log",
            "days_since_latest",
            "data_quality_anomaly",
            "anomaly_level",
            "if_score",
            "is_anomaly"
        ]
    ].head(30)
)

result.to_csv(
    "isolation_forest_result.csv",
    index=False,
    encoding="utf-8-sig"
)
