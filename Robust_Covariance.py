
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

# Eksik/geçersiz verili satırlar modele gönderilmeden doğrudan anomali
# olarak atanır. cov_score boş kalır; çünkü bu satırlar model tarafından
# skorlanmamıştır.
df["prediction"] = -1
df["is_anomaly"] = True
df["cov_score"] = np.nan
df["anomaly_level"] = 100.0

if not X.empty:
    model.fit(X)

    valid_predictions = model.predict(X)
    # decision_function: pozitif -> normal, negatif -> anomali
    # (Mahalanobis mesafesinden türetilmiş, işareti IF ile aynı yönde).
    valid_scores = model.decision_function(X)

    df.loc[valid_model_mask, "prediction"] = valid_predictions
    df.loc[valid_model_mask, "is_anomaly"] = valid_predictions == -1
    df.loc[valid_model_mask, "cov_score"] = valid_scores

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
            "kayit_id",
            "fatura_no",
            "cift_grup_id",
            "aciklama_kategorisi",
            "onay_durumu",
            "grup_buyuklugu",
            "aciklama_risk",
            "onay_risk",
            "data_quality_anomaly",
            "anomaly_level",
            "cov_score",
            "is_anomaly",
            "is_anomali"
        ]
    ].head(30)
)

evaluate_against_ground_truth(result, "Robust Covariance")

result.to_csv(
    "robust_covariance_result.csv",
    index=False,
    encoding="utf-8-sig"
)
