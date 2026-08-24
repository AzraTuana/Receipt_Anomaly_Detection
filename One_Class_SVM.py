
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

# Eksik/geçersiz verili satırlar modele gönderilmeden doğrudan anomali
# olarak atanır. svm_score boş kalır; çünkü bu satırlar model tarafından
# skorlanmamıştır.
df["prediction"] = -1
df["is_anomaly"] = True
df["svm_score"] = np.nan
df["anomaly_level"] = 100.0

if not X.empty:
    X_scaled = scaler.fit_transform(X)

    model.fit(X_scaled)

    valid_predictions = model.predict(X_scaled)
    # decision_function: pozitif -> sınırın içinde (normal),
    # negatif -> sınırın dışında (anomali).
    valid_scores = model.decision_function(X_scaled)

    df.loc[valid_model_mask, "prediction"] = valid_predictions
    df.loc[valid_model_mask, "is_anomaly"] = valid_predictions == -1
    df.loc[valid_model_mask, "svm_score"] = valid_scores

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
            "svm_score",
            "is_anomaly",
            "is_anomali"
        ]
    ].head(30)
)

evaluate_against_ground_truth(result, "One-Class SVM")

result.to_csv(
    "one_class_svm_result.csv",
    index=False,
    encoding="utf-8-sig"
)
