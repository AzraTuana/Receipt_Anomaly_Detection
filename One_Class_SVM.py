
from main import *
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

#------------------------------------------------
#ONE-CLASS SVM MODELI
#------------------------------------------------
# One-Class SVM, "normal" kayıtların çoğunu içine alan bir sınır (RBF
# çekirdeğiyle doğrusal olmayan bir sınır) öğrenir; bu sınırın dışında
# kalan kayıtlar anomali sayılır. contamination yerine `nu` parametresi
# kullanılır: nu, eğitim hatalarının oranına üst sınır ve destek
# vektörlerinin oranına alt sınır koyar; contamination ile birebir aynı
# garanti olmasa da benzer bir rol oynar, bu yüzden diğer modellerle
# karşılaştırılabilir olsun diye aynı değer (0.02) kullanılıyor.
#
# RBF çekirdeği de LOF gibi mesafeye dayalı olduğundan features arasındaki
# ölçek farkı sonucu domine eder; StandardScaler ile ölçeklendirme şart.

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
