
from main import *
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

#------------------------------------------------
#AUTOENCODER MODELI
#------------------------------------------------
contamination = 0.02

scaler = StandardScaler()

autoencoder = MLPRegressor(
    hidden_layer_sizes=(2,),
    activation="tanh",
    solver="adam",
    max_iter=5000,
    random_state=42,
)

# Eksik/geçersiz verili satırlar modele gönderilmeden doğrudan anomali
# olarak atanır. ae_score boş kalır; çünkü bu satırlar model tarafından
# skorlanmamıştır.
df["prediction"] = -1
df["is_anomaly"] = True
df["ae_score"] = np.nan
df["anomaly_level"] = 100.0

if not X.empty:
    X_scaled = scaler.fit_transform(X)

    autoencoder.fit(X_scaled, X_scaled)
    X_reconstructed = autoencoder.predict(X_scaled)

    # Satır başına yeniden üretim hatası (ortalama kare hata)
    reconstruction_error = np.mean(
        (X_scaled - X_reconstructed) ** 2,
        axis=1
    )

    # contamination oranına karşılık gelen eşik: hatası bu eşiğin
    # üzerinde olan kayıtlar anomali kabul edilir (IsolationForest'ın
    # contamination parametresiyle aynı mantık, elle uygulanmış hali).
    threshold = np.quantile(reconstruction_error, 1 - contamination)

    valid_predictions = np.where(reconstruction_error > threshold, -1, 1)

    df.loc[valid_model_mask, "prediction"] = valid_predictions
    df.loc[valid_model_mask, "is_anomaly"] = valid_predictions == -1
    df.loc[valid_model_mask, "ae_score"] = reconstruction_error

    minimum = reconstruction_error.min()
    maximum = reconstruction_error.max()

    if maximum != minimum:
        valid_anomaly_level = (
            (reconstruction_error - minimum)
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
            "ae_score",
            "is_anomaly",
            "is_anomali"
        ]
    ].head(30)
)

evaluate_against_ground_truth(result, "Autoencoder")

result.to_csv(
    "autoencoder_result.csv",
    index=False,
    encoding="utf-8-sig"
)
