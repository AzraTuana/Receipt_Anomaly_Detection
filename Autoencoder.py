
from main import *
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

#------------------------------------------------
#AUTOENCODER MODELI
#------------------------------------------------
contamination = 0.28

scaler = StandardScaler()

autoencoder = MLPRegressor(
    hidden_layer_sizes=(2,),
    activation="tanh",
    solver="adam",
    max_iter=5000,
    random_state=42,
)

# Eksik/gecersiz verili satirlar modele gonderilmeden dogrudan anomali
# olarak atanir. ae_score bos kalir; cunku bu satirlar model tarafindan
# skorlanmamistir.
df["prediction"] = -1
df["is_anomaly"] = True
df["ae_score"] = np.nan
df["anomaly_level"] = 100.0

if not X_egitim.empty:
    X_egitim_scaled = scaler.fit_transform(X_egitim)
    X_scaled = scaler.transform(X)

    autoencoder.fit(X_egitim_scaled, X_egitim_scaled)

    # Esik egitim setinin yeniden uretim hatasindan cikarilir, sonra
    # skorlanacak tum kayitlara (egitim+dogrulama+test) sabit olarak uygulanir.
    egitim_reconstructed = autoencoder.predict(X_egitim_scaled)
    egitim_error = np.mean((X_egitim_scaled - egitim_reconstructed) ** 2, axis=1)
    threshold = np.quantile(egitim_error, 1 - contamination)

    X_reconstructed = autoencoder.predict(X_scaled)
    reconstruction_error = np.mean(
        (X_scaled - X_reconstructed) ** 2,
        axis=1
    )

    tahmin = np.where(reconstruction_error > threshold, -1, 1)

    df.loc[X.index, "prediction"] = tahmin
    df.loc[X.index, "is_anomaly"] = tahmin == -1
    df.loc[X.index, "ae_score"] = reconstruction_error

    minimum = reconstruction_error.min()
    maximum = reconstruction_error.max()

    if maximum != minimum:
        anomaly_level = (
            (reconstruction_error - minimum)
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
            "ae_score",
            "is_anomaly"
        ]
    ].head(30)
)

anomali_oranini_raporla(df, X.index, "Autoencoder")

result.to_csv(
    "autoencoder_result.csv",
    index=False,
    encoding="utf-8-sig"
)
