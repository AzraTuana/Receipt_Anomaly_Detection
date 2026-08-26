
from main import *
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

#------------------------------------------------
#AUTOENCODER MODELI
#------------------------------------------------
# Calisma prensibi: Ag, girdiyi yine kendisi hedef olacak sekilde yeniden
# uretmeyi ogrenir. Simetrik darbogaz katmanlari ortak ozellikleri dort boyutlu
# bir temsile sikistirdigi icin egitimdeki baskin normal oruntuleri iyi,
# alisilmadik kombinasyonlari ise daha yuksek yeniden uretim hatasiyla kurar.
# Bu projedeki kullanim alani: Sabit bir dagilim sekli varsaymadan ozellikler
# arasindaki dogrusal olmayan baglantilari ogrenip yuksek yeniden uretim hatali
# fisleri yakalar. Egitimdeki anomaliler de ogrenilebileceginden esik ve egitim
# verisinin temizligi sonucu dogrudan etkiler.

# Egitim yeniden uretim hatalarinin en yuksek %28'lik bolumu, anomali esigini
# belirler. Bu oran kesin usulsuzluk orani olarak yorumlanmamalidir.
contamination = 0.28

# Sinir aginin optimizasyonunu ve hata hesabini dengeli tutmak icin olcekleme
# egitim verisine uydurulur, diger splitlere degistirilmeden uygulanir.
scaler = StandardScaler()

autoencoder = MLPRegressor(
    # 17 -> 10 -> 4 -> 10 -> 17 darbogazi, modeli girdiyi kopyalamak yerine
    # ozet oruntu ogrenmeye zorlar. MLPRegressor yeniden uretici agdir.
    hidden_layer_sizes=(10, 4, 10),
    activation="tanh",
    solver="adam",
    max_iter=5000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=30,
    random_state=42,
)

df["prediction"] = -1
df["is_anomaly"] = True
df["ae_score"] = np.nan
df["anomaly_level"] = 0.0

if not X_egitim.empty:
    X_egitim_scaled = scaler.fit_transform(X_egitim)
    X_scaled = scaler.transform(X)

    # Hedefin girdiye esit verilmesi autoencoder'in yeniden uretim gorevidir.
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

hibrit_karari_uygula(
    score_column="ae_score",
    lower_scores_more_anomalous=False,
    default_threshold=float(threshold) if not X_egitim.empty else np.inf,
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
        [*MODEL_REPORT_COLUMNS, "ae_score"]
    ].head(30)
)

anomali_oranini_raporla(df, df.index, "Autoencoder")

result.drop(columns=CSV_HARIC_KOLONLAR).to_csv(
    "autoencoder_result.csv",
    index=False,
    encoding="utf-8-sig"
)
