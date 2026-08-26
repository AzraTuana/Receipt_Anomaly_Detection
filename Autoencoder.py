
from main import *
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

contamination = 0.28

scaler = StandardScaler()

autoencoder = MLPRegressor(
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

    autoencoder.fit(X_egitim_scaled, X_egitim_scaled)

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
