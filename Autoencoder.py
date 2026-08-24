
from main import *
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

#------------------------------------------------
#AUTOENCODER MODELI
#------------------------------------------------
# Autoencoder, X'i kendi kendine yeniden üretmeye (reconstruct) çalışan bir
# sinir ağıdır: girdi -> sıkıştırma (encoder) -> darboğaz -> geri açma
# (decoder) -> çıktı. Ağ, normal kayıtların örüntüsünü öğrenip düşük hatayla
# yeniden üretebildiği için normal örüntüye uymayan (anomali) kayıtlarda
# yeniden üretim hatası (reconstruction error) belirgin şekilde yükselir.
# Bu hata, Isolation Forest'taki if_score'un karşılığı olarak kullanılıyor.
#
# TensorFlow/PyTorch bu ortamda kurulu olmadığından, scikit-learn'ün
# MLPRegressor'ı ile hafif bir autoencoder kuruluyor: girdi katmanı X'i,
# çıktı katmanı da yine X'i hedefliyor (fit(X, X)), aradaki tek gizli
# katman (2 nörön) darboğaz görevi görüyor. 3 özellikli küçük bir veri
# setinde daha derin bir mimari gereksiz karmaşıklık katardı.
#
# contamination, Isolation Forest ile aynı (0.02) tutuluyor; böylece iki
# modelin işaretlediği anomali SAYISI karşılaştırılabilir kalıyor.

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
            "ae_score",
            "is_anomaly"
        ]
    ].head(30)
)

result.to_csv(
    "autoencoder_result.csv",
    index=False,
    encoding="utf-8-sig"
)
