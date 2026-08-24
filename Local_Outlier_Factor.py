
from main import *
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

#------------------------------------------------
#LOCAL OUTLIER FACTOR (LOF) MODELI
#------------------------------------------------
# LOF, her kaydın yerel komşuluğuna göre ne kadar "seyrek" bir bölgede
# olduğunu ölçer: bir noktanın komşularının yoğunluğu, o komşuların kendi
# komşularının yoğunluğuyla karşılaştırılır. Kendi çevresine göre belirgin
# şekilde daha seyrek bir bölgede kalan kayıtlar anomali sayılır. Isolation
# Forest'tan farkı, global değil YEREL bir kıyaslama yapması: farklı
# yoğunluktaki kümeler (ör. sık tekrar eden vs. nadir işletmeler) bir arada
# olsa bile her kayıt kendi komşuluğuna göre değerlendirilir.
#
# LOF, komşuluk hesabı için Öklid mesafesi kullanır; bu yüzden features
# arasındaki ölçek farkı (days_since_latest yüzlerce/binlerce iken
# total_log ve company_deviation_log birkaç birimlik aralıkta) sonucu
# tek başına domine eder. Bu yüzden StandardScaler ile ölçeklendirme
# şart (Isolation Forest'ta ağaç tabanlı bölünme ölçekten etkilenmediği
# için buna gerek yoktu).
#
# novelty=False (varsayılan) modunda LOF sadece fit_predict destekler;
# ayrı bir predict() yoktur çünkü model "yeni" veri üzerinde değil,
# eğitildiği verinin kendi içindeki yerel yoğunluk farklarını ölçer.

contamination = 0.02

scaler = StandardScaler()

model = LocalOutlierFactor(
    n_neighbors=20,
    contamination=contamination,
    n_jobs=-1
)

# Eksik/geçersiz verili satırlar modele gönderilmeden doğrudan anomali
# olarak atanır. lof_score boş kalır; çünkü bu satırlar model tarafından
# skorlanmamıştır.
df["prediction"] = -1
df["is_anomaly"] = True
df["lof_score"] = np.nan
df["anomaly_level"] = 100.0

if not X.empty:
    X_scaled = scaler.fit_transform(X)

    valid_predictions = model.fit_predict(X_scaled)
    # negative_outlier_factor_: normal kayıtlarda ~-1'e yakın, anomalilerde
    # daha küçük (daha negatif) değerler alır.
    valid_scores = model.negative_outlier_factor_

    df.loc[valid_model_mask, "prediction"] = valid_predictions
    df.loc[valid_model_mask, "is_anomaly"] = valid_predictions == -1
    df.loc[valid_model_mask, "lof_score"] = valid_scores

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
            "lof_score",
            "is_anomaly"
        ]
    ].head(30)
)

result.to_csv(
    "lof_result.csv",
    index=False,
    encoding="utf-8-sig"
)
