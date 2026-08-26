
from main import *
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

#------------------------------------------------
#LOCAL OUTLIER FACTOR (LOF) MODELI
#------------------------------------------------
# Calisma prensibi: Her kaydin yerel yogunlugunu en yakin komsularinin
# yogunluguyla karsilastirir. Komsularina gore belirgin bicimde daha seyrek bir
# bolgede kalan kaydin LOF aykirilik skoru artar; boylece global olarak normal
# gorunen fakat kendi alt grubunda olagandisi olan kayitlar bulunabilir.
# Bu projedeki kullanim alani: Benzer fis profillerinin olusturdugu yerel
# kumelerdeki aykirilari yakalar; ornegin genel veri icin makul olsa da yakin
# tutar/kategori profiline sahip kayitlardan kopan bir faturayi one cikarir.

# Skorlanan kayitlarin yaklasik bu payinin aykiri olacagi varsayilir.
contamination = 0.28

# Komsuluk uzakliklarinin parasal ozelliklerin buyuklugunce domine edilmemesi
# icin olcekleme egitim verisinden ogrenilir.
scaler = StandardScaler()

# novelty=True: egitimde fit edilip baska bir kayit setini (dogrulama/test)
# skorlayabilmek icin gerekli; varsayilan novelty=False sadece fit_predict
# (tek veri seti) destekler.
model = LocalOutlierFactor(
    # Yerel yogunluk her kaydin 20 en yakin egitim komsusundan hesaplanir.
    n_neighbors=20,
    contamination=contamination,
    novelty=True,
    n_jobs=-1
)

# Eksik/gecersiz verili satirlar modele gonderilmeden dogrudan anomali
# olarak atanir. lof_score bos kalir; cunku bu satirlar model tarafindan
# skorlanmamistir.
df["prediction"] = -1
df["is_anomaly"] = True
df["lof_score"] = np.nan
df["anomaly_level"] = 0.0

if not X_egitim.empty:
    X_egitim_scaled = scaler.fit_transform(X_egitim)
    X_scaled = scaler.transform(X)

    # novelty=True sayesinde egitim komsuluk yapisi daha sonra tum splitleri skorlar.
    model.fit(X_egitim_scaled)

    # score_samples: normal kayitlarda ~-1'e yakin, anomalilerde
    # daha kucuk (daha negatif) degerler alir (negative_outlier_factor_ ile ayni yon).
    skor = model.score_samples(X_scaled)

    df.loc[X.index, "lof_score"] = skor

    raw_anomaly = -skor
    minimum = raw_anomaly.min()
    maximum = raw_anomaly.max()

    if maximum != minimum:
        anomaly_level = (
            (raw_anomaly - minimum)
            / (maximum - minimum)
        ) * 100
    else:
        anomaly_level = np.zeros(len(X))

    df.loc[X.index, "anomaly_level"] = anomaly_level

hibrit_karari_uygula(
    score_column="lof_score",
    lower_scores_more_anomalous=True,
    default_threshold=float(model.offset_),
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
        [*MODEL_REPORT_COLUMNS, "lof_score"]
    ].head(30)
)

anomali_oranini_raporla(df, df.index, "Local Outlier Factor")

result.drop(columns=CSV_HARIC_KOLONLAR).to_csv(
    "lof_result.csv",
    index=False,
    encoding="utf-8-sig"
)
