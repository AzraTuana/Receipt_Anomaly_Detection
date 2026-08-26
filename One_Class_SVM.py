
from main import *
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

#------------------------------------------------
#ONE-CLASS SVM MODELI
#------------------------------------------------
# Calisma prensibi: Etiketli anomali orneklerine ihtiyac duymadan egitim
# kayitlarini kapsayan bir karar siniri ogrenir. RBF cekirdegi kayitlari daha
# yuksek boyutlu bir uzayda karsilastirdigi icin dogrusal olmayan bir "normal
# bolge" kurabilir; sinirin disinda kalanlar anomali sayilir.
# Bu projedeki kullanim alani: Fis ozelliklerinin karmasik kombinasyonlarinda
# normal davranisin dis sinirini belirler. Global bir sinir kurdugundan yerel
# komsuluk anomalilerine LOF kadar dogrudan odaklanmaz.

# nu, egitimde anomali olabilecek kayitlar icin yaklasik ust siniri ve destek
# vektoru payi icin alt siniri belirleyen temel karar-siniri parametresidir.
nu = 0.28

# SVM uzakliklara duyarlidir; parasal ve ikili ozelliklerin esit kosullarda
# degerlendirilmesi icin olcekleme yalnizca egitim verisine uydurulur.
scaler = StandardScaler()

model = OneClassSVM(
    kernel="rbf",
    nu=nu,
    gamma="scale"
)

# Eksik/gecersiz verili satirlar modele gonderilmeden dogrudan anomali
# olarak atanir. svm_score bos kalir; cunku bu satirlar model tarafindan
# skorlanmamistir.
df["prediction"] = -1
df["is_anomaly"] = True
df["svm_score"] = np.nan
df["anomaly_level"] = 0.0

if not X_egitim.empty:
    X_egitim_scaled = scaler.fit_transform(X_egitim)
    X_scaled = scaler.transform(X)

    # Model, olceklenmis egitim kayitlarini normal sinifin ornekleri kabul eder.
    model.fit(X_egitim_scaled)

    # decision_function: pozitif -> sinirin icinde (normal),
    # negatif -> sinirin disinda (anomali).
    skor = model.decision_function(X_scaled)

    df.loc[X.index, "svm_score"] = skor

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
    score_column="svm_score",
    lower_scores_more_anomalous=True,
    default_threshold=0.0,
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
        [*MODEL_REPORT_COLUMNS, "svm_score"]
    ].head(30)
)

anomali_oranini_raporla(df, df.index, "One-Class SVM")

result.drop(columns=CSV_HARIC_KOLONLAR).to_csv(
    "one_class_svm_result.csv",
    index=False,
    encoding="utf-8-sig"
)
