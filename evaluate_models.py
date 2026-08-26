import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


PROJECT_ROOT = Path(__file__).resolve().parent
LABEL_ROOT = PROJECT_ROOT / "text_files_doğrulama"

LABEL_FILES = {
    "dogrulama": LABEL_ROOT / "dogrulama_etiket.json",
    "test": LABEL_ROOT / "test_etiket.json",
}

MODEL_RESULTS = {
    "Isolation Forest": {
        "path": PROJECT_ROOT / "isolation_forest_result.csv",
        "score_column": "if_score",
    },
    "Robust Covariance": {
        "path": PROJECT_ROOT / "robust_covariance_result.csv",
        "score_column": "cov_score",
    },
    "One-Class SVM": {
        "path": PROJECT_ROOT / "one_class_svm_result.csv",
        "score_column": "svm_score",
    },
    "Local Outlier Factor": {
        "path": PROJECT_ROOT / "lof_result.csv",
        "score_column": "lof_score",
    },
    "Autoencoder": {
        "path": PROJECT_ROOT / "autoencoder_result.csv",
        "score_column": "ae_score",
    },
}

METRICS_OUTPUT = PROJECT_ROOT / "model_metrics.csv"
PREDICTIONS_OUTPUT = PROJECT_ROOT / "model_prediction_comparison.csv"
TYPE_METRICS_OUTPUT = PROJECT_ROOT / "model_metrics_by_anomaly_type.csv"
LEADERBOARD_OUTPUT = PROJECT_ROOT / "model_siralamasi.csv"


def boolean_series(series: pd.Series, column_name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "false": False,
        "0": False,
    }
    invalid = ~normalized.isin(mapping)
    if invalid.any():
        values = sorted(normalized.loc[invalid].unique())
        raise ValueError(f"'{column_name}' kolonunda gecersiz boolean degerleri var: {values}")

    return normalized.map(mapping).astype(bool)


def load_labels() -> pd.DataFrame:
    frames = []

    for split_name, label_path in LABEL_FILES.items():
        if not label_path.exists():
            raise FileNotFoundError(f"Etiket dosyasi bulunamadi: {label_path}")

        with label_path.open(encoding="utf-8") as file:
            records = json.load(file)

        if not isinstance(records, list):
            raise ValueError(f"Etiket dosyasi liste icermeli: {label_path}")

        frame = pd.DataFrame(records)
        required = {"kayit_id", "is_anomali"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label_path.name} eksik kolonlar: {sorted(missing)}")

        if frame["kayit_id"].duplicated().any():
            duplicates = frame.loc[frame["kayit_id"].duplicated(), "kayit_id"].tolist()
            raise ValueError(f"{label_path.name} tekrar eden kayit_id: {duplicates[:10]}")

        frame["split"] = split_name
        frame["ground_truth_is_anomaly"] = boolean_series(
            frame["is_anomali"],
            "is_anomali",
        )
        frames.append(frame)

    labels = pd.concat(frames, ignore_index=True)
    if labels.duplicated(["kayit_id", "split"]).any():
        raise ValueError("Etiketlerde tekrar eden kayit_id/split cifti bulundu.")

    return labels


def metric_row(model_name: str, split_name: str, rows: pd.DataFrame) -> dict:
    y_true = rows["ground_truth_is_anomaly"].astype(bool)
    y_pred = rows["predicted_is_anomaly"].astype(bool)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=True,
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()

    return {
        "model": model_name,
        "split": split_name,
        "degerlendirilen_kayit": len(rows),
        "gercek_anomali": int(y_true.sum()),
        "tahmin_edilen_anomali": int(y_pred.sum()),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy_score(y_true, y_pred),
    }


def evaluate_model(model_name: str, config: dict, labels: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    result_path = config["path"]
    score_column = config["score_column"]
    if not result_path.exists():
        raise FileNotFoundError(f"Model sonuc dosyasi bulunamadi: {result_path}")

    required_columns = {
        "kayit_id",
        "split",
        "is_anomaly",
        "model_is_anomaly",
        "rule_based_anomaly",
        "rule_anomaly_reasons",
        "anomaly_level",
        "model_anomaly_level",
        "decision_threshold",
        "threshold_source",
        "calibration_validation_f1",
        score_column,
    }
    result = pd.read_csv(result_path, low_memory=False)
    missing = required_columns - set(result.columns)
    if missing:
        raise ValueError(f"{result_path.name} eksik kolonlar: {sorted(missing)}")
    result = result[list(required_columns)]

    evaluation = labels.merge(
        result,
        on=["kayit_id", "split"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing_predictions = evaluation.loc[evaluation["_merge"] != "both", "kayit_id"]
    if not missing_predictions.empty:
        raise ValueError(
            f"{model_name} icin etiketi olup tahmini olmayan kayitlar: "
            f"{missing_predictions.head(10).tolist()}"
        )

    evaluation = evaluation.drop(columns="_merge")
    evaluation["predicted_is_anomaly"] = boolean_series(
        evaluation["is_anomaly"],
        "is_anomaly",
    )
    evaluation["model_only_is_anomaly"] = boolean_series(
        evaluation["model_is_anomaly"],
        "model_is_anomaly",
    )
    evaluation["rule_based_anomaly"] = boolean_series(
        evaluation["rule_based_anomaly"],
        "rule_based_anomaly",
    )

    metrics = []
    for split_name in LABEL_FILES:
        split_rows = evaluation.loc[evaluation["split"] == split_name]
        metrics.append(metric_row(model_name, split_name, split_rows))
    metrics.append(metric_row(model_name, "dogrulama+test", evaluation))

    detail_columns = [
        "kayit_id",
        "fatura_no",
        "split",
        "ground_truth_is_anomaly",
        "anomali_turleri",
        "onay_durumu",
        "rule_based_anomaly",
        "rule_anomaly_reasons",
        "predicted_is_anomaly",
        "model_only_is_anomaly",
        "anomaly_level",
        "model_anomaly_level",
        "decision_threshold",
        "threshold_source",
        "calibration_validation_f1",
        score_column,
    ]
    detail = evaluation.reindex(columns=detail_columns).rename(
        columns={
            "predicted_is_anomaly": f"{model_name}_prediction",
            "model_only_is_anomaly": f"{model_name}_model_only_prediction",
            "anomaly_level": f"{model_name}_anomaly_level",
            "model_anomaly_level": f"{model_name}_model_only_level",
            "decision_threshold": f"{model_name}_threshold",
            "threshold_source": f"{model_name}_threshold_source",
            "calibration_validation_f1": f"{model_name}_validation_f1",
            score_column: f"{model_name}_raw_score",
        }
    )

    return metrics, detail


def combine_prediction_tables(details: list[pd.DataFrame]) -> pd.DataFrame:
    label_columns = [
        "kayit_id",
        "fatura_no",
        "split",
        "ground_truth_is_anomaly",
        "anomali_turleri",
        "onay_durumu",
        "rule_based_anomaly",
        "rule_anomaly_reasons",
    ]
    combined = details[0].copy()

    for detail in details[1:]:
        model_columns = [column for column in detail.columns if column not in label_columns]
        combined = combined.merge(
            detail[["kayit_id", "split", *model_columns]],
            on=["kayit_id", "split"],
            how="inner",
            validate="one_to_one",
        )

    return combined.sort_values(["split", "kayit_id"]).reset_index(drop=True)


def anomaly_type_metrics(comparison: pd.DataFrame) -> pd.DataFrame:
    prediction_columns = {
        model_name: f"{model_name}_prediction"
        for model_name in MODEL_RESULTS
    }
    prediction_columns["Kesin Kural Katmani"] = "rule_based_anomaly"
    anomaly_types = sorted(
        {
            anomaly_type
            for types in comparison["anomali_turleri"]
            if isinstance(types, list)
            for anomaly_type in types
        }
    )

    rows = []
    for split_name in [*LABEL_FILES, "dogrulama+test"]:
        split_rows = (
            comparison
            if split_name == "dogrulama+test"
            else comparison.loc[comparison["split"] == split_name]
        )
        for anomaly_type in anomaly_types:
            type_mask = split_rows["anomali_turleri"].apply(
                lambda types: isinstance(types, list) and anomaly_type in types
            )
            type_rows = split_rows.loc[type_mask]
            total = len(type_rows)
            if total == 0:
                continue

            for model_name, prediction_column in prediction_columns.items():
                caught = int(boolean_series(type_rows[prediction_column], prediction_column).sum())
                rows.append(
                    {
                        "model": model_name,
                        "split": split_name,
                        "anomali_turu": anomaly_type,
                        "gercek_anomali": total,
                        "yakalanan_anomali": caught,
                        "recall": caught / total,
                    }
                )

    result = pd.DataFrame(rows)
    result["recall"] = result["recall"].round(4)
    return result


def build_leaderboard(metrics_table: pd.DataFrame, split_name: str) -> pd.DataFrame:
    leaderboard = (
        metrics_table
        .loc[
            metrics_table["split"] == split_name,
            ["model", "precision", "recall", "f1", "accuracy"],
        ]
        .sort_values("f1", ascending=False)
        .reset_index(drop=True)
    )
    leaderboard.insert(0, "sira", leaderboard.index + 1)
    return leaderboard


def main() -> None:
    labels = load_labels()
    all_metrics = []
    all_details = []

    for model_name, config in MODEL_RESULTS.items():
        metrics, detail = evaluate_model(model_name, config, labels)
        all_metrics.extend(metrics)
        all_details.append(detail)

    metrics_table = pd.DataFrame(all_metrics)
    metric_columns = ["precision", "recall", "f1", "accuracy"]
    metrics_table[metric_columns] = metrics_table[metric_columns].round(4)
    metrics_table.to_csv(METRICS_OUTPUT, index=False, encoding="utf-8-sig")

    leaderboard = build_leaderboard(metrics_table, split_name="test")
    leaderboard.to_csv(LEADERBOARD_OUTPUT, index=False, encoding="utf-8-sig")

    comparison_table = combine_prediction_tables(all_details)
    type_metrics_table = anomaly_type_metrics(comparison_table)
    type_metrics_table.to_csv(TYPE_METRICS_OUTPUT, index=False, encoding="utf-8-sig")

    comparison_table["anomali_turleri"] = comparison_table["anomali_turleri"].apply(
        lambda value: "|".join(value) if isinstance(value, list) else ""
    )
    comparison_table.to_csv(PREDICTIONS_OUTPUT, index=False, encoding="utf-8-sig")

    display_columns = [
        "model",
        "split",
        "degerlendirilen_kayit",
        "gercek_anomali",
        "tahmin_edilen_anomali",
        "precision",
        "recall",
        "f1",
        "accuracy",
    ]
    print("\nMODEL DEGERLENDIRME SONUCLARI")
    print(metrics_table[display_columns].to_string(index=False))
    print("\nNIHAI SIRALAMA (test seti, F1'e gore)")
    print(leaderboard.to_string(index=False))
    print(f"\nMetrik tablosu: {METRICS_OUTPUT}")
    print(f"Anomali turu metrikleri: {TYPE_METRICS_OUTPUT}")
    print(f"Kayit bazli karsilastirma: {PREDICTIONS_OUTPUT}")
    print(f"Nihai siralama: {LEADERBOARD_OUTPUT}")
    print(
        "Egitim etiketlerinin yalnizca toplu anomali orani model onseli "
        "olarak kullanildi; satir etiketleri model fit adimina verilmedi."
    )


if __name__ == "__main__":
    main()
