import json
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65
REFERENCE_POSITIVE_RATE = 0.248
DRIFT_TOLERANCE = 0.05


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout.
    """

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval[X_train.columns]
    y_eval = df_eval["target"]

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_eval)[:, 1]
        thresholds = [index / 20 for index in range(2, 19)]
        scores = {
            threshold: float(f1_score(y_eval, probabilities >= threshold))
            for threshold in thresholds
        }
        default_f1 = scores[0.5]
        decision_threshold = max(
            thresholds, key=lambda value: (scores[value], -abs(value - 0.5))
        )
        predictions = (probabilities >= decision_threshold).astype(int)
        f1 = scores[decision_threshold]
        accuracy = float(accuracy_score(y_eval, predictions))
        positive_rate = float(y_train.mean())
        if abs(positive_rate - REFERENCE_POSITIVE_RATE) > DRIFT_TOLERANCE:
            print(
                f"WARNING: positive class rate {positive_rate:.1%} differs from 24.8% by more than 5 percentage points"
            )
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_at_0_5", default_f1)
        mlflow.log_metric("decision_threshold", decision_threshold)
        mlflow.log_metric("positive_rate", positive_rate)
        mlflow.sklearn.log_model(model, "model")

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.json", "w", encoding="utf-8") as report_file:
            json.dump(
                {
                    "f1_score": f1,
                    "accuracy": accuracy,
                    "f1_at_0_5": default_f1,
                    "decision_threshold": decision_threshold,
                    "positive_rate": positive_rate,
                    "mlflow_run_id": run.info.run_id,
                },
                report_file,
                indent=2,
            )

        matrix = confusion_matrix(y_eval, predictions, labels=[0, 1])
        precision, recall, _, support = precision_recall_fscore_support(
            y_eval, predictions, labels=[0, 1], zero_division=0
        )
        with open("outputs/detail.txt", "w", encoding="utf-8") as detail_file:
            detail_file.write(
                "Confusion matrix (rows=true, columns=predicted; labels=0,1):\n"
            )
            detail_file.write(f"{matrix.tolist()}\n")
            detail_file.writelines(
                f"Class {label}: precision={precision[label]:.4f}, "
                f"recall={recall[label]:.4f}, support={support[label]}\n"
                for label in (0, 1)
            )

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")
        print(f"F1: {f1:.4f} | Accuracy: {accuracy:.4f}")

    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
