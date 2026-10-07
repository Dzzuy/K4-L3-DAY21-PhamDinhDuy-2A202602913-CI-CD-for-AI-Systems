import json
import os

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from src.train import train

FEATURE_NAMES = [
    "age",
    "workclass",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]


def _make_temp_data(tmp_path):
    """
    Tao dataset nho voi cung schema Adult de su dung trong test.

    pytest cung cap `tmp_path` la mot thu muc tam thoi, tu dong xoa sau khi test ket thuc.
    Ham nay dung du lieu ngau nhien nen khong can ket noi cloud storage hay tai file CSV thuc.
    """
    rng = np.random.default_rng(0)
    n = 200

    X = rng.random((n, len(FEATURE_NAMES)))
    y = rng.integers(0, 2, size=n)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y
    train_path = str(tmp_path / "train.csv")
    eval_path = str(tmp_path / "holdout.csv")
    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path, index=False)
    return train_path, eval_path


@pytest.fixture(autouse=True)
def isolated_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())


def test_train_returns_float(tmp_path):
    """Kiem tra ham train() tra ve mot so thuc nam trong [0.0, 1.0]."""
    train_path, eval_path = _make_temp_data(tmp_path)

    f1 = train(
        {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
        data_path=train_path,
        eval_path=eval_path,
    )
    assert isinstance(f1, float)
    assert 0.0 <= f1 <= 1.0


def test_report_file_created(tmp_path):
    """Kiem tra file outputs/report.json duoc tao sau khi huan luyen."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("outputs/report.json")
    with open("outputs/report.json", encoding="utf-8") as report_file:
        report = json.load(report_file)
    model = joblib.load("models/model.joblib")
    holdout = pd.read_csv(eval_path)
    predictions = (
        model.predict_proba(holdout[FEATURE_NAMES])[:, 1]
        >= report["decision_threshold"]
    ).astype(int)
    assert report["f1_score"] == pytest.approx(f1_score(holdout["target"], predictions))
    assert report["accuracy"] == pytest.approx(
        accuracy_score(holdout["target"], predictions)
    )
    assert 0.1 <= report["decision_threshold"] <= 0.9
    assert 0.0 <= report["f1_at_0_5"] <= report["f1_score"]
    assert 0.0 <= report["positive_rate"] <= 1.0
    assert "Confusion matrix" in (tmp_path / "outputs/detail.txt").read_text()


def test_model_file_created(tmp_path):
    """Kiem tra file models/model.joblib duoc tao sau khi huan luyen."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("models/model.joblib")
    assert hasattr(joblib.load("models/model.joblib"), "predict")
