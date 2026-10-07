"""Deterministic tests for release decisions."""

import json
from io import BytesIO

import pytest
from botocore.exceptions import ClientError

from src import artifacts

evaluate_release = artifacts.evaluate_release


def test_first_release_requires_threshold():
    evaluate_release(0.65, None)
    with pytest.raises(ValueError, match="Quality gate"):
        evaluate_release(0.649, None)


def test_existing_model_blocks_regression():
    evaluate_release(0.72, 0.72)
    evaluate_release(0.73, 0.72)
    with pytest.raises(ValueError, match="Rollback guard"):
        evaluate_release(0.71, 0.72)


def test_invalid_scores_fail_closed():
    with pytest.raises(ValueError):
        evaluate_release(float("nan"), None)
    with pytest.raises(ValueError):
        evaluate_release(0.7, float("nan"))


def test_stage_gate_and_promotion_keep_last_good_model(tmp_path, monkeypatch):
    class FakeS3:
        def __init__(self):
            self.objects = {}

        def upload_file(self, filename, bucket, key):
            self.objects[key] = (tmp_path / filename).read_bytes()

        def get_object(self, Bucket, Key):
            if Key not in self.objects:
                raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
            return {"Body": BytesIO(self.objects[Key])}

        def copy_object(self, Bucket, CopySource, Key):
            self.objects[Key] = self.objects[CopySource["Key"]]

        def put_object(self, Bucket, Key, Body, ContentType):
            self.objects[Key] = Body

    store = FakeS3()
    monkeypatch.setattr(artifacts, "s3_client", lambda: store)
    monkeypatch.setenv("ARTIFACT_BUCKET", "lab-bucket")
    monkeypatch.setenv("GITHUB_RUN_ID", "1")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "outputs").mkdir()
    (tmp_path / "models/model.joblib").write_bytes(b"first-model")
    (tmp_path / "outputs/report.json").write_text('{"f1_score": 0.72}')

    artifacts.stage()
    artifacts.gate()
    artifacts.promote()
    assert store.objects["artifacts/current/model.joblib"] == b"first-model"
    assert (
        json.loads(store.objects[artifacts.CURRENT_MANIFEST])["prefix"]
        == "artifacts/runs/1-1"
    )

    monkeypatch.setenv("GITHUB_RUN_ID", "2")
    (tmp_path / "models/model.joblib").write_bytes(b"worse-model")
    (tmp_path / "outputs/report.json").write_text('{"f1_score": 0.70}')
    artifacts.stage()
    with pytest.raises(ValueError, match="Rollback guard"):
        artifacts.promote()
    assert store.objects["artifacts/current/model.joblib"] == b"first-model"
