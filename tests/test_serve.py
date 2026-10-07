"""HTTP contract checks for the inference API without cloud access."""

import asyncio
from io import BytesIO

import httpx
import joblib

from src import serve


class StubModel:
    def predict_proba(self, rows):
        assert len(rows[0]) == 10
        return [[0.3, 0.7]]


def test_score_and_health(monkeypatch):
    monkeypatch.setattr(serve, "download_model", lambda: (StubModel(), 0.65))

    async def check():
        async with serve.lifespan(serve.app):
            transport = httpx.ASGITransport(app=serve.app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                assert (await client.get("/healthz")).json() == {"status": "ok"}
                response = await client.post("/score", json={"features": [1] * 10})
                assert response.status_code == 200
                assert response.json() == {"prediction": 1, "label": "thu_nhap_cao"}

    asyncio.run(check())


def test_score_rejects_wrong_feature_count(monkeypatch):
    monkeypatch.setattr(serve, "download_model", lambda: (StubModel(), 0.65))

    async def check():
        async with serve.lifespan(serve.app):
            transport = httpx.ASGITransport(app=serve.app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post("/score", json={"features": [1, 2]})
                assert response.status_code == 400

    asyncio.run(check())


def test_download_uses_manifest_and_matching_report(tmp_path, monkeypatch):
    source_model = tmp_path / "source.joblib"
    joblib.dump(StubModel(), source_model)
    objects = {
        serve.MANIFEST_KEY: b'{"prefix":"artifacts/runs/123-1"}',
        "artifacts/runs/123-1/model.joblib": source_model.read_bytes(),
        "artifacts/runs/123-1/report.json": b'{"decision_threshold":0.3}',
    }

    class FakeS3:
        def get_object(self, Bucket, Key):
            assert Bucket == "lab-bucket"
            return {"Body": BytesIO(objects[Key])}

        def download_file(self, bucket, key, filename):
            assert bucket == "lab-bucket"
            with open(filename, "wb") as destination:
                destination.write(objects[key])

    monkeypatch.setenv("ARTIFACT_BUCKET", "lab-bucket")
    monkeypatch.setattr(serve.boto3, "client", lambda service: FakeS3())
    monkeypatch.setattr(serve, "MODEL_PATH", tmp_path / "models/model.joblib")
    monkeypatch.setattr(serve, "REPORT_PATH", tmp_path / "models/report.json")
    model, threshold = serve.download_model()
    assert threshold == 0.3
    assert model.predict_proba([[1] * 10])[0][1] == 0.7
