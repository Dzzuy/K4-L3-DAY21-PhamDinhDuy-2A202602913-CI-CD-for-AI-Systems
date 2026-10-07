"""Income inference API. Download the released model at startup."""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import boto3
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MANIFEST_KEY = "artifacts/current/manifest.json"
MODEL_PATH = Path.home() / "models" / "model.joblib"
REPORT_PATH = Path.home() / "models" / "report.json"


def download_model() -> tuple[object, float]:
    """Load a model and its matching threshold from S3."""
    bucket_name = os.environ["ARTIFACT_BUCKET"]
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    manifest = json.loads(
        s3.get_object(Bucket=bucket_name, Key=MANIFEST_KEY)["Body"].read()
    )
    prefix = manifest["prefix"]
    s3.download_file(bucket_name, f"{prefix}/model.joblib", str(MODEL_PATH))
    s3.download_file(bucket_name, f"{prefix}/report.json", str(REPORT_PATH))
    with REPORT_PATH.open(encoding="utf-8") as report_file:
        report = json.load(report_file)
    return joblib.load(MODEL_PATH), float(report["decision_threshold"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.decision_threshold = download_model()
    yield


app = FastAPI(lifespan=lifespan)


class ScoreRequest(BaseModel):
    features: list[float]


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/score")
async def score(req: ScoreRequest) -> dict[str, int | str]:
    if len(req.features) != 10:
        raise HTTPException(
            status_code=400, detail="Expected 10 features (adult income)"
        )
    probability = float(app.state.model.predict_proba([req.features])[0][1])
    prediction = int(probability >= app.state.decision_threshold)
    return {
        "prediction": prediction,
        "label": "thu_nhap_cao" if prediction else "thu_nhap_thap",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
