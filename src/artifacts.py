"""Stage, check, and promote immutable training artifacts in S3."""

import json
import math
import os
import sys
from typing import Any

import boto3
from botocore.exceptions import ClientError

F1_THRESHOLD = 0.65
CURRENT_MANIFEST = "artifacts/current/manifest.json"


def candidate_prefix() -> str:
    run_id = os.environ["GITHUB_RUN_ID"]
    attempt = os.environ["GITHUB_RUN_ATTEMPT"]
    return f"artifacts/runs/{run_id}-{attempt}"


def s3_client() -> Any:
    return boto3.client("s3")


def bucket_name() -> str:
    return os.environ["ARTIFACT_BUCKET"]


def read_json(client: Any, bucket: str, key: str) -> dict:
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(body)


def evaluate_release(new_f1: float, current_f1: float | None) -> None:
    """Fail closed for invalid, below-threshold, or regressing scores."""
    if not math.isfinite(new_f1) or new_f1 < F1_THRESHOLD:
        raise ValueError(f"Quality gate failed: new F1 {new_f1} < {F1_THRESHOLD}")
    if current_f1 is not None and (
        not math.isfinite(current_f1) or new_f1 < current_f1
    ):
        raise ValueError(f"Rollback guard: new F1 {new_f1} < current F1 {current_f1}")


def current_report(client: Any, bucket: str) -> dict | None:
    try:
        manifest = read_json(client, bucket, CURRENT_MANIFEST)
    except ClientError as error:
        if error.response["Error"]["Code"] in {"404", "NoSuchKey"}:
            return None
        raise
    return read_json(client, bucket, f"{manifest['prefix']}/report.json")


def stage() -> None:
    client = s3_client()
    bucket = bucket_name()
    prefix = candidate_prefix()
    client.upload_file("models/model.joblib", bucket, f"{prefix}/model.joblib")
    client.upload_file("outputs/report.json", bucket, f"{prefix}/report.json")
    print(f"Staged candidate at s3://{bucket}/{prefix}")


def gate() -> None:
    client = s3_client()
    bucket = bucket_name()
    candidate = read_json(client, bucket, f"{candidate_prefix()}/report.json")
    previous = current_report(client, bucket)
    new_f1 = float(candidate["f1_score"])
    old_f1 = float(previous["f1_score"]) if previous else None
    print(f"Quality comparison: new F1={new_f1}, current F1={old_f1}")
    evaluate_release(new_f1, old_f1)


def promote() -> None:
    gate()
    client = s3_client()
    bucket = bucket_name()
    prefix = candidate_prefix()
    for name in ("model.joblib", "report.json"):
        client.copy_object(
            Bucket=bucket,
            CopySource={"Bucket": bucket, "Key": f"{prefix}/{name}"},
            Key=f"artifacts/current/{name}",
        )
    client.put_object(
        Bucket=bucket,
        Key=CURRENT_MANIFEST,
        Body=json.dumps({"prefix": prefix}).encode(),
        ContentType="application/json",
    )
    print(f"Promoted {prefix} to current")


if __name__ == "__main__":
    commands = {"stage": stage, "gate": gate, "promote": promote}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        raise SystemExit("Usage: python -m src.artifacts {stage|gate|promote}")
    commands[sys.argv[1]]()
