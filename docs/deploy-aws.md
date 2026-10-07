# AWS deployment checklist

The repository is configured for S3 and an EC2 Ubuntu VM in `us-east-1`. The private bucket is `vinai-day21-duy-2a202602913-060668531069`, the GitHub Actions role is `arn:aws:iam::060668531069:role/Day21GitHubActions`, and the server is `i-09055bcf39111bf5e` (`44.205.17.245` at launch). Never commit or paste credentials into this repository.

## One-time storage setup

1. The bucket was created with all four **Block Public Access** settings enabled. The CI role can list this bucket and read/write only `dvc/*` and `artifacts/*` objects. It has no AWS access key in GitHub.
2. The DVC remote `labstore` points to `s3://vinai-day21-duy-2a202602913-060668531069/dvc`. `DVC_SITE_CACHE_DIR=/tmp/dvc-day21 AWS_PROFILE=lab16 .venv/bin/dvc push` uploaded all three datasets, and `dvc status -c` confirmed the cache and remote are in sync. The cache override is needed only on this sandboxed workstation; GitHub runners do not need it. The three `.dvc` pointer files belong in Git, while the CSVs stay out of Git.
3. Verify the three data objects under the bucket's `dvc/` prefix in the S3 console for the submission screenshot.

## VM setup

1. The Ubuntu 24.04 `t3.micro` instance has profile `Day21IncomeEC2`. Its S3 policy reads only `artifacts/current/manifest.json`, `artifacts/runs/*`, and `deployment/serve.py`. `AmazonSSMManagedInstanceCore` lets AWS Systems Manager manage the VM without an access key or inbound SSH.
2. The dedicated security group exposes only TCP 8080 for the lab API. Port 22 is closed. The launch public IP is above; check the current IP again if the VM is stopped and restarted.
3. Cloud-init installed Python 3.12 and `~/income-venv` with `fastapi==0.111.0 uvicorn==0.29.0 scikit-learn==1.4.2 joblib==1.4.2 boto3==1.43.100`. It created `~/src`, `~/models`, and `/etc/systemd/system/income-api.service`. The service is enabled but waits for the first model release before starting.
4. The workflow uploads `src/serve.py` to S3, promotes the validated model, and sends an `AWS-RunShellScript` command to this one VM to download source, restart the service, and check `/healthz`.

## GitHub Actions secrets

The AWS IAM OIDC provider `token.actions.githubusercontent.com` and role `Day21GitHubActions` are configured. Its trust policy accepts only this GitHub repository's immutable subject ID on `main`, with audience `sts.amazonaws.com`. The role's policy is scoped to this bucket. The workflow uses `aws-actions/configure-aws-credentials` to obtain temporary credentials; it does not need a long-lived AWS access key in GitHub.

Required secrets: `AWS_ROLE_ARN`, `ARTIFACT_BUCKET` (bucket name only), and `SERVER_INSTANCE_ID`. Bonus 1 additionally uses `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, and `MLFLOW_TRACKING_PASSWORD` for DagsHub. Keep secret values out of logs and screenshots. The training job reads its new run back from DagsHub and compares F1 and accuracy before staging the model.

## Artifact and release flow

Train stages `model.joblib` and `report.json` at `artifacts/runs/<run-id>-<attempt>/`. Quality Gate requires positive-class F1 >= 0.65 and no regression against the currently released report. Release copies the validated pair to `artifacts/current/` for lab evidence, then writes `artifacts/current/manifest.json` last. The API follows the manifest to load the immutable model/report pair and its chosen decision threshold.

The workflow stages `src/serve.py` in S3 before promotion, then deploys it through Systems Manager, restarts `income-api`, and checks `/healthz`. Verify `/score` through the VM public IP after a successful release.

## Continuous training evidence

Capture the first successful four-job workflow and download its `report` artifact before changing data. Then run `python append_batch.py` once, `dvc add data/train_batch1.csv`, and `dvc push` **before** pushing the commit containing `data/train_batch1.csv.dvc`. The data-only push should trigger a second four-job workflow. Record both reports and screenshots before making later changes.
