# Day 21 evidence capture

Keep each image below 1 MB. Browser captures must include the address bar, per `nop-bai/README.md`. Do not include credentials or secret values.

| File | Capture after | What must be visible | Can it be retaken later? |
|---|---|---|---|
| `01-mlflow-ui.png` | At least three local training runs | Run parameters, `f1_score`, `accuracy`, browser URL | Yes: saved MLflow runs can be reopened. The current image needs a retake with the URL bar. |
| `02-actions-buoc-2.png` | First successful baseline workflow | Four green jobs and workflow run identity | Yes: completed Actions runs remain in the repository history. |
| `03-actions-buoc-3.png` | Data-only commit triggers second successful workflow | Data commit title and four green jobs | Yes: completed Actions runs remain in history. |
| `04-curl-api.png` | First model is released and EC2 API is running | VM public IP, successful `/healthz` and `/score` calls | Only while the VM is running; capture before stopping it. |
| `05-cloud-storage.png` | First model is released | S3 console URL and bucket contents under `dvc/` and `artifacts/current/model.joblib` | Yes: objects remain after the VM stops. |

Before the first workflow run, confirm GitHub Actions secrets, DVC pull, EC2 service setup, and baseline data pointer. Capture the baseline workflow and report before appending batch 2. Push the new DVC data object before the data-only Git commit. Capture the second workflow and report before making later changes. Preserve a failed quality-gate run or its test evidence to show F1 below 0.65 blocks release.
