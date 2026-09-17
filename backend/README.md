# DeepTrace API (FastAPI)

Serves the exported model bundle (`model.onnx` + `model_card.json` + `preprocess.json` +
`bundle_manifest.json`). Preprocessing, decision rule and heatmaps come from the shared
`deeptrace_ml` package, so the API runs exactly the code that was evaluated.

## Run locally (Windows PowerShell)

```powershell
cd deeptrace\backend
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install facenet-pytorch==2.6.0 --no-deps
.venv\Scripts\python.exe -m pip install -e ..\ml --no-deps
$env:DEEPTRACE_MODEL_DIR = "..\ml\artifacts\effnet_b0"     # an exported bundle
.venv\Scripts\python.exe -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/api/v1/docs. Without a bundle the API still starts: `/health` reports
`degraded` with the reason, and `/predict` returns `503 MODEL_NOT_LOADED`.

## Endpoints

| Method | Path | Notes |
|---|---|---|
| POST | `/api/v1/predict` | multipart `file`; `store` (default false), `explain` (default true) |
| GET | `/api/v1/model-info` | model card without internal heatmap parameters |
| GET | `/api/v1/health` | model + database status |
| POST | `/api/v1/auth/register`, `/api/v1/auth/login` | only with `DEEPTRACE_DATABASE_URL` + `DEEPTRACE_JWT_SECRET` |
| GET/DELETE | `/api/v1/history`, `/api/v1/history/{id}`, `/api/v1/history/{id}/image` | owner only |

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `DEEPTRACE_MODEL_DIR` | `models/current` | Local bundle directory |
| `DEEPTRACE_MODEL_REPO_ID` / `DEEPTRACE_MODEL_REVISION` | – / `main` | Download the bundle from the Hugging Face Hub if the directory is empty |
| `DEEPTRACE_DATABASE_URL` | – | e.g. `postgresql+asyncpg://user:pass@db:5432/deeptrace`; enables accounts + history |
| `DEEPTRACE_JWT_SECRET` | – | Required with a database; ≥ 32 characters |
| `DEEPTRACE_RATE_LIMIT_REQUESTS` / `_WINDOW_SECONDS` | 10 / 60 | Per-client sliding window |
| `DEEPTRACE_TRUST_FORWARDED_FOR` | false | Use `X-Forwarded-For` for the client IP (only behind your own proxy) |
| `DEEPTRACE_RETENTION_DAYS` | 30 | Stored history images are purged after this |
| `DEEPTRACE_FRONTEND_DIR` | – | Serve the built React app (single-container deployments) |
| `DEEPTRACE_CORS_ORIGINS` | `["http://localhost:5173"]` | JSON list |

## Tests

```powershell
.venv\Scripts\python.exe -m pytest
```

29 tests. They build a tiny real ONNX model with the same output contract as the exported CNNs and use
a stub face detector, so they need neither PyTorch nor a trained model. History tests run on SQLite.
