# DeepTrace API. Build from the REPO ROOT (it needs ml/src for the shared serving code):
#   docker build -f backend/Dockerfile -t deeptrace-backend .
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# CPU-only PyTorch for MTCNN (a CUDA build would add ~2 GB for nothing on CPU hosts)
COPY backend/requirements-torch.txt ./
RUN pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu \
 && pip install facenet-pytorch==2.6.0 --no-deps

COPY backend/requirements.txt ./
RUN pip install -r requirements.txt

# Shared preprocessing / heatmap / serving code — the same code used in training and evaluation
COPY ml/pyproject.toml ml/pyproject.toml
COPY ml/src ml/src
RUN pip install ./ml --no-deps

COPY backend/app app

RUN useradd --create-home --uid 1000 deeptrace \
 && mkdir -p /app/models /app/data/uploads \
 && chown -R deeptrace /app
USER deeptrace

ENV DEEPTRACE_MODEL_DIR=/app/models/current \
    DEEPTRACE_UPLOAD_DIR=/app/data/uploads \
    PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/v1/health', timeout=4)"

CMD ["sh", "-c", "uvicorn app.main:create_app --factory --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
