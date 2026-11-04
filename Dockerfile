FROM python:3.13-slim@sha256:ae14d4ce8771606477a81446e7ef429edd6762305fc50c84507c1041fa8285d1 AS training

WORKDIR /app

ARG MODEL_VERSION=1.0.1

COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

COPY training/ ./training/
COPY data/raw/breast_cancer.csv ./data/raw/breast_cancer.csv
COPY params.yaml .

RUN python training/train.py \
    --data-path data/raw/breast_cancer.csv \
    --output-dir artifacts \
    --model-version "${MODEL_VERSION}"


FROM python:3.13-slim@sha256:ae14d4ce8771606477a81446e7ef429edd6762305fc50c84507c1041fa8285d1 AS runtime

WORKDIR /app

ARG MODEL_VERSION=1.0.1
ARG GIT_SHA=unknown

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/artifacts/model.joblib \
    MODEL_CARD_PATH=/app/artifacts/model_card.json \
    PARAMS_PATH=/app/params.yaml \
    MODEL_VERSION=${MODEL_VERSION} \
    GIT_SHA=${GIT_SHA} \
    DECISION_THRESHOLD=0.5 \
    LOG_LEVEL=INFO \
    REDIS_PORT=6379

LABEL org.opencontainers.image.title="Breast Cancer Triage API"
LABEL org.opencontainers.image.version="${MODEL_VERSION}"
LABEL org.opencontainers.image.revision="${GIT_SHA}"
LABEL org.opencontainers.image.description="Containerized breast cancer ML prediction service"

RUN useradd --create-home --uid 10001 appuser

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY --chown=appuser:appuser app/ ./app/
COPY --from=training --chown=appuser:appuser /app/artifacts/model.joblib ./artifacts/model.joblib
COPY --from=training --chown=appuser:appuser /app/artifacts/model_card.json ./artifacts/model_card.json
COPY --chown=appuser:appuser params.yaml ./params.yaml

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

ENTRYPOINT ["uvicorn"]
CMD ["app.main:app", "--host", "0.0.0.0", "--port", "8000"]
