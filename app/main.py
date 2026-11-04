
from contextlib import asynccontextmanager
from pathlib import Path
import json
import logging
import os

import joblib
import pandas as pd
import redis
import yaml

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = Path(
    os.getenv("MODEL_PATH", str(BASE_DIR / "artifacts" / "model.joblib"))
)
MODEL_CARD_PATH = Path(
    os.getenv("MODEL_CARD_PATH", str(BASE_DIR / "artifacts" / "model_card.json"))
)
PARAMS_PATH = Path(
    os.getenv("PARAMS_PATH", str(BASE_DIR / "params.yaml"))
)

MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
GIT_SHA = os.getenv("GIT_SHA", "unknown")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

FEATURES = [
    "mean_radius", "mean_texture", "mean_perimeter", "mean_area",
    "mean_smoothness", "mean_compactness", "mean_concavity",
    "mean_concave_points", "mean_symmetry", "mean_fractal_dimension",
    "radius_error", "texture_error", "perimeter_error", "area_error",
    "smoothness_error", "compactness_error", "concavity_error",
    "concave_points_error", "symmetry_error", "fractal_dimension_error",
    "worst_radius", "worst_texture", "worst_perimeter", "worst_area",
    "worst_smoothness", "worst_compactness", "worst_concavity",
    "worst_concave_points", "worst_symmetry", "worst_fractal_dimension",
]

model = None
model_card = {}
decision_threshold = 0.5
redis_client = None

REDIS_COUNTER_KEY = "prediction_count"


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mean_radius: float = Field(ge=0)
    mean_texture: float = Field(ge=0)
    mean_perimeter: float = Field(ge=0)
    mean_area: float = Field(ge=0)
    mean_smoothness: float = Field(ge=0)
    mean_compactness: float = Field(ge=0)
    mean_concavity: float = Field(ge=0)
    mean_concave_points: float = Field(ge=0)
    mean_symmetry: float = Field(ge=0)
    mean_fractal_dimension: float = Field(ge=0)

    radius_error: float = Field(ge=0)
    texture_error: float = Field(ge=0)
    perimeter_error: float = Field(ge=0)
    area_error: float = Field(ge=0)
    smoothness_error: float = Field(ge=0)
    compactness_error: float = Field(ge=0)
    concavity_error: float = Field(ge=0)
    concave_points_error: float = Field(ge=0)
    symmetry_error: float = Field(ge=0)
    fractal_dimension_error: float = Field(ge=0)

    worst_radius: float = Field(ge=0)
    worst_texture: float = Field(ge=0)
    worst_perimeter: float = Field(ge=0)
    worst_area: float = Field(ge=0)
    worst_smoothness: float = Field(ge=0)
    worst_compactness: float = Field(ge=0)
    worst_concavity: float = Field(ge=0)
    worst_concave_points: float = Field(ge=0)
    worst_symmetry: float = Field(ge=0)
    worst_fractal_dimension: float = Field(ge=0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_card, decision_threshold, redis_client

    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)

    if MODEL_CARD_PATH.is_file():
        with open(MODEL_CARD_PATH, "r", encoding="utf-8") as file:
            model_card = json.load(file)

    if PARAMS_PATH.is_file():
        with open(PARAMS_PATH, "r", encoding="utf-8") as file:
            params = yaml.safe_load(file) or {}
        decision_threshold = float(
            os.getenv(
                "DECISION_THRESHOLD",
                params.get("decision_threshold", 0.5)
            )
        )
    else:
        decision_threshold = float(
            os.getenv("DECISION_THRESHOLD", "0.5")
        )

    if not 0 <= decision_threshold <= 1:
        raise RuntimeError("DECISION_THRESHOLD must be between 0 and 1")

    redis_host = os.getenv("REDIS_HOST")

    if redis_host:
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1,
        )

    yield

    if redis_client is not None:
        redis_client.close()


app = FastAPI(
    title="Breast Cancer Triage API",
    description="ML-based breast cancer prediction service",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    input_data = pd.DataFrame([data.model_dump()], columns=FEATURES)

    probability = float(model.predict_proba(input_data)[0][1])
    prediction = "malignant" if probability >= decision_threshold else "benign"

    if redis_client is not None:
        try:
            redis_client.incr(REDIS_COUNTER_KEY)
        except redis.exceptions.RedisError:
            logger.warning("Redis unavailable; prediction counter not updated")

    return {
        "label": prediction,
        "malignant_probability": round(probability, 4),
        "threshold": decision_threshold,
        "model_version": MODEL_VERSION,
    }


@app.get("/model/info")
def model_info():
    return {
        **model_card,
        "model_version": MODEL_VERSION,
        "model_family": model_card.get("model_family", "unknown"),
        "sklearn_version": model_card.get("sklearn_version", "unknown"),
        "git_commit": GIT_SHA if GIT_SHA != "unknown" else model_card.get("git_commit", "unknown"),
        "threshold": decision_threshold,
        "feature_count": len(FEATURES),
        "test_metrics": model_card.get("metrics", {}).get("test", {}),
    }


@app.get("/stats")
def stats():
    if redis_client is None:
        return {
            "redis_available": False,
            "prediction_count": None,
        }

    try:
        count = redis_client.get(REDIS_COUNTER_KEY)
        return {
            "redis_available": True,
            "prediction_count": int(count) if count else 0,
        }
    except redis.exceptions.RedisError:
        return {
            "redis_available": False,
            "prediction_count": None,
        }


@app.get("/")
def root():
    return {
        "message": "Breast Cancer Triage API is running",
        "docs": "/docs",
    }
