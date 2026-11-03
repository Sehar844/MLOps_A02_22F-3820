
from contextlib import asynccontextmanager
from pathlib import Path
import json
import os

import joblib
import pandas as pd
import redis

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "artifacts" / "model.joblib"
MODEL_CARD_PATH = BASE_DIR / "artifacts" / "model_card.json"
PARAMS_PATH = BASE_DIR / "params.yaml"


FEATURES = [
    "mean_radius",
    "mean_texture",
    "mean_perimeter",
    "mean_area",
    "mean_smoothness",
    "mean_compactness",
    "mean_concavity",
    "mean_concave_points",
    "mean_symmetry",
    "mean_fractal_dimension",
    "radius_error",
    "texture_error",
    "perimeter_error",
    "area_error",
    "smoothness_error",
    "compactness_error",
    "concavity_error",
    "concave_points_error",
    "symmetry_error",
    "fractal_dimension_error",
    "worst_radius",
    "worst_texture",
    "worst_perimeter",
    "worst_area",
    "worst_smoothness",
    "worst_compactness",
    "worst_concavity",
    "worst_concave_points",
    "worst_symmetry",
    "worst_fractal_dimension",
]


model = None
model_card = {}
decision_threshold = 0.5

redis_client = None
REDIS_COUNTER_KEY = "prediction_count"


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mean_radius: float = Field(gt=0)
    mean_texture: float = Field(gt=0)
    mean_perimeter: float = Field(gt=0)
    mean_area: float = Field(gt=0)
    mean_smoothness: float = Field(gt=0)
    mean_compactness: float = Field(ge=0)
    mean_concavity: float = Field(ge=0)
    mean_concave_points: float = Field(ge=0)
    mean_symmetry: float = Field(gt=0)
    mean_fractal_dimension: float = Field(gt=0)

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

    worst_radius: float = Field(gt=0)
    worst_texture: float = Field(gt=0)
    worst_perimeter: float = Field(gt=0)
    worst_area: float = Field(gt=0)
    worst_smoothness: float = Field(gt=0)
    worst_compactness: float = Field(ge=0)
    worst_concavity: float = Field(ge=0)
    worst_concave_points: float = Field(ge=0)
    worst_symmetry: float = Field(gt=0)
    worst_fractal_dimension: float = Field(gt=0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_card, decision_threshold, redis_client

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    if MODEL_CARD_PATH.exists():
        with open(MODEL_CARD_PATH, "r") as file:
            model_card = json.load(file)

    if PARAMS_PATH.exists():
        import yaml

        with open(PARAMS_PATH, "r") as file:
            params = yaml.safe_load(file)

        decision_threshold = float(
            params.get("decision_threshold", 0.5)
        )

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


app = FastAPI(
    title="Breast Cancer Triage API",
    description="ML-based breast cancer prediction service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(data: PredictionInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    input_data = pd.DataFrame(
        [data.model_dump()],
        columns=FEATURES,
    )

    probability = float(
        model.predict_proba(input_data)[0][1]
    )

    prediction = int(
        probability >= decision_threshold
    )

    # Increment Redis counter on every successful prediction
    # If Redis is unavailable, prediction should still work
    if redis_client is not None:
        try:
            redis_client.incr(REDIS_COUNTER_KEY)
        except redis.exceptions.RedisError:
            pass

    return {
        "prediction": prediction,
        "diagnosis": (
            "Malignant" if prediction == 1 else "Benign"
        ),
        "malignant_probability": round(probability, 4),
        "decision_threshold": decision_threshold,
        "model_version": model_card.get(
            "model_version", "1.0.0"
        ),
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


@app.get("/model/info")
def model_info():
    return model_card


@app.get("/")
def root():
    return {
        "message": "Breast Cancer Triage API is running",
        "docs": "/docs",
    }
