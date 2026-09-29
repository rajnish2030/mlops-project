import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING",
    "http://127.0.0.1:5000"
)

MODEL_URI = "models:/house-price-predictor@champion"

FEATURES = [
    "sqft",
    "bedrooms",
    "bathrooms",
    "age_years",
    "garage",
    "location_score"
]

mlflow.set_tracking_uri(TRACKING_URI)

model = mlflow.sklearn.load_model(MODEL_URI)

app = FastAPI(
    title="House Price Predictor",
    version="1.0.0"
)

class HouseFeatures(BaseModel):
    sqft: float = Field(..., gt=0, le=20000)
    bedrooms: int = Field(..., gt=0, le=20)
    bathrooms: float = Field(..., gt=0, le=20)
    age_years: int = Field(..., ge=0, le=200)
    garage: int = Field(..., ge=0, le=20)
    location_score: float = Field(..., ge=1, le=10)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": MODEL_URI
    }


@app.post("/predict")
def predict(features: HouseFeatures):

    input_df = pd.DataFrame(
        [[
            features.sqft,
            features.bedrooms,
            features.bathrooms,
            features.age_years,
            features.garage,
            features.location_score
        ]],
        columns=FEATURES
    )

    prediction = model.predict(input_df)[0]

    return {
        "predicted_price": round(float(prediction), 2)
    }


static_dir = BASE_DIR / "static"

app.mount(
    "/static",
    StaticFiles(directory=static_dir),
    name="static"
)


@app.get("/")
def frontend():
    return FileResponse(
        static_dir / "index.html"
    )
