from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd

app = FastAPI(title="House Price Predictor")

# CORS — agar frontend alag domain se call kare toh zaroori
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # production mein specific domain daalna
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_URI = "models:/house-price-predictor@champion"
model = mlflow.sklearn.load_model(MODEL_URI)

class HouseFeatures(BaseModel):
    sqft: float
    bedrooms: int
    bathrooms: int
    age_years: int
    garage: int
    location_score: float

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": MODEL_URI}

@app.post("/predict")
def predict(features: HouseFeatures):
    input_df = pd.DataFrame([features.dict()])
    prediction = model.predict(input_df)[0]
    return {"predicted_price": round(float(prediction), 2)}

# Frontend serve
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")