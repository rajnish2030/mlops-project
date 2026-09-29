import os
import pandas as pd
import boto3
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np

BUCKET = "house-prediction-11"

KEY = "ml-house-predition/2026-09-25/Mlops_house_predication_raw_data(1).csv"

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING",
    "http://127.0.0.1:5000"
)

MODEL_NAME = "house-price-predictor"

FEATURES = [
    "sqft",
    "bedrooms",
    "bathrooms",
    "age_years",
    "garage",
    "location_score"
]

TARGET = "price"


def fetch_data():
    s3 = boto3.client("s3")

    obj = s3.get_object(
        Bucket=BUCKET,
        Key=KEY
    )

    df = pd.read_csv(
        StringIO(
            obj["Body"].read().decode("utf-8")
        )
    )

    return df


mlflow.set_tracking_uri(TRACKING_URI)

mlflow.set_experiment("mlops-house-prediction")

df = fetch_data()

print(f"Fetched shape: {df.shape}")
print(df.head())

X = df[FEATURES]

y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

with mlflow.start_run() as run:

    n_estimators = 150

    max_depth = 8

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    preds = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        preds
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            preds
        )
    )

    r2 = r2_score(
        y_test,
        preds
    )

    mlflow.log_param(
        "model_type",
        "RandomForestRegressor"
    )

    mlflow.log_param(
        "n_estimators",
        n_estimators
    )

    mlflow.log_param(
        "max_depth",
        max_depth
    )

    mlflow.log_param(
        "data_source",
        f"s3://{BUCKET}/{KEY}"
    )

    mlflow.log_metric(
        "mae",
        mae
    )

    mlflow.log_metric(
        "rmse",
        rmse
    )

    mlflow.log_metric(
        "r2_score",
        r2
    )

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        registered_model_name=MODEL_NAME,
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ]
    )

    print()
    print(f"Run ID: {run.info.run_id}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")
    print()
    print("Model training completed successfully.")


client = MlflowClient(
    tracking_uri=TRACKING_URI
)

versions = client.search_model_versions(
    f"name='{MODEL_NAME}'"
)

if not versions:
    raise RuntimeError(
        f"No registered model versions found for {MODEL_NAME}"
    )

latest_version = max(
    versions,
    key=lambda version: int(version.version)
)

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=latest_version.version
)

print()
print(f"Model registered: {MODEL_NAME}")
print(
    f"Champion alias assigned to version: "
    f"{latest_version.version}"
)