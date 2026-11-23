from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field

# Load the trained model once when the API starts.
MODEL_PATH = Path(__file__).resolve().parent.parent / "iris_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run train_model.py first."
            )

model = joblib.load(MODEL_PATH)

app = FastAPI(
        title="Iris Classification API",
        description="Predict Iris flower species using a trained KNN model.",
        version="1.0.0"
        )

species = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
        }

# Define the expected request body.
class FlowerInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict_flower(flower: FlowerInput):

    # Convert the request into a DataFrame with the original
    # Iris dataset's feature names and column order.
    measurements = pd.DataFrame([{
        "sepal length (cm)": flower.sepal_length,
        "sepal width (cm)": flower.sepal_width,
        "petal length (cm)": flower.petal_length,
        "petal width (cm)": flower.petal_width
        }])

    prediction = model.predict(measurements)[0]
    probabilities = model.predict_proba(measurements)[0]

    probability_results = {
        species[int(class_id)]: round(float(probability), 4)
        for class_id, probability in zip(model.classes_, probabilities)
        }
    return {
            "predicted_species": species[int(prediction)],
            "probabilities": probability_results
            }
