from fastapi import FastAPI, HTTPException
import uvicorn
import os
import pickle
from pydantic import BaseModel
from typing import List

app = FastAPI(
    title="Credit Default Prediction API",
    description="API for predicting credit default probabilities using a pre-trained Random Forest model.",
    version="1.0.0",
)

# Placeholder for the loaded model and scaler
model = None
scaler = None

# Define the input data model using Pydantic
class CreditFeatures(BaseModel):
    # These feature names should match the training data used for the model
    # Assuming 6 features as per project mgmt.md, adjust as needed
    feature1: float
    feature2: float
    feature3: float
    feature4: float
    feature5: float
    feature6: float

@app.on_event("startup")
async def load_assets():
    global model, scaler
    model_path = os.path.join(os.getcwd(), "Models", "credit_scoring_model.pkl")
    scaler_path = os.path.join(os.getcwd(), "Models", "scaler.pkl")

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Please ensure it exists.")
        raise RuntimeError(f"Model file not found at {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError(f"Error loading model: {e}")

    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print(f"Scaler loaded successfully from {scaler_path}")
    except FileNotFoundError:
        print(f"Error: Scaler file not found at {scaler_path}. Please ensure it exists.")
        raise RuntimeError(f"Scaler file not found at {scaler_path}")
    except Exception as e:
        print(f"Error loading scaler: {e}")
        raise RuntimeError(f"Error loading scaler: {e}")

@app.get("/")
async def root():
    return {"message": "Credit Default Prediction API is running!"}

@app.post("/predict_default")
async def predict_default(features: CreditFeatures):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or Scaler not loaded. Server startup failed.")

    try:
        # Convert input features to a list of lists or numpy array for scaling and prediction
        # Ensure the order of features matches the order used during training
        input_data = [[
            features.feature1,
            features.feature2,
            features.feature3,
            features.feature4,
            features.feature5,
            features.feature6
        ]]

        # Scale the input features
        scaled_data = scaler.transform(input_data)

        # Make prediction
        prediction_proba = model.predict_proba(scaled_data)[:, 1] # Probability of default (class 1)

        return {"probability_of_default": prediction_proba[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 