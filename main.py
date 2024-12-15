from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Union
import numpy as np
import joblib
import pandas as pd
from loguru import logger
from scripts import DatetimeFeatureExtractor, LabelEncoderTransformer

# Load the full pipeline (preprocessing + model)
pipeline = joblib.load(r'pipelines/final_model_pipeline.pkl')  # Use forward slashes for better compatibility

# Create FastAPI instance
app = FastAPI()

# Define request data schema using Pydantic
class InputData(BaseModel):
    loan_type: str
    Total_Amount: float
    Total_Amount_to_Repay: float
    disbursement_date: str  # Accept as string for datetime transformations in the pipeline
    due_date: str           # Accept as string for datetime transformations
    duration: int
    New_versus_Repeat: str
    Amount_Funded_By_Lender: float
    Lender_portion_Funded: float
    Lender_portion_to_be_repaid: float

# Define response schema
class PredictionResponse(BaseModel):
    prediction: int
    probability: float

@app.post("/predict", response_model=PredictionResponse)
def predict(data: InputData):
    try:
        # Convert input to a dictionary and prepare for pipeline
        input_dict = data.dict()

        # Convert to DataFrame (as expected by the pipeline)
        input_df = pd.DataFrame([input_dict])

        # Log incoming data for debugging
        logger.info(f"Input Data: {input_df}")

        # Preprocess the data and make predictions
        prediction = pipeline.predict(input_df)
        probability = max(pipeline.predict_proba(input_df)[0])

        # Log the results
        logger.info(f"Prediction: {prediction}, Probability: {probability}")

        # Return the prediction result
        return PredictionResponse(prediction=int(prediction[0]), probability=float(probability))

    except Exception as e:
        # Log the error
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")
