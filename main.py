from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import IsolationForest
import shap
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

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    confidence: float  # Confidence of the prediction
    feature_contributions: dict[str, float]  # Feature importance values
    model_version: str


@app.post("/predict", response_model=PredictionResponse)
def predict(data: InputData):
    try:
        # Convert input to a DataFrame
        input_dict = data.dict()
        input_df = pd.DataFrame([input_dict])

        # Log the input data types and sample
        logger.info(f"Input Data Types: {input_df.dtypes}")
        logger.info(f"Input Data Sample: {input_df.head()}")

        # Validate numeric columns
        num_cols = ["Total_Amount", "Total_Amount_to_Repay", "Amount_Funded_By_Lender", "Lender_portion_Funded", "Lender_portion_to_be_repaid"]
        input_df[num_cols] = input_df[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

        # Log cleaned input data
        logger.info(f"Cleaned Input Data: {input_df}")

        # Preprocess the data and make predictions
        prediction = pipeline.predict(input_df)
        probabilities = pipeline.predict_proba(input_df)[0]
        probability = max(probabilities)
        confidence = max(probabilities) - sorted(probabilities)[-2]  # Difference between top 2 probabilities

        # Feature contributions using model feature importances
        feature_importances = pipeline.named_steps['classifier'].feature_importances_
        feature_contributions = {
            feature: round(importance, 4)  # Rounded to 4 decimal places
            for feature, importance in zip(input_df.columns, feature_importances)
        }

        # Log the prediction results
        logger.info(f"Prediction: {prediction}, Probability: {probability}, Confidence: {confidence}")
        logger.info(f"Feature Contributions: {feature_contributions}")

        # Return the prediction result
        return PredictionResponse(
            prediction=int(prediction[0]),
            probability=float(probability),
            confidence=float(confidence),
            feature_contributions=feature_contributions,
            model_version="v1.0.0"
        )

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")
