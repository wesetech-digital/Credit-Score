from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib

# Load the full pipeline (preprocessing + model)
pipeline = joblib.load('pipelines\pipeline_20241213_162715.pkl')

# Create FastAPI instance
app = FastAPI()

# Define request data schema using Pydantic
class InputData(BaseModel):
    Lender_portion_to_be_repaid: float
    Total_Amount_to_Repay: float
    Total_Amount: float
    Amount_Funded_By_Lender: float
    customer_id: float
    tbl_loan_id: float
    loan_type_Type_1: float
    duration: float
    due_date_day: float
    Lender_portion_Funded: float
    disbursement_date_day: float
    due_date_month: float
    lender_id: float
    disbursement_date_month: float
    disbursement_date_year: float
    due_date_year: float
    New_versus_Repeat: float
    loan_type_Type_7: float
    loan_type_Type_4: float
    loan_type_Type_6: float

# Define response schema
class PredictionResponse(BaseModel):
    prediction: int
    probability: float

# Define a prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(data: InputData):
    print(data)
    try:
        # Convert the incoming data into an array for prediction
        input_array = np.array([[data.Lender_portion_to_be_repaid, data.Total_Amount_to_Repay, data.Total_Amount,
                                 data.Amount_Funded_By_Lender, data.customer_id, data.tbl_loan_id, data.loan_type_Type_1,
                                 data.duration, data.due_date_day, data.Lender_portion_Funded, data.disbursement_date_day,
                                 data.due_date_month, data.lender_id, data.disbursement_date_month, data.disbursement_date_year,
                                 data.due_date_year, data.New_versus_Repeat, data.loan_type_Type_7, data.loan_type_Type_4,
                                 data.loan_type_Type_6]])

        # Preprocess the data using the pipeline
        new_data_preprocessed = pipeline.transform(input_array)
        input(new_data_preprocessed)
        # Make the prediction
        prediction = loaded_model.predict(new_data_preprocessed)[0]
        probability = max(loaded_model.predict_proba(new_data_preprocessed)[0])

        # Return the prediction result
        return PredictionResponse(prediction=int(prediction), probability=float(probability))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")
