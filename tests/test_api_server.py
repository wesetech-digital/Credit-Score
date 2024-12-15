import pytest
from fastapi.testclient import TestClient
from main import app  # Replace with the actual path to your FastAPI app
import pandas as pd
import joblib

client = TestClient(app)

# Valid input data
valid_input_data = {
    "loan_type": "Type_1",
    "Total_Amount": 10000.0,
    "Total_Amount_to_Repay": 12000.0,
    "disbursement_date": "2023-01-01",
    "due_date": "2024-01-01",
    "duration": 12,
    "New_versus_Repeat": "New",
    "Amount_Funded_By_Lender": 8000.0,
    "Lender_portion_Funded": 6000.0,
    "Lender_portion_to_be_repaid": 7000.0
}

# Invalid input data (e.g., missing field)
invalid_input_data = {
    "loan_type": "Type_h1",
    "Total_Amount": 10000.0,
    "Total_Amount_to_Repay": 12000.0,
    "disbursement_date": "2023-01-01",
    "due_date": "2024-01-01",
    "duration": 12,
    "New_versus_Repeat": "New",
    "Amount_Funded_By_Lender": 8000.0
    # Missing fields
}

# Edge case (e.g., very small or large amounts)
edge_case_input_data = {
    "loan_type": "Type_1",
    "Total_Amount": 0.01,
    "Total_Amount_to_Repay": 0.02,
    "disbursement_date": "2023-01-01",
    "due_date": "2024-01-01",
    "duration": 1,
    "New_versus_Repeat": "Repeat",
    "Amount_Funded_By_Lender": 0.01,
    "Lender_portion_Funded": 0.01,
    "Lender_portion_to_be_repaid": 0.02
}

# Test for valid data
def test_predict_valid_data():
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probability" in response.json()
    assert isinstance(response.json()['prediction'], int)
    assert isinstance(response.json()['probability'], float)

# Test for invalid data (missing fields)
def test_predict_invalid_data():
    response = client.post("/predict", json=invalid_input_data)
    assert response.status_code == 422  # Unprocessable entity due to validation errors

# Test for edge case (very small loan amount)
def test_predict_edge_case():
    response = client.post("/predict", json=edge_case_input_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probability" in response.json()
    assert isinstance(response.json()['prediction'], int)
    assert isinstance(response.json()['probability'], float)

# Test for error handling (simulate failure in model prediction)
def test_predict_model_failure(monkeypatch):
    # Simulate an exception during prediction
    def mock_predict(*args, **kwargs):
        raise Exception("Mock model failure")
    pipeline = joblib.load(r'pipelines/final_model_pipeline.pkl') 
    monkeypatch.setattr(pipeline, "predict", mock_predict)
    
    response = client.post("/predict", json=invalid_input_data)
    assert response.status_code == 422
    assert "detail" in response.json()
