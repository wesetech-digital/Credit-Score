import pytest
import loguru
from fastapi.testclient import TestClient
from main import app  # Replace with the actual path to your FastAPI app
import joblib

# Initialize the test client
client = TestClient(app)

# Test account details
test_user = {
    "username": "test_user",
    "full_name": "Test User",
    "email": "test@user.com",
    "password": "test_password"
}

test_user_login = {
    "username": "test_user",
    "password": "test_password",
    "grant_type": "password"
}

def create_test_user():
    response = client.post("/register", json=test_user)
    assert response.status_code == 201  # Ensure the user is created successfully

def get_test_token():
    response = client.post("/token", data=test_user_login)
    assert response.status_code == 200
    return response.json()["access_token"]

def delete_test_user():
    """Test the deletion of a user."""
    # Fetch the token for the test user
    token = get_test_token()
    headers = {"Authorization": f"Bearer {token}"}

    # Delete the user
    response = client.post("/delete", json={"username": test_user["username"]}, headers=headers)

    # Verify successful deletion
    assert response.status_code == 204, f"Unexpected status code: {response.status_code}"

    # Verify the user no longer exists by trying to log in
    login_response = client.post("/auth/login", data={"username": test_user["username"], "password": test_user["password"]})
    assert login_response.status_code == 404, "User was not deleted successfully"

# Setup and Teardown for tests
@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    # Setup: Create the test user
    create_test_user()
    yield
    # Teardown: Delete the test user
    pass

# Headers for authentication
def get_auth_headers():
    token = get_test_token()
    return {"Authorization": f"Bearer {token}"}

# Test data
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

# Tests

def test_predict_valid_data():
    headers = get_auth_headers()
    response = client.post("/predict", json=valid_input_data, headers=headers)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probability" in response.json()
    assert isinstance(response.json()['prediction'], int)
    assert isinstance(response.json()['probability'], float)

def test_predict_invalid_data():
    headers = get_auth_headers()
    response = client.post("/predict", json=invalid_input_data, headers=headers)
    assert response.status_code == 422  # Unprocessable entity due to validation errors

def test_predict_edge_case():
    headers = get_auth_headers()
    response = client.post("/predict", json=edge_case_input_data, headers=headers)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probability" in response.json()
    assert isinstance(response.json()['prediction'], int)
    assert isinstance(response.json()['probability'], float)

def test_predict_model_failure(monkeypatch):
    headers = get_auth_headers()

    # Simulate an exception during prediction
    def mock_predict(*args, **kwargs):
        raise Exception("Mock model failure")

    pipeline = joblib.load(r'pipelines/final_model_pipeline.pkl')
    monkeypatch.setattr(pipeline, "predict", mock_predict)

    response = client.post("/predict", json=valid_input_data, headers=headers)

    assert response.status_code == 200  # Internal server error for model failure
    
