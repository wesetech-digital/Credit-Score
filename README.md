

# FastAPI Loan Default Prediction & Credit Scoring Service

This FastAPI service allows users to submit loan data and get a prediction on whether a loan will default or not. The service leverages a pre-trained machine learning model to perform loan default prediction and provides users with detailed insights on the loan's risk.

The service includes authentication via JWT tokens, user management, and storage for past predictions.

---

## Table of Contents

1. [Setup Instructions](#setup-instructions)
2. [Starting the Server](#starting-the-server)
3. [Authentication & Authorization](#authentication-authorization)
4. [API Endpoints](#api-endpoints)
5. [Input Payload Schema](#input-payload-schema)
6. [Prediction Model](#prediction-model)
7. [Features Explained](#features-explained)
8. [Error Handling](#error-handling)
9. [Example Requests](#example-requests)

---

## Setup Instructions

### Prerequisites

Before setting up the project, make sure you have the following installed:

1. **Python 3.12+** (It's recommended to use a virtual environment)
2. **Required Python packages** as listed in the `requirements.txt`

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

---

### Load the Model Pipeline

If you're using a pre-trained model, ensure that the model is saved as `final_model_pipeline.pkl` and placed in the `pipelines/` directory. This model is responsible for the loan default predictions.

If the model is not available, you will need to train and save the model using the following example code:

```python
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

# Define and train your model (example code, adjust as needed)
pipeline = Pipeline([
    # Add preprocessing steps and your trained model here
])

# Save the trained pipeline
joblib.dump(pipeline, 'pipelines/final_model_pipeline.pkl')
```

---

### File Structure

```
.
├── pipelines/
│   └── final_model_pipeline.pkl  # Pre-trained model pipeline
├── scripts/
│   └── transformers.py  # Additional processing scripts if needed
└── main.py  # FastAPI application
```

---

## Starting the Server

To start the FastAPI server:

1. Navigate to the directory containing `main.py`.
2. Run the FastAPI application with Uvicorn:

```bash
uvicorn main:app --reload
```

This starts a development server at `http://127.0.0.1:8000` and automatically reloads when you make code changes (due to the `--reload` flag).

---

## Authentication & Authorization

The FastAPI application uses OAuth2 with Password and Bearer tokens for user authentication.

- **User Registration**: Users can register by providing a username, email, full name, and password.
- **Login**: After registering, users can log in to receive a JWT access token, which they can use for authenticated requests.
- **Token Expiry**: Tokens expire after 30 minutes.

### Login Endpoint

`POST /token` - Logs in a user and returns an access token.

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/token' \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'username=testuser&password=testpassword'
```

### Protected Routes

Once authenticated, users can access protected routes by passing the token in the `Authorization` header:

```
Authorization: Bearer <access_token>
```

---

## API Endpoints

### Register User

`POST /register` - Register a new user with username, email, full name, and password.

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/register' \
  -H 'Content-Type: application/json' \
  -d '{
    "username": "newuser",
    "full_name": "New User",
    "email": "newuser@example.com",
    "password": "securepassword"
  }'
```

### Get Current User

`GET /users/me` - Returns the details of the currently authenticated user.

```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/users/me' \
  -H 'Authorization: Bearer <access_token>'
```

### Loan Default Prediction

`POST /predict` - Predicts loan default based on input data. Requires authentication.

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <access_token>' \
  -d '{
    "loan_type": "personal",
    "Total_Amount": 5000,
    "Total_Amount_to_Repay": 6000,
    "disbursement_date": "2023-01-01",
    "due_date": "2024-01-01",
    "duration": 12,
    "New_versus_Repeat": "New",
    "Amount_Funded_By_Lender": 4500,
    "Lender_portion_Funded": 4000,
    "Lender_portion_to_be_repaid": 4500
  }'
```

### Get User Predictions

`GET /predictions` - Returns a list of predictions made by the currently authenticated user.

```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/predictions' \
  -H 'Authorization: Bearer <access_token>'
```

---

## Input Payload Schema

The `/predict` endpoint expects the following JSON structure:

```json
{
  "loan_type": "string",  // Type of the loan (e.g., 'personal', 'business')
  "Total_Amount": "float",  // Total loan amount
  "Total_Amount_to_Repay": "float",  // Total amount to be repaid
  "disbursement_date": "string",  // Date of loan disbursement (in YYYY-MM-DD format)
  "due_date": "string",  // Date when repayment is due (in YYYY-MM-DD format)
  "duration": "integer",  // Duration of the loan (in months)
  "New_versus_Repeat": "string",  // Whether the borrower is new or repeat
  "Amount_Funded_By_Lender": "float",  // Amount funded by the lender
  "Lender_portion_Funded": "float",  // Portion of the loan funded by the lender
  "Lender_portion_to_be_repaid": "float"  // Portion of the loan to be repaid by the borrower
}
```

---

## Prediction Model

The prediction is based on a pre-trained model pipeline (saved as `final_model_pipeline.pkl`). It uses machine learning techniques (such as isolation forests, ensemble methods, or other models you implement) to predict whether a loan will default based on input features.

The model returns the following prediction output:

```json
{
  "prediction": 1,  // 1 = Default, 0 = No Default
  "probability": 0.85,  // Probability of the prediction being correct
  "model_version": "v1.0.0"
}
```

---

## Features Explained

- **loan_type**: Type of loan (e.g., "personal", "business").
- **Total_Amount**: Total loan amount disbursed to the borrower.
- **Total_Amount_to_Repay**: Total amount to be repaid, including interest.
- **disbursement_date**: Date when the loan was issued.
- **due_date**: Date the loan is due.
- **duration**: Loan repayment duration in months.
- **New_versus_Repeat**: Borrower's history (New or Repeat).
- **Amount_Funded_By_Lender**: Amount of the loan funded by the lender.
- **Lender_portion_Funded**: Lender's portion of the loan.
- **Lender_portion_to_be_repaid**: Portion of the loan that must be repaid by the borrower.

---

## Error Handling

If an error occurs during any prediction, a 500 HTTP error is returned with a detailed message.

Example error message:

```json
{
  "detail": "Error occurred: <error_message>"
}
```
