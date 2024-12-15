# FastAPI Loan Default Prediction & Credit Scoring Service

This FastAPI service allows users to submit loan data for prediction on whether a loan will default, using a pre-trained machine learning model pipeline. The service is built with **FastAPI**, **scikit-learn**, and **SHAP** for model explainability.

## Table of Contents
1. [Setup Instructions](#setup-instructions)
2. [Starting the Server](#starting-the-server)
3. [Hitting the Endpoint](#hitting-the-endpoint)
4. [Input Payload Schema](#input-payload-schema)
5. [Features Explained](#features-explained)
6. [Expected Output](#expected-output)
7. [Error Handling](#error-handling)

## Setup Instructions

### Prerequisites

Ensure that the following are installed on your machine:
1. Python 3.12+ (It is recommended to use a virtual environment for isolation)
2. The required Python packages listed in the `requirements.txt` file

To install the necessary packages, run:

```bash
pip install -r requirements.txt
```

### Load the Model Pipeline

Ensure that you have the pre-trained model pipeline saved as `final_model_pipeline.pkl` in the `pipelines/` directory. The pipeline file contains both the preprocessing steps and the trained model.

If you don’t have this file, you will need to train and save the pipeline using the following code:

```python
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

# Define and train your model (this is an example, adapt to your needs)
pipeline = Pipeline([
    # Add your preprocessing and model steps here
])

# Save the pipeline
joblib.dump(pipeline, 'pipelines/final_model_pipeline.pkl')
```

### File Structure

```
.
├── pipelines/
│   └── final_model_pipeline.pkl
├── scripts/
│   └──transformers.py
└── main.py      # The FastAPI app
```

## Starting the Server

1. Navigate to the directory containing `main.py`.
2. Run the FastAPI application with Uvicorn using the following command:

```bash
uvicorn main:app --reload
```

This will start a development server on `http://127.0.0.1:8000`. The `--reload` flag ensures the server automatically reloads when you make changes to the code.

## Hitting the Endpoint

Once the server is running, you can access the prediction endpoint at:

```
POST http://127.0.0.1:8000/predict
```

Additionally, FastAPI automatically generates interactive documentation for the API. You can view the documentation by visiting the following URL in your browser:

```
http://127.0.0.1:8000/docs
```

This will provide you with an interactive interface to test the endpoint directly from your browser.

Use **Postman** or **cURL** to test the endpoint.

### Example cURL Request

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
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

### Example Postman Request

1. Select `POST` method.
2. Set URL to `http://127.0.0.1:8000/predict`.
3. In the **Body** tab, select `raw` and set the type to `JSON`.
4. Paste the example JSON request body.

## Input Payload Schema

The API expects the following JSON structure in the request body:

```json
{
  "loan_type": "string",  // Type of the loan (e.g., 'personal', 'business')
  "Total_Amount": "float",  // Total loan amount
  "Total_Amount_to_Repay": "float",  // Total amount to be repaid
  "disbursement_date": "string",  // Date of loan disbursement (in YYYY-MM-DD format)
  "due_date": "string",  // Date when repayment is due (in YYYY-MM-DD format)
  "duration": "integer",  // Duration of the loan (in months)
  "New_versus_Repeat": "string",  // Whether the borrower is new or repeat (e.g., 'New', 'Repeat')
  "Amount_Funded_By_Lender": "float",  // Amount funded by the lender
  "Lender_portion_Funded": "float",  // Portion of the loan funded by the lender
  "Lender_portion_to_be_repaid": "float"  // Portion of the loan to be repaid by the borrower
}
```

## Features Explained

- **loan_type**: Describes the type of loan (e.g., "personal", "business"). This may influence the prediction as different loan types may have different risk profiles.
  
- **Total_Amount**: The total amount of the loan that the borrower has received.

- **Total_Amount_to_Repay**: The total amount the borrower needs to repay, including interest and fees.

- **disbursement_date**: The date when the loan amount was disbursed to the borrower.

- **due_date**: The date by which the loan repayment is due.

- **duration**: The duration of the loan in months, indicating how long the borrower has to repay the loan.

- **New_versus_Repeat**: This feature indicates whether the borrower is a first-time borrower ("New") or a returning borrower ("Repeat").

- **Amount_Funded_By_Lender**: The amount of the loan funded by the lender (could differ from the total loan amount).

- **Lender_portion_Funded**: The portion of the loan funded by the lender.

- **Lender_portion_to_be_repaid**: The portion of the loan that needs to be repaid by the borrower to the lender.

## Expected Output

The model will return a JSON response containing the following information:

```json
{
  "prediction": 1,  // 1 = Default, 0 = No Default
  "probability": 0.85,  // Probability of the prediction being correct
  "confidence": 0.15,  // Confidence, the difference between top two probabilities
  "feature_contributions": {
    "Total_Amount": 0.12,
    "Total_Amount_to_Repay": 0.07,
    "disbursement_date": -0.05,
    "due_date": 0.03,
    "duration": -0.02,
    "New_versus_Repeat": 0.1,
    "Amount_Funded_By_Lender": 0.15,
    "Lender_portion_Funded": -0.04,
    "Lender_portion_to_be_repaid": 0.04
  },
  "model_version": "v1.0.0"
}
```

### Explanation of the Response

- **prediction**: The model's predicted class (1 = Default, 0 = No Default).
- **probability**: The highest probability of the prediction.
- **confidence**: The confidence of the prediction, calculated as the difference between the top two predicted probabilities.
- **feature_contributions**: A dictionary with each feature's contribution to the model's decision (feature importance). Higher values indicate features that had a stronger impact on the prediction.
- **model_version**: The version of the model used for making the prediction.

## Error Handling

If an error occurs during prediction, the server will return an HTTP 500 status with an error message, which may include issues such as invalid input format or a problem within the model pipeline.

```json
{
  "detail": "Error occurred: <error_message>"
}
```

---

This README provides a comprehensive guide to setting up and using the FastAPI Loan Default Prediction & Credit Scoring Service.