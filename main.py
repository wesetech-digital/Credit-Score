from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd

# Secret key and algorithm for JWT
SECRET_KEY = "odhoudoosgoduhspivpiduodosvdonsdopi"  # Replace with a secure key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def probability_to_score(probability: float) -> int:
    """
    Transforms a probability (0 to 1) into a score between 0 and 999.
    A probability of 0 gives a score of 999, and a probability of 1 gives a score of 0.
    
    Args:
        probability (float): Probability of default (0 <= probability <= 1)
    
    Returns:
        int: Corresponding score (0 to 999)
    """
    if not (0 <= probability <= 1):
        raise ValueError("Probability must be between 0 and 1.")
    return int(999 - (probability * 999))

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# In-memory database (replace with a real database in production)
fake_users_db = {}
stored_predictions = []  # List to store input and prediction results

# FastAPI instance
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)

# Models
class User(BaseModel):
    username: str
    full_name: str
    email: EmailStr
    disabled: bool = False

class UserInDB(User):
    hashed_password: str

class UserCreate(BaseModel):
    username: str
    full_name: str
    email: EmailStr
    password: str

class InputData(BaseModel):
    loan_type: str
    Total_Amount: float
    Total_Amount_to_Repay: float
    disbursement_date: str
    due_date: str
    duration: int
    New_versus_Repeat: str
    Amount_Funded_By_Lender: float
    Lender_portion_Funded: float
    Lender_portion_to_be_repaid: float

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    score: int
    model_version: str
    input_data: dict


class DeleteUserRequest(BaseModel):
    username: str


# Utility functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db, username: str):
    user = db.get(username)
    if user:
        return UserInDB(**user)
    return None

def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.now() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username)
    if user is None:
        raise credentials_exception
    return user

# Endpoints
@app.post("/register", status_code=status.HTTP_201_CREATED)
def register_user(user: UserCreate):
    if user.username in fake_users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists",
        )
    hashed_password = get_password_hash(user.password)
    fake_users_db[user.username] = {
        "username": user.username,
        "full_name": user.full_name,
        "email": user.email,
        "hashed_password": hashed_password,
        "disabled": False,
    }
    return {"message": f"User {user.username} registered successfully"}

@app.post("/delete", status_code=204)
def delete_user(
    request: DeleteUserRequest,
    current_user: User = Depends(get_current_user),  # For authentication and authorization
):

    # Fetch the user to be deleted
    user_to_delete = fake_users_db.get(request.username)


    if not user_to_delete:
        raise HTTPException(status_code=404, detail="User not found")

    # # Prevent the current user from deleting themselves
    # if user_to_delete.get('username') == current_user.username:
    #     raise HTTPException(status_code=400, detail="Cannot delete your own account")

    # Delete the user
    fake_users_db.delete(user_to_delete)
    fake_users_db.commit()

    return {"detail": f"User {request.username} successfully deleted"}

@app.post("/token")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=User)
def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.post("/predict", response_model=PredictionResponse)
def predict(data: InputData, current_user: User = Depends(get_current_user)):
    logger.info(f"Prediction requested by user: {current_user.username}")

    # Dummy prediction logic
    prediction = 1 if data.Total_Amount > 5000 else 0
    probability = 0.85 if prediction == 1 else 0.15
    model_version = "v1.0.0"

    # Store the prediction and input data
    stored_predictions.append({
        "username": current_user.username,
        "input_data": data.model_dump(),
        "prediction": prediction,
        "probability": probability,
    })

        # Calculate score from probability
    score = probability_to_score(probability)

    return PredictionResponse(
        prediction=prediction,
        score=score,
        probability=probability,
        model_version=model_version,
        input_data=data.model_dump(),
    )

@app.get("/predictions")
def get_user_predictions(current_user: User = Depends(get_current_user)):
    user_predictions = [
        record for record in stored_predictions if record["username"] == current_user.username
    ]
    return {"predictions": user_predictions}

