from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import joblib
import pandas as pd

# Load artifacts once at startup
model = joblib.load('model_artifacts/random_forest.pkl')
scaler = joblib.load('model_artifacts/scaler.pkl')
label_encoders = joblib.load('model_artifacts/label_encoders.pkl')
columns_info = joblib.load('model_artifacts/columns.pkl')

numeric_cols = columns_info['numeric_cols']
train_columns = columns_info['all_columns']

app = FastAPI(title="Churn Prediction API")

# Define input data schema with validation
class CustomerData(BaseModel):
    age: int = Field(..., ge=0, le=120)
    watch_hours: float = Field(..., ge=0)
    last_login_days: int = Field(..., ge=0)
    monthly_fee: float = Field(..., ge=0)
    number_of_profiles: int = Field(..., ge=1)
    avg_watch_time_per_day: float = Field(..., ge=0)

    gender: str
    subscription_type: str
    region: str
    device: str
    payment_method: str
    favorite_genre: str

    @field_validator('gender')
    def gender_must_be_known(cls, v):
        if v not in ['Male', 'Female']:
            raise ValueError('gender must be Male or Female')
        return v

    # Add other validators here using @field_validator if needed


def preprocess_input(df):
    df = df.copy()
    for col, encoder in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col])
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Unknown category in {col}: {e}")

    df = pd.get_dummies(df, drop_first=True)
    df = df.reindex(columns=train_columns, fill_value=0)
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df
@app.get("/")
def home():
    return {"message": "Welcome to the Churn Prediction API. Visit /docs to test it."}

@app.post("/predict")
def predict(customer: CustomerData):
    input_df = pd.DataFrame([customer.dict()])
    processed = preprocess_input(input_df)
    prediction = model.predict(processed)[0]
    probability = model.predict_proba(processed)[0][1]

    return {
        "prediction": int(prediction),
        "label": "Churn" if prediction == 1 else "No Churn",
        "probability": round(float(probability), 4)
    }
