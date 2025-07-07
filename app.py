import streamlit as st
import joblib
import pandas as pd

# Load artifacts once
model = joblib.load('model_artifacts/random_forest.pkl')
scaler = joblib.load('model_artifacts/scaler.pkl')
label_encoders = joblib.load('model_artifacts/label_encoders.pkl')
columns_info = joblib.load('model_artifacts/columns.pkl')

numeric_cols = columns_info['numeric_cols']
train_columns = columns_info['all_columns']

st.title("📊 Churn Prediction App")
st.markdown("Provide customer details to predict churn probability.")


def preprocess_input(df):
    df = df.copy()
    for col, encoder in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col])
            except ValueError as e:
                st.error(f"Unknown category in {col}: {e}")
                return None

    df = pd.get_dummies(df, drop_first=True)
    df = df.reindex(columns=train_columns, fill_value=0)
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df


with st.form("churn_form"):
    st.header("Enter Customer Information")
    
    # Numeric Inputs
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    watch_hours = st.number_input("Watch Hours", min_value=0.0, value=20.0)
    last_login_days = st.number_input("Days Since Last Login", min_value=0, value=5)
    monthly_fee = st.number_input("Monthly Fee", min_value=0.0, value=15.99)
    number_of_profiles = st.number_input("Number of Profiles", min_value=1, value=2)
    avg_watch_time_per_day = st.number_input("Average Watch Time per Day (hrs)", min_value=0.0, value=2.5)

    # Categorical Inputs
    gender = st.selectbox("Gender", ["Male", "Female"])
    subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    region = st.selectbox("Region", ["North", "South", "East", "West"])
    device = st.selectbox("Device", ["Mobile", "Tablet", "TV", "Laptop"])
    payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "PayPal", "UPI"])
    favorite_genre = st.selectbox("Favorite Genre", ["Drama", "Comedy", "Action", "Horror", "Sci-Fi", "Romance"])

    submitted = st.form_submit_button("Predict")

    if submitted:
        # Create input DataFrame
        input_data = pd.DataFrame([{
            "age": age,
            "watch_hours": watch_hours,
            "last_login_days": last_login_days,
            "monthly_fee": monthly_fee,
            "number_of_profiles": number_of_profiles,
            "avg_watch_time_per_day": avg_watch_time_per_day,
            "gender": gender,
            "subscription_type": subscription_type,
            "region": region,
            "device": device,
            "payment_method": payment_method,
            "favorite_genre": favorite_genre
        }])

        processed = preprocess_input(input_data)

        if processed is not None:
            prediction = model.predict(processed)[0]
            probability = model.predict_proba(processed)[0][1]

            st.success(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
            st.info(f"Probability of Churn: {round(float(probability), 4)}")
