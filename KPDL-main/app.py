import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

st.title("Dự đoán bệnh tim")

# Load model
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'Model')
model = joblib.load(os.path.join(MODEL_DIR, 'stacking_model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
selected_features = joblib.load(os.path.join(MODEL_DIR, 'selected_features.pkl'))

# Input
age = st.number_input("Age", 0, 120, 25, step=1, format="%.0f")
gender = st.number_input("Gender", 0, 1, 0, step=1, format="%.0f")
bp = st.number_input("BloodPressure", value=120, step=1, format="%.0f")
chol = st.number_input("Cholesterol", value=200, step=1, format="%.0f")
hr = st.number_input("HeartRate", value=70, step=1, format="%.0f")
qpf = st.number_input("QuantumPatternFeature", value=1, step=1, format="%.0f")

if st.button("Dự đoán"):
    df = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'BloodPressure': bp,
        'Cholesterol': chol,
        'HeartRate': hr,
        'QuantumPatternFeature': qpf
    }])

    # Feature engineering
    df['BP_Cholesterol'] = df['BloodPressure'] * df['Cholesterol']
    df['Age_BP'] = df['Age'] * df['BloodPressure']

    X = df[selected_features]
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]

    labels = {0: 'Không bệnh', 1: 'Nhẹ', 2: 'Trung bình', 3: 'Nặng'}

    st.success(f"Kết quả: {labels[pred]}")
    st.write("Xác suất:", proba)