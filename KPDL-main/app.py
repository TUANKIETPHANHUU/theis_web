import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

st.title("Dự đoán bệnh tim")

# Load model
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
model = joblib.load(os.path.join(MODEL_DIR, 'stacking_model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
selected_features = joblib.load(os.path.join(MODEL_DIR, 'selected_features.pkl'))

# Input
age = st.number_input("Age")
gender = st.number_input("Gender")
bp = st.number_input("BloodPressure")
chol = st.number_input("Cholesterol")
hr = st.number_input("HeartRate")
qpf = st.number_input("QuantumPatternFeature")

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