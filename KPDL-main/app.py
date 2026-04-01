import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ====== CONFIG ======
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>❤️ Dự đoán bệnh tim</h1>", unsafe_allow_html=True)
st.markdown("---")

# Load model
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'Model')
model = joblib.load(os.path.join(MODEL_DIR, 'stacking_model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
selected_features = joblib.load(os.path.join(MODEL_DIR, 'selected_features.pkl'))

# ====== INPUT UI ======
st.subheader("📋 Nhập thông tin bệnh nhân")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🎂 Tuổi", min_value=0, max_value=120, step=1)
    gender = st.selectbox("🚻 Giới tính", ["Nữ", "Nam"])
    bp = st.number_input("🩸 Huyết áp", step=1)

with col2:
    chol = st.number_input("🧪 Cholesterol", step=1)
    hr = st.number_input("💓 Nhịp tim", step=1)
    qpf = st.slider("⚛️ Quantum Feature", 0.0, 1.0, 0.5)

# convert gender
gender = 1 if gender == "Nam" else 0

st.markdown("---")

# ====== BUTTON ======
if st.button("🔍 Dự đoán", use_container_width=True):

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

    # ====== RESULT UI ======
    st.markdown("## 📊 Kết quả")

    if pred == 0:
        st.success(f"✅ {labels[pred]}")
    elif pred == 1:
        st.info(f"⚠️ {labels[pred]}")
    elif pred == 2:
        st.warning(f"⚠️ {labels[pred]}")
    else:
        st.error(f"🚨 {labels[pred]}")

    # Hiển thị xác suất đẹp hơn
    st.markdown("### 📈 Xác suất")
    st.progress(float(np.max(proba)))

    st.write({
        "Không bệnh": round(proba[0], 3),
        "Nhẹ": round(proba[1], 3),
        "Trung bình": round(proba[2], 3),
        "Nặng": round(proba[3], 3)
    })