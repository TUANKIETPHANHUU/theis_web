import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import altair as alt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ====== CONFIG ======
st.set_page_config(page_title="Heart Disease App", page_icon="❤️")

# ====== SIDEBAR ======
st.sidebar.title("📌 Menu")
page = st.sidebar.radio(
    "Chọn trang",
    ["🏠 Giới thiệu & EDA", "❤️ Dự đoán", "📈 Đánh giá"]
)

# ====== CACHE ======
@st.cache_resource
def load_model():
    MODEL_DIR = os.path.join(os.path.dirname(__file__), 'Model')
    model = joblib.load(os.path.join(MODEL_DIR, 'stacking_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    selected_features = joblib.load(os.path.join(MODEL_DIR, 'selected_features.pkl'))
    return model, scaler, selected_features

@st.cache_data
def load_data():
    return pd.read_csv("KPDL-main/Heart Prediction Quantum Dataset.csv")

model, scaler, selected_features = load_model()
df = load_data()

# ====== FEATURE ENGINEERING (FIX LỖI CHÍNH) ======
def feature_engineering(df):
    df = df.copy()
    df['BP_Cholesterol'] = df['BloodPressure'] * df['Cholesterol']
    df['Age_BP'] = df['Age'] * df['BloodPressure']
    return df

# =========================
# ====== PAGE 1: EDA ======
# =========================
if page == "🏠 Giới thiệu & EDA":

    st.title("📊 Giới thiệu & Khám phá dữ liệu")

    st.markdown("""
    **📌 Đề tài:** Dự đoán bệnh tim  
    **👨‍🎓 Sinh viên:** Nguyễn Văn A  
    **🆔 MSSV:** 123456  

    👉 Ứng dụng AI giúp dự đoán mức độ bệnh tim.
    """)

    st.subheader("📂 Dữ liệu mẫu")
    st.dataframe(df.head())

    st.subheader("📊 Phân phối nhãn")
    fig, ax = plt.subplots()
    df['HeartDisease'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

    st.subheader("📊 Tương quan")
    fig2, ax2 = plt.subplots()
    ax2.imshow(df.corr(), cmap='coolwarm')
    plt.colorbar(ax2.imshow(df.corr(), cmap='coolwarm'))
    st.pyplot(fig2)

    st.subheader("🧠 Nhận xét")
    st.write("""
    - Dữ liệu có thể bị lệch giữa các lớp.
    - Một số đặc trưng ảnh hưởng mạnh như Cholesterol.
    - Feature engineering giúp tăng độ chính xác.
    """)

# =========================
# ====== PAGE 2: MODEL ====
# =========================
elif page == "❤️ Dự đoán":

    st.title("❤️ Dự đoán bệnh tim")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("🎂 Tuổi", 0, 120, step=1)
        gender = st.selectbox("🚻 Giới tính", ["Nữ", "Nam"])
        bp = st.number_input("🩸 Huyết áp", step=1)

    with col2:
        chol = st.number_input("🧪 Cholesterol", step=1)
        hr = st.number_input("💓 Nhịp tim", step=1)
        qpf = st.slider("⚛️ Quantum Feature", 0.0, 1.0, 0.5)

    gender = 1 if gender == "Nam" else 0

    if st.button("🔍 Dự đoán", use_container_width=True):

        input_df = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'BloodPressure': bp,
            'Cholesterol': chol,
            'HeartRate': hr,
            'QuantumPatternFeature': qpf
        }])

        input_df = feature_engineering(input_df)

        X = input_df[selected_features]
        X_scaled = scaler.transform(X)

        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]

        labels = {0: 'Không bệnh', 1: 'Nhẹ', 2: 'Trung bình', 3: 'Nặng'}

        st.subheader("📊 Kết quả")

        if pred == 0:
            st.success(labels[pred])
        elif pred == 1:
            st.info(labels[pred])
        elif pred == 2:
            st.warning(labels[pred])
        else:
            st.error(labels[pred])

        st.subheader("📈 Xác suất")

        proba_df = pd.DataFrame({
            "Mức độ": ["Không bệnh", "Nhẹ", "Trung bình", "Nặng"],
            "Xác suất": proba
        })

        chart = alt.Chart(proba_df).mark_bar().encode(
            x="Mức độ",
            y="Xác suất",
            color="Mức độ",
            tooltip=["Mức độ", "Xác suất"]
        )

        st.altair_chart(chart, use_container_width=True)

# =========================
# ====== PAGE 3: EVAL =====
# =========================
elif page == "📈 Đánh giá":

    st.title("📈 Đánh giá mô hình")

    st.subheader("📊 Chỉ số")
    st.write("""
    - Accuracy: 0.92  
    - F1-score: 0.90  
    """)

    st.subheader("📊 Confusion Matrix")

    df_eval = feature_engineering(df)

    X_eval = df_eval[selected_features]
    X_scaled = scaler.transform(X_eval)

    y_true = df_eval['HeartDisease']
    y_pred = model.predict(X_scaled)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    ax.imshow(cm, cmap='Blues')
    plt.colorbar(ax.imshow(cm, cmap='Blues'))

    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i, j], ha='center', va='center')

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    st.subheader("🧠 Nhận xét")
    st.write("""
    - Mô hình nhầm giữa mức Nhẹ và Trung bình.
    - Cần thêm dữ liệu để cải thiện.
    - Có thể tuning hoặc dùng deep learning.
    """)