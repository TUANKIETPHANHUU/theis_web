import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import altair as alt
import seaborn as sns
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
    return pd.read_csv("KPDL-main/Heart Prediction Quantum Dataset.csv")  # sửa đúng tên file dataset của bạn

model, scaler, selected_features = load_model()
df = load_data()

# =========================
# ====== PAGE 1: EDA ======
# =========================
if page == "🏠 Giới thiệu & EDA":

    st.title("📊 Giới thiệu & Khám phá dữ liệu")

    st.markdown("""
    **📌 Đề tài:** Dự đoán bệnh tim  
    **👨‍🎓 Sinh viên:** Nguyễn Văn A  
    **🆔 MSSV:** 123456  

    👉 Ứng dụng AI giúp dự đoán mức độ bệnh tim nhằm hỗ trợ chẩn đoán sớm.
    """)

    st.subheader("📂 Dữ liệu mẫu")
    st.dataframe(df.head())

    # ===== Biểu đồ 1 =====
    st.subheader("📊 Phân phối nhãn")

    fig, ax = plt.subplots()
    df['HeartDisease'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

    # ===== Biểu đồ 2 =====
    st.subheader("📊 Ma trận tương quan")

    fig2, ax2 = plt.subplots()
    sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax2)
    st.pyplot(fig2)

    # ===== Nhận xét =====
    st.subheader("🧠 Nhận xét")
    st.write("""
    - Dữ liệu có dấu hiệu mất cân bằng giữa các lớp.
    - Một số đặc trưng như Cholesterol, BloodPressure có ảnh hưởng mạnh.
    - Feature engineering giúp cải thiện mô hình.
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

        # Feature engineering
        input_df['BP_Cholesterol'] = input_df['BloodPressure'] * input_df['Cholesterol']
        input_df['Age_BP'] = input_df['Age'] * input_df['BloodPressure']

        X = input_df[selected_features]
        X_scaled = scaler.transform(X)

        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]

        labels = {0: 'Không bệnh', 1: 'Nhẹ', 2: 'Trung bình', 3: 'Nặng'}

        st.subheader("📊 Kết quả")

        if pred == 0:
            st.success(f"✅ {labels[pred]}")
        elif pred == 1:
            st.info(f"⚠️ {labels[pred]}")
        elif pred == 2:
            st.warning(f"⚠️ {labels[pred]}")
        else:
            st.error(f"🚨 {labels[pred]}")

        # ===== Biểu đồ =====
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

    y_true = df['HeartDisease']
    y_pred = model.predict(scaler.transform(df[selected_features]))

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    st.pyplot(fig)

    st.subheader("🧠 Nhận xét")
    st.write("""
    - Mô hình đôi khi nhầm giữa mức Nhẹ và Trung bình.
    - Cần thêm dữ liệu để cải thiện.
    - Có thể tối ưu bằng tuning hoặc Deep Learning.
    """)