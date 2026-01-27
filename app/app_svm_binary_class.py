import streamlit as st
import numpy as np
import joblib

# streamlit run app_svm_binary_class.py

# Load model & scaler

@st.cache_resource
def load_model():
    model = joblib.load("../model/svm_binary_class_model.joblib")
    scaler = joblib.load("../data/binary_class/scaler.pkl")  # chỉ scale age
    return model, scaler

model, scaler = load_model()


# CATEGORY LIST (THEO X_TRAIN)

DX_TYPE_CATS = ["confocal", "consensus", "follow_up", "histo"]
SEX_CATS = ["female", "male", "unknown"]

LOCALIZATION_CATS = [
    "abdomen", "acral", "back", "chest", "ear", "face", "foot",
    "genital", "hand", "lower extremity", "neck",
    "scalp", "trunk", "upper extremity", "unknown"
]

DATASET_CATS = [
    "rosendahl", "vidir_molemax", "vidir_modern", "vienna_dias"
]


# UI

st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Skin Cancer Detection (Binary Classification)")
st.markdown("Dữ liệu **HAM10000 metadata**")

st.divider()


# INPUT

age = st.slider("Tuổi", 0, 100, 50)

dx_type = st.selectbox(
    "Phương pháp chẩn đoán",
    DX_TYPE_CATS
)

sex = st.selectbox(
    "Giới tính",
    SEX_CATS
)

localization = st.selectbox(
    "Vị trí tổn thương",
    LOCALIZATION_CATS
)

dataset = st.selectbox(
    "Nguồn dữ liệu",
    DATASET_CATS
)


# ONE-HOT FUNCTION
def one_hot(value, categories):
    return [1 if value == c else 0 for c in categories]


# BUILD FEATURE VECTOR 
age_scaled = scaler.transform([[age]])  # (1,1)

dx_type_ohe = one_hot(dx_type, DX_TYPE_CATS)
sex_ohe = one_hot(sex, SEX_CATS)
loc_ohe = one_hot(localization, LOCALIZATION_CATS)
dataset_ohe = one_hot(dataset, DATASET_CATS)

X_input = np.array([[
    age_scaled[0][0],
    *dx_type_ohe,
    *sex_ohe,
    *loc_ohe,
    *dataset_ohe
]])


# PREDICT
st.divider()
if st.button("🔍 Dự đoán", use_container_width=True):

    # predict_proba: [P(benign), P(malignant)]
    prob_malignant = model.predict_proba(X_input)[0][1]

    threshold = 0.5  # ngưỡng chuẩn binary

    if prob_malignant >= threshold:
        st.error(
            f"⚠️ **NGHI NGỜ ÁC TÍNH**\n\n"
            f"Xác suất ác tính: **{prob_malignant:.2%}**"
        )
    else:
        st.success(
            f"✅ **LÀNH TÍNH**\n\n"
            f"Xác suất ác tính: **{prob_malignant:.2%}**"
        )

    st.caption("Ứng dụng hỗ trợ quyết định, không thay thế bác sĩ.")

st.divider()
st.caption("📌 SVM | Manual One-Hot | EXACT FEATURE MATCH")


# streamlit run app_svm_binary_class.py
