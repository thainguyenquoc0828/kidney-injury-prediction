import os
import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.api as sm
import shap
import joblib

st.set_page_config(page_title="AKI Risk Prediction", layout="wide")

st.title("AKI Risk Prediction")

FEATURES = [
    "admission_age",
    "weight",
    "resp_rate_max",
    "temperature_max",
    "aniongap_max",
    "urineoutput_12h_sum",
    "gcs_unable_max",
    "infection_suspected_flag",
]

CONTINUOUS_FEATURES = [
    "admission_age",
    "weight",
    "resp_rate_max",
    "temperature_max",
    "aniongap_max",
    "urineoutput_12h_sum",
]

BINARY_FEATURES = ["gcs_unable_max", "infection_suspected_flag"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


@st.cache_resource
def get_model_artifact():
    # Kiểm tra sự tồn tại của các file model artifact trước khi tải
    required_files = ["medians.pkl", "scaler.pkl", "logit_model.pkl", "skl_model.pkl", "synthetic_background.pkl"]
    for file_name in required_files:
        file_path = os.path.join(MODELS_DIR, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Không tìm thấy file artifact: {file_path}. "
                f"Vui lòng chạy file 'train_local.py' dưới máy của bạn trước để tạo thư mục 'models/'."
            )

    logit_data = joblib.load(os.path.join(MODELS_DIR, "logit_model.pkl"))
    
    return {
        "medians": joblib.load(os.path.join(MODELS_DIR, "medians.pkl")),
        "scaler": joblib.load(os.path.join(MODELS_DIR, "scaler.pkl")),
        "logit_params": logit_data["params"],
        "logit_cov": logit_data["cov"],
        "skl_model": joblib.load(os.path.join(MODELS_DIR, "skl_model.pkl")),
        "background": joblib.load(os.path.join(MODELS_DIR, "synthetic_background.pkl")),
    }


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def preprocess_inputs(raw_values, model_artifact):
    row = pd.DataFrame([raw_values], columns=FEATURES)

    for col in CONTINUOUS_FEATURES:
        row[col] = pd.to_numeric(row[col], errors="coerce")
        row[col] = row[col].fillna(model_artifact["medians"][col])

    for col in BINARY_FEATURES:
        row[col] = pd.to_numeric(row[col], errors="coerce").fillna(0)

    row[CONTINUOUS_FEATURES] = model_artifact["scaler"].transform(row[CONTINUOUS_FEATURES])
    row[BINARY_FEATURES] = row[BINARY_FEATURES].astype(float)
    return row


def make_prediction(raw_values, model_artifact):
    X_input = preprocess_inputs(raw_values, model_artifact)
    X_design = sm.add_constant(X_input, has_constant="add")

    params = model_artifact["logit_params"].values
    cov = model_artifact["logit_cov"].values

    x_matrix = np.asarray(X_design, dtype=float)
    eta = x_matrix @ params
    var_eta = np.einsum("ij,jk,ik->i", x_matrix, cov, x_matrix)
    se_eta = np.sqrt(np.maximum(var_eta, 0))

    prob = sigmoid(float(eta[0]))
    ci_low = sigmoid(float(eta[0] - 1.96 * se_eta[0]))
    ci_high = sigmoid(float(eta[0] + 1.96 * se_eta[0]))

    return prob, ci_low, ci_high, X_input


def get_local_shap(X_input, model_artifact):
    background = model_artifact["background"]
    explainer = shap.Explainer(model_artifact["skl_model"], background)
    shap_values = explainer(X_input)

    values = np.array(shap_values.values).reshape(-1)
    if len(values) != len(FEATURES):
        values = np.zeros(len(FEATURES), dtype=float)

    shap_df = pd.DataFrame({
        "feature": FEATURES,
        "shap_value": values,
    }).sort_values("shap_value", key=lambda s: s.abs(), ascending=False)
    return shap_df


# Tải các mô hình và tham số đã huấn luyện từ trước
model_artifact = get_model_artifact()

with st.sidebar:
    st.header("Thông tin bệnh nhân")
    raw_values = {}

    raw_values["admission_age"] = st.number_input("admission_age", min_value=0.0, value=71.0, step=1.0)
    raw_values["weight"] = st.number_input("weight", min_value=0.0, value=74.0, step=1.0)
    raw_values["resp_rate_max"] = st.number_input("resp_rate_max", min_value=0.0, value=22.0, step=1.0)
    raw_values["temperature_max"] = st.number_input("temperature_max", min_value=0.0, value=37.0, step=0.1)
    raw_values["aniongap_max"] = st.number_input("aniongap_max", min_value=0.0, value=12.0, step=0.1)
    raw_values["urineoutput_12h_sum"] = st.number_input("urineoutput_12h_sum", min_value=0.0, value=1800.0, step=50.0)
    raw_values["gcs_unable_max"] = st.selectbox("gcs_unable_max", [0, 1], index=1)
    raw_values["infection_suspected_flag"] = st.selectbox("infection_suspected_flag", [0, 1], index=1)

    predict_button = st.button("Dự đoán", type="primary")

if predict_button:
    prob, ci_low, ci_high, X_input = make_prediction(raw_values, model_artifact)
    shap_df = get_local_shap(X_input, model_artifact)

    st.subheader("Kết quả dự đoán")
    col1, col2, col3 = st.columns(3)
    col1.metric("Xác suất bị AKI", f"{prob * 100:.2f}%")
    col2.metric("Lower 95% CI", f"{ci_low * 100:.2f}%")
    col3.metric("Upper 95% CI", f"{ci_high * 100:.2f}%")

    if prob >= 0.5:
        st.success("Nguy cơ cao")
    else:
        st.info("Nguy cơ thấp")

    st.subheader("Phương trình Logistic")
    params = model_artifact["logit_params"]
    intercept = params["const"]
    eq_parts = [f"{intercept:.4f}"]
    for feat in FEATURES:
        coef = params[feat]
        eq_parts.append(f"{coef:.4f}*{feat}")

    st.code("logit(p) = " + " + ".join(eq_parts), language="text")

    st.subheader("Local SHAP cho bệnh nhân")
    st.bar_chart(shap_df.set_index("feature")["shap_value"])

    st.dataframe(
        shap_df.assign(shap_value=lambda d: d["shap_value"].round(4)),
        use_container_width=True,
    )
else:
    st.info("Nhấn nút Dự đoán để xem kết quả.")