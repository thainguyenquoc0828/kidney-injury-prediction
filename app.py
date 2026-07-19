# cd 'd:/2024/work/Project/AKI'; python -m streamlit run app.py --server.headless true
import os
import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.api as sm
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
DATA_PATH = os.path.join(BASE_DIR, "data", "AKI_patients_with_features_train.csv")


@st.cache_resource
def get_model_artifact():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Không tìm thấy file huấn luyện: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if "aki_label" not in df.columns:
        raise ValueError("File huấn luyện không có cột aki_label")

    y = df["aki_label"].astype(int)
    X = df[FEATURES].copy()

    X_proc = X.copy()
    for col in CONTINUOUS_FEATURES:
        X_proc[col] = pd.to_numeric(X_proc[col], errors="coerce")
    for col in BINARY_FEATURES:
        X_proc[col] = pd.to_numeric(X_proc[col], errors="coerce")

    medians = {col: float(X_proc[col].median()) for col in CONTINUOUS_FEATURES}
    for col in CONTINUOUS_FEATURES:
        X_proc[col] = X_proc[col].fillna(medians[col])
    for col in BINARY_FEATURES:
        X_proc[col] = X_proc[col].fillna(0)

    scaler = StandardScaler()
    X_proc[CONTINUOUS_FEATURES] = scaler.fit_transform(X_proc[CONTINUOUS_FEATURES])
    X_proc[BINARY_FEATURES] = X_proc[BINARY_FEATURES].astype(float)

    X_design = sm.add_constant(X_proc, has_constant="add")
    logit_model = sm.Logit(y, X_design).fit(disp=False)

    skl_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    skl_model.fit(X_proc, y)

    return {
        "medians": medians,
        "scaler": scaler,
        "logit_model": logit_model,
        "skl_model": skl_model,
        "train_X": X_proc,
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

    params = model_artifact["logit_model"].params.values
    cov = model_artifact["logit_model"].cov_params().values

    x_matrix = np.asarray(X_design, dtype=float)
    eta = x_matrix @ params
    var_eta = np.einsum("ij,jk,ik->i", x_matrix, cov, x_matrix)
    se_eta = np.sqrt(np.maximum(var_eta, 0))

    prob = sigmoid(float(eta[0]))
    ci_low = sigmoid(float(eta[0] - 1.96 * se_eta[0]))
    ci_high = sigmoid(float(eta[0] + 1.96 * se_eta[0]))

    return prob, ci_low, ci_high, X_input


def get_local_shap(X_input, model_artifact):
    background = model_artifact["train_X"].sample(n=min(100, len(model_artifact["train_X"])), random_state=42)
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
    params = model_artifact["logit_model"].params
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
