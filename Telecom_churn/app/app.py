# =====================
# 1️⃣ Imports
# =====================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.metrics import precision_recall_curve
from pathlib import Path
import base64
from PIL import Image

# =====================
# 2️⃣ Configuración general
# =====================
st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="📉",
    layout="wide"
)

# =====================
# 3️⃣ Rutas absolutas
# =====================
BASE_DIR = Path(__file__).resolve().parent.parent  # raíz del proyecto

MODEL_PATH = BASE_DIR / "models/final_model/best_model.pkl"
X_TEST_PATH = BASE_DIR / "data/processed/X_test.csv"
Y_TEST_PATH = BASE_DIR / "data/processed/y_test.csv"
ASSETS_DIR = BASE_DIR / "assets"
BANNER_IMAGE = ASSETS_DIR / "churn_img2.png"

# =====================
# 4️⃣ Banner
# =====================
def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_base64 = get_base64_image(BANNER_IMAGE)

st.markdown(f"""
<style>
.banner {{
    background-image: url("data:image/png;base64,{img_base64}");
    background-size: cover;
    background-position: center;
    height: 300px;
    padding: 70px 50px;
    border-radius: 20px;
    color: white;
}}
</style>

<div class="banner">
    <h1>📉 Customer Churn Prediction</h1>
    <h3>Machine Learning powered retention insights</h3>
</div>
""", unsafe_allow_html=True)


# =====================
# 5️⃣ Slogan centrado
# =====================
st.markdown(
    "<h2 style='text-align:center; color:#2c3e50; font-weight:500;'>"
    "<em>Turn customer data into retention decisions.</em>"
    "</h2>",
    unsafe_allow_html=True
)

# =====================
# 6️⃣ Recuadro azul claro overview
# =====================
st.markdown("""
<style>
.info-box {
    background-color: #e8f2ff;
    padding: 20px 25px;
    border-radius: 12px;
    border-left: 6px solid #4da3ff;
    font-size: 16px;
    line-height: 1.6;
}
</style>

<div class="info-box">
    <strong>📌 Application Overview</strong><br><br>
    This application leverages a machine learning model to estimate the probability of customer churn.
    It is designed to support <strong>data-driven retention strategies</strong> by enabling:
    <ul>
        <li>📁 Batch churn prediction from Excel files</li>
        <li>🎯 Customer risk segmentation based on churn probability</li>
        <li>📈 A Precision–Recall simulator to evaluate business trade-offs and decision thresholds</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# =====================
# 7️⃣ Constantes y funciones
# =====================
FEATURES = {
    "Tenure in Months": "Number of months the customer has stayed with the company.",
    "Monthly Charge": "Monthly amount charged to the customer.",
    "Avg Monthly GB Download": "Average monthly data usage (GB).",
    "Contract": "Type of customer contract.",
    "Paperless Billing": "Whether the customer uses paperless billing.",
    "Payment Method": "Customer payment method.",
    "Unlimited Data": "Whether the customer has unlimited data."
}

def assign_risk(prob):
    if prob < 45:
        return "Low"
    elif prob <= 65:
        return "High"
    else:
        return "Very High"

def recall_at_precision(y_true, y_proba, precision_target):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    valid_recalls = recall[precision >= precision_target]
    if len(valid_recalls) == 0:
        return None
    return valid_recalls.max()

# =====================
# 8️⃣ Carga recursos
# =====================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_test_data():
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).values.ravel()
    return X_test, y_test

model = load_model()
X_test, y_test = load_test_data()

# =====================
# 9️⃣ Selector de modo
# =====================
mode = st.radio(
    "Select interaction mode:",
    ["📁 Batch churn prediction", "📈 Precision–Recall simulator"]
)

# ==========================================================
# 📁 MODO 1 — BATCH PREDICTION
# ==========================================================
if mode == "📁 Batch churn prediction":

    st.subheader("🔎 Feature information")
    with st.expander("Click to see feature descriptions"):
        for feature, desc in FEATURES.items():
            st.markdown(f"**{feature}**: {desc}")

    st.subheader("📥 Download input template")
    template_df = pd.DataFrame(columns=["Customer ID"] + list(FEATURES.keys()))
    excel_bytes = BytesIO()
    with pd.ExcelWriter(excel_bytes, engine="openpyxl") as writer:
        template_df.to_excel(writer, index=False)
    st.download_button(
        label="Download Excel template",
        data=excel_bytes.getvalue(),
        file_name="churn_input_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.subheader("📤 Upload filled Excel file")
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
    if uploaded_file:
        df_uploaded = pd.read_excel(uploaded_file)
        expected_cols = ["Customer ID"] + list(FEATURES.keys())
        missing_cols = [c for c in expected_cols if c not in df_uploaded.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            st.success("File uploaded successfully ✔️")
            X = df_uploaded[list(FEATURES.keys())]
            churn_proba = model.predict_proba(X)[:, 1] * 100
            df_uploaded["Churn Probability (%)"] = churn_proba
            df_uploaded["Risk Level"] = df_uploaded["Churn Probability (%)"].apply(assign_risk)

            st.subheader("📊 Prediction preview")
            st.dataframe(df_uploaded.head(20))

            st.subheader("📈 Risk distribution")
            risk_dist = df_uploaded["Risk Level"].value_counts(normalize=True) * 100
            st.bar_chart(risk_dist)

            csv_output = df_uploaded.to_csv(index=False)
            st.download_button(
                label="📥 Download predictions CSV",
                data=csv_output,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )

# ==========================================================
# 📈 MODO 2 — PRECISION–RECALL SIMULATOR
# ==========================================================
if mode == "📈 Precision–Recall simulator":

    st.subheader("🎯 Precision–Recall trade-off")
    st.write(
        """
        This tool helps answer business questions like:
        **“If we want at least X% precision, what recall can we expect?”**
        """
    )

    precision_target = st.slider("Select desired precision (%)", min_value=50, max_value=95, value=70, step=1)
    y_proba_test = model.predict_proba(X_test)[:, 1]
    recall_value = recall_at_precision(y_true=y_test, y_proba=y_proba_test, precision_target=precision_target / 100)

    if recall_value is None:
        st.warning("This precision level cannot be achieved by the model.")
    else:
        st.metric("Expected Recall (%)", f"{recall_value * 100:.2f}")

        # Curva PR compacta
        precision, recall, _ = precision_recall_curve(y_test, y_proba_test)
        fig = plt.figure(figsize=(5, 3), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(recall, precision)
        ax.axhline(y=precision_target / 100, linestyle="--", label="Target precision")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision–Recall Curve")
        ax.legend()
        ax.grid(alpha=0.3)

        # Centrado en la página
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.pyplot(fig, use_container_width=False)


