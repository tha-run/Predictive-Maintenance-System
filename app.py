"""
========================================================
  Industrial IoT Predictive Maintenance System
  Frontend : Streamlit Application
  Purpose  : Real-time machine failure probability prediction
========================================================
"""

import joblib
import pandas as pd
import streamlit as st

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IoT Predictive Maintenance",
    page_icon="⚙️",
    layout="centered",
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = "model/predictive_maintenance_model.pkl"

# These must exactly match the columns the model was trained on
# (original names with spaces and brackets as used during training)
FEATURE_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type_H",
    "Type_L",
    "Type_M",
]

FAILURE_THRESHOLD = 0.3   # probability above which we raise a warning


# ── Load Model (cached so it only loads once) ─────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained RandomForest model from disk."""
    return joblib.load(MODEL_PATH)


model = load_model()


# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚙️ Industrial IoT Predictive Maintenance")
st.markdown(
    "Enter the current sensor readings below to predict the **probability of machine failure** in real time."
)
st.divider()


# ── Sidebar — About ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
        **Model:** RandomForestClassifier  
        **Trained on:** AI4I 2020 Industrial IoT Dataset  
        **Target:** Machine Failure (0 / 1)  
        **Decision threshold:** `> 0.30`

        ---
        **Features used:**
        - Air temperature
        - Process temperature
        - Rotational speed
        - Torque
        - Tool wear
        - Machine type (L / M / H)
        """
    )
    st.info("No data is stored or transmitted. All inference runs locally.")


# ── Input Form ────────────────────────────────────────────────────────────────
st.subheader("🔧 Sensor Input")

col1, col2 = st.columns(2)

with col1:
    air_temp = st.number_input(
        "Air Temperature (K)",
        min_value=250.0, max_value=400.0,
        value=298.1, step=0.1,
        help="Typical range: 295 – 304 K",
    )
    rotational_speed = st.number_input(
        "Rotational Speed (rpm)",
        min_value=0, max_value=3000,
        value=1551, step=1,
        help="Typical range: 1168 – 2886 rpm",
    )
    tool_wear = st.number_input(
        "Tool Wear (min)",
        min_value=0, max_value=300,
        value=108, step=1,
        help="Cumulative tool wear in minutes (0 – 253)",
    )

with col2:
    process_temp = st.number_input(
        "Process Temperature (K)",
        min_value=250.0, max_value=400.0,
        value=308.6, step=0.1,
        help="Typical range: 305 – 314 K",
    )
    torque = st.number_input(
        "Torque (Nm)",
        min_value=0.0, max_value=100.0,
        value=42.8, step=0.1,
        help="Typical range: 3.8 – 76.6 Nm",
    )
    machine_type = st.selectbox(
        "Machine Type",
        options=["L", "M", "H"],
        index=2,
        help="L = Low quality, M = Medium quality, H = High quality",
    )

st.divider()


# ── Predict Button ────────────────────────────────────────────────────────────
if st.button("🔍 Predict Failure Probability", use_container_width=True, type="primary"):

    # ── Build one-hot encoded input row ──────────────────────────────────────
    # Start with all Type_* columns set to 0, then flip the selected one
    input_data = {
        "Air temperature [K]"     : air_temp,
        "Process temperature [K]" : process_temp,
        "Rotational speed [rpm]"  : rotational_speed,
        "Torque [Nm]"             : torque,
        "Tool wear [min]"         : tool_wear,
        "Type_H"                  : 1 if machine_type == "H" else 0,
        "Type_L"                  : 1 if machine_type == "L" else 0,
        "Type_M"                  : 1 if machine_type == "M" else 0,
    }

    # Convert to DataFrame and align column order exactly as training
    input_df = pd.DataFrame([input_data])[FEATURE_COLUMNS]

    # ── Run inference ─────────────────────────────────────────────────────────
    failure_prob  = model.predict_proba(input_df)[0, 1]
    predicted_label = int(failure_prob >= FAILURE_THRESHOLD)

    # ── Display Results ───────────────────────────────────────────────────────
    st.subheader("📊 Prediction Result")

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.metric(
            label="Failure Probability",
            value=f"{failure_prob:.4f}",
            delta=f"{'Above' if failure_prob > FAILURE_THRESHOLD else 'Below'} threshold ({FAILURE_THRESHOLD})",
            delta_color="inverse",
        )

    with res_col2:
        st.metric(
            label="Predicted Status",
            value="⚠️ FAILURE" if predicted_label == 1 else "✅ NORMAL",
        )

    # ── Alert Banner ──────────────────────────────────────────────────────────
    if failure_prob > FAILURE_THRESHOLD:
        st.warning(
            f"⚠️ **High failure risk detected!**  "
            f"Failure probability is **{failure_prob:.4f}** — immediate inspection recommended.",
            icon="🚨",
        )
    else:
        st.success(
            f"✅ **Machine operating normally.**  "
            f"Failure probability is **{failure_prob:.4f}** — no action required.",
            icon="✅",
        )

    # ── Input Summary ─────────────────────────────────────────────────────────
    with st.expander("📋 View Input Summary"):
        st.dataframe(
            input_df.T.rename(columns={0: "Value"}),
            use_container_width=True,
        )
