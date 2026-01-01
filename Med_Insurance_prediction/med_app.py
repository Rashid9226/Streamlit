import numpy as np
import streamlit as st
import tensorflow as tf
import traceback

st.set_page_config(page_title="Medical Insurance Predictor", layout="centered")
st.title("🏥 Medical Insurance Cost Prediction")


# ---------------------------------------------------
# Load trained model (cached, with error handling)
# ---------------------------------------------------
@st.cache_resource
def load_trained_model(path: str):
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception:
        st.error("Failed to load the model. See details below.")
        st.text(traceback.format_exc())
        return None

model = load_trained_model("med_ins_Pred.keras")
if model is None:
    st.stop()

# ---------------------------------------------------
# User Inputs
# ---------------------------------------------------
age = st.number_input("Age", min_value=18, max_value=64, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
bmi = st.number_input("BMI", min_value=16.0, max_value=53.0)
children = st.selectbox("Children", [0, 1, 2, 3, 4, 5])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox(
    "Region",
    ["southwest", "southeast", "northwest", "northeast"]
)

# ---------------------------------------------------
# Encoding (Must match training)
# ---------------------------------------------------
gender = 1 if gender == "Male" else 0
smoker = 1 if smoker == "yes" else 0

region_southwest = 1 if region == "southwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_northwest = 1 if region == "northwest" else 0
region_northeast = 1 if region == "northeast" else 0

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
if st.button("Predict Insurance Cost"):
    input_data = np.array([[
        age,
        gender,
        bmi,
        children,
        smoker,
        region_southwest,
        region_southeast,
        region_northwest,
        region_northeast
    ]], dtype=np.float32)

    prediction = model.predict(input_data)
    st.success(f"💰 Estimated Insurance Cost: {prediction[0][0]:.2f}")
