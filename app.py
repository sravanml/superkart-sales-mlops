"""Streamlit frontend for SuperKart predictions."""
import joblib
import pandas as pd
import streamlit as st

from pipeline import FEATURES, MODEL

st.title("SuperKart Sales Estimator")
st.write("Estimate sales revenue for a product at an existing SuperKart store.")


@st.cache_resource
def load_model():
    return joblib.load(MODEL)


if not MODEL.is_file():
    st.error("The trained model is missing. Run the training workflow first.")
    st.stop()

artifact = load_model()
with st.form("product_store"):
    inputs = {}
    for field in FEATURES:
        label = field.replace("_", " ")
        if field in artifact["categories"]:
            inputs[field] = st.selectbox(label, artifact["categories"][field])
        elif field == "Store_Establishment_Year":
            inputs[field] = st.number_input(label, value=int(artifact["defaults"][field]), step=1)
        else:
            inputs[field] = st.number_input(label, min_value=0.0,
                                            value=artifact["defaults"][field], format="%.3f")
    submitted = st.form_submit_button("Estimate sales")

if submitted:
    frame = pd.DataFrame([inputs], columns=FEATURES)
    estimate = artifact["pipeline"].predict(frame)[0]
    st.metric("Estimated product–store sales", f"{estimate:,.2f}")
