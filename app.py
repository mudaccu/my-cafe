"""TMUC Unit 25 - simple Streamlit front end for a trusted classroom model.

Run training first, then: python -m streamlit run app.py
"""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "cafe_pipeline.joblib"

st.set_page_config(page_title="Campus Cafe ML Demo", page_icon="☕", layout="centered")
st.title("☕ Campus Cafe Wait-Time Demo")
st.caption("TMUC Unit 25 | Supervised regression | Classroom illustration")
st.info("Uses synthetic training records. This is not a real cafe wait-time service.")

if not MODEL_PATH.exists():
    st.error("Model file missing. Run `python train_model.py` before starting this app.")
    st.stop()

# joblib can execute arbitrary code when loading an untrusted file.
# Only load this model file because you generated it with the trusted training script.
@st.cache_resource
def load_trusted_pipeline():
    return joblib.load(MODEL_PATH)

model = load_trusted_pipeline()

with st.form("estimate_form"):
    queue_people = st.number_input("People currently in queue", min_value=0, max_value=34, value=12, step=1)
    open_counters = st.number_input("Open service counters", min_value=1, max_value=4, value=2, step=1)
    order_items = st.number_input("Items in the order", min_value=1, max_value=5, value=2, step=1)
    service_period = st.selectbox("Service period", options=["Normal", "Rush"])
    submitted = st.form_submit_button("Predict waiting time")

if submitted:
    # Feature names must match the names used at model training time.
    user_input = pd.DataFrame([{
        "queue_people": queue_people,
        "open_counters": open_counters,
        "order_items": order_items,
        "service_period": service_period,
    }])
    estimate = float(model.predict(user_input)[0])
    st.metric("Estimated waiting time", f"{estimate:.1f} minutes")
    st.write("Input sent to the model:")
    st.dataframe(user_input, hide_index=True)
    st.caption("The output is a pattern-based estimate, not a guarantee.")

st.divider()
st.write("**Team challenge:** What happens when you increase queue length or open another counter? Why?")
