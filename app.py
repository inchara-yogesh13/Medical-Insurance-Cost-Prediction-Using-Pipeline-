import streamlit as st
import pandas as pd
import joblib
pipeline = joblib.load("pipeline.pkl")
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Medical Insurance Cost Prediction")

st.write(
    "Enter your details below to estimate your annual medical insurance cost."
)
age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0,
    step=0.1
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0
)

smoker = st.selectbox(
    "Smoker",
    ["yes", "no"]
)

region = st.selectbox(
    "Region",
    [
        "northeast",
        "northwest",
        "southeast",
        "southwest"
    ]
)
if st.button("Predict Insurance Cost"):
    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })
    prediction = pipeline.predict(input_data)
    st.success(
        f"Estimated Medical Insurance Cost: ₹ {prediction[0]:,.2f}"
    )
        
