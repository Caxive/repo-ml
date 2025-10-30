import streamlit as st
import numpy as np
import joblib

# -------------------------------
# Load Trained Model
# -------------------------------
model = joblib.load("insurance-response-predictor.pkl")

# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(page_title="Vehicle Insurance Response Predictor", page_icon="🚗")
st.title("🚗 Vehicle Insurance Response Predictor")
st.write("---")

st.header("🧾 Fill Out the Details")

# -------------------------------
# Input Fields
# -------------------------------
gender = st.selectbox("Your Gender", ["Select", "Male", "Female"])
age = st.number_input("Your Age", min_value=18, max_value=100, step=1)
dl = st.selectbox("Do you have a Driving License?", ["Select", "Yes", "No"])
reg_code = st.number_input("Your Region Code", min_value=0, step=1)
prev_insured = st.selectbox("Are you previously insured?", ["Select", "Yes", "No"])
vehicle_age = st.selectbox("What's your vehicle age?", ["Select", "Less than 1 Year", "1-2 Years", "More than 2 Years"])
vehicle_damage = st.selectbox("Has your vehicle ever been damaged?", ["Select", "Yes", "No"])
annual_premium = st.number_input("Your Annual Premium", min_value=1000.0, max_value=100000.0, step=100.0)
sales_channel = st.number_input("Policy Sales Channel", min_value=1, max_value=200, step=1)
vintage = st.number_input("Vintage (Days with company)", min_value=0, max_value=300, step=1)

st.write("---")

# -------------------------------
# Preprocess Inputs
# -------------------------------
def preprocess_inputs():
    try:
        Male = 1 if gender.lower() == "male" else 0
        DL = 1 if dl.lower() == "yes" else 0
        PrevIns = 1 if prev_insured.lower() == "yes" else 0
        VehDam = 1 if vehicle_damage.lower() == "yes" else 0
        lessThanOne = 1 if vehicle_age == "Less than 1 Year" else 0
        moreThanTwo = 1 if vehicle_age == "More than 2 Years" else 0

        return np.array([[Male, age, DL, reg_code, PrevIns, VehDam,
                          annual_premium, sales_channel, vintage,
                          lessThanOne, moreThanTwo]])
    except Exception as e:
        st.error(f"Error processing input: {e}")
        return None

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict"):
    if "Select" in [gender, dl, prev_insured, vehicle_age, vehicle_damage]:
        st.warning("⚠️ Please fill out all fields before predicting.")
    else:
        input_data = preprocess_inputs()
        if input_data is not None:
            try:
                prediction = model.predict(input_data)
                if prediction[0] == 1:
                    st.success("✅ The customer **will buy** the insurance.")
                else:
                    st.error("❌ The customer **will not buy** the insurance.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
