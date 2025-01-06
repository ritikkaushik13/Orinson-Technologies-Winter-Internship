import streamlit as st
import numpy as np
import joblib

# Load your model
model = joblib.load("SLReg.joblib")

# Streamlit App Title
st.title("Salary Predictor")

# Description
st.write("Know your estimated salary based on years of experience.")

# Input from user
year = st.number_input("Enter Years of Experience:", min_value=0, max_value=50, value=0, step=1)

# Predict button
if st.button("Submit"):
    try:
        new = np.array([[year]])  # Ensure a 2D array
        result = model.predict(new)
        st.success(f"Approximately {result[0]:.2f} US Dollars.")
    except Exception as e:
        st.error("Invalid input or an error occurred during prediction.")
        st.write(f"Error: {e}")
