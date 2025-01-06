import streamlit as st
import numpy as np
import joblib
import os

# Debug: Check current working directory
st.write(f"Current Working Directory: {os.getcwd()}")

# Construct the full path to the model file
model_path = os.path.join(os.getcwd(), "second_Task/SLReg.joblib")

# Debug: Check if the file exists
if os.path.isfile(model_path):
    st.write(f"Model file found at: {model_path}")
else:
    st.error(f"Model file not found at: {model_path}")

# Load your model
try:
    model = joblib.load(model_path)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error(f"Model file not found: {model_path}. Please ensure it exists.")
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")

# Streamlit App Title
st.title("Salary Predictor")

# Description
st.write("Know your estimated salary based on years of experience.")

# Input from user
year = st.number_input("Enter Years of Experience:", min_value=0, max_value=50, value=0, step=1)

# Predict button
if st.button("Submit"):
    try:
        if 'model' in locals():  # Check if the model is loaded
            new = np.array([[year]])  # Ensure a 2D array
            result = model.predict(new)
            st.success(f"Approximately {result[0]:.2f} US Dollars.")
        else:
            st.error("Model is not loaded. Prediction cannot proceed.")
    except Exception as e:
        st.error("Invalid input or an error occurred during prediction.")
        st.write(f"Error: {e}")
