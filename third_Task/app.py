import streamlit as st
import joblib
import numpy as np
import os

st.write(f"Current Working Directory: {os.getcwd()}")

# Construct the full path to the model file
model_path = os.path.join(os.getcwd(), "third_Task/House_Pricing.joblib")

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

# Streamlit app
st.title("California Housing Price Prediction App")
st.write("Enter the features to predict the house price in California.")

# Collect user input for features
MedInc = st.number_input("Median Income in block group (MedInc):", value=0.0, help="e.g., 3.87 for $38,700/year")
HouseAge = st.number_input("Median age of houses in block group (HouseAge):", value=0.0, help="e.g., 25 for 25 years")
AveRooms = st.number_input("Average number of rooms per household (AveRooms):", value=0.0, help="e.g., 6.5")
AveBedrms = st.number_input("Average number of bedrooms per household (AveBedrms):", value=0.0, help="e.g., 1.1")
Population = st.number_input("Block group population (Population):", value=0.0, help="e.g., 1200")
AveOccup = st.number_input("Average number of occupants per household (AveOccup):", value=0.0, help="e.g., 3.5")
Latitude = st.number_input("Block group latitude (Latitude):", value=0.0, help="e.g., 34.1")
Longitude = st.number_input("Block group longitude (Longitude):", value=0.0, help="e.g., -118.2")

# Prediction button
if st.button("Predict"):
    # Create a NumPy array of inputs
    features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
    
    # Make prediction
    prediction = model.predict(features)[0]
    
    # Display the prediction
    st.success(f"Estimated House Price: ${prediction * 100000:.2f}")

# Footer
st.write("---")
st.caption("Note: Predictions are based on the model trained on the California Housing dataset.")

