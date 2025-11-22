import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
# Function to map survey answers to percentages
def map_survey_answer_to_percentages(answer_choice):
    """Maps survey responses to percentages based on dataset distribution."""
    if answer_choice == "Strongly Agree":
        return [20,80, 0]  # A_P, U_P, SN_P (high quality, ~5 stars)
    elif answer_choice == "Agree":
        return [35,60,5]  # Moderate quality, ~4 stars
    elif answer_choice == "Disagree":
        return [45,45, 10]  # Lower quality, ~3 stars
    elif answer_choice == "Strongly Disagree":
        return [20, 30, 50]  #j Poor quality, ~1-2 stars
    else:
        return [33, 33, 34]  # Neutral default, balanced

# Load the trained model and scaler
try:
    model = load_model('care_cnn.keras')
    with open('care_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# Streamlit app
st.title("Hospital Star Rating Prediction")
st.write("Answer the following questions about care in hospital to predict the hospital's star rating.")

# Define answer options
answer_options = ["Strongly Agree", "Agree", "Disagree", "Strongly Disagree"]

# Input fields for survey questions
care1 = st.selectbox(
    "who they understant their care when they left the hospital?",
    options=answer_options,
    key="care1"
)

care2 = st.selectbox(
    "patient who agree that the staff took my preferences into accounts when determining my health care needs?",
    options=answer_options,
    key="care2"
)

care3 = st.selectbox(
    "patients who agree that they understood their reponsibilities in managing their health?",
    options=answer_options,
    key="care3"
)

care4 = st.selectbox(
    "patients who agree that they understood the purpose of their medication when leaving the hospital?",
    options=answer_options,
    key="care4"
)

# Button to make prediction
if st.button("Predict Star Rating"):
    # Map survey answers to percentages
    comm_percentages = map_survey_answer_to_percentages(care1)
    explain_percentages = map_survey_answer_to_percentages(care2)
    listen_percentages = map_survey_answer_to_percentages(care3)
    respect_percentages = map_survey_answer_to_percentages(care4)

    # Combine percentages into input array
    input_data = np.array([
        comm_percentages[0],    # H_COMP_2_A_P
        comm_percentages[2],    # H_COMP_2_SN_P
        comm_percentages[1],    # H_COMP_2_U_P
        explain_percentages[0],  # H_DOCTOR_EXPLAIN_A_P
        explain_percentages[2],  # H_DOCTOR_EXPLAIN_SN_P
        explain_percentages[1],  # H_DOCTOR_EXPLAIN_U_P
        listen_percentages[0],   # H_DOCTOR_LISTEN_A_P
        listen_percentages[2],   # H_DOCTOR_LISTEN_SN_P
        listen_percentages[1],   # H_DOCTOR_LISTEN_U_P
        respect_percentages[0],  # H_DOCTOR_RESPECT_A_P
        respect_percentages[2],  # H_DOCTOR_RESPECT_SN_P
        respect_percentages[1]   # H_DOCTOR_RESPECT_U_P
    ]).reshape(1, -1)

    # Scale the input data and reshape for CNN
    try:
        input_scaled = scaler.transform(input_data)
        # Reshape for 1D CNN: (samples, timesteps, features)
        input_scaled = input_scaled.reshape(1, 12, 1)
    except Exception as e:
        st.error(f"Error scaling or reshaping input data: {e}")
        st.stop()

    # Make prediction
    try:
        prediction = model.predict(input_scaled)
        predicted_rating = np.argmax(prediction, axis=1) + 1  # Convert back to 1-5 scale
        st.success(f"Predicted Hospital Star Rating: **{predicted_rating[0]}**")
    except Exception as e:
        st.error(f"Error making prediction: {e}")

    # Display input percentages for verification
    st.subheader("Input Percentages")
    st.write("Based on your selections, the following percentages were used as input to the model:")
    st.write(f"Doctor Communication - Always: {comm_percentages[0]}%, Usually: {comm_percentages[1]}%, Sometimes/Never: {comm_percentages[2]}%")
    st.write(f"Doctor Explanation - Always: {explain_percentages[0]}%, Usually: {explain_percentages[1]}%, Sometimes/Never: {explain_percentages[2]}%")
    st.write(f"Doctor Listening - Always: {listen_percentages[0]}%, Usually: {listen_percentages[1]}%, Sometimes/Never: {listen_percentages[2]}%")
    st.write(f"Doctor Respect - Always: {respect_percentages[0]}%, Usually: {respect_percentages[1]}%, Sometimes/Never: {respect_percentages[2]}%")

