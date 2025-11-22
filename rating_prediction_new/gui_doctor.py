import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
# Function to map survey answers to percentages
def map_survey_answer_to_percentages(answer_choice):
    """Maps survey responses to percentages based on dataset distribution."""
    if answer_choice == "Always":
        return [90, 8, 2]  # A_P, U_P, SN_P (high quality, ~5 stars)
    elif answer_choice == "Usually":
        return [85, 10, 5]  # Moderate quality, ~4 stars
    elif answer_choice == "Sometimes":
        return [75, 15, 10]  # Lower quality, ~3 stars
    elif answer_choice == "Never":
        return [20, 30, 50]  # Poor quality, ~1-2 stars
    else:
        return [33, 33, 34]  # Neutral default, balanced

# Load the trained model and scaler
try:
    model = load_model('doctor_new_cnn.keras')
    with open('doctor_new_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# Streamlit app
st.title("Hospital Star Rating Prediction")
st.write("Answer the following questions about doctor performance to predict the hospital's star rating.")

# Define answer options
answer_options = ["Always", "Usually", "Sometimes", "Never"]

# Input fields for survey questions
st.subheader("Doctor Communication")
doctor_communication = st.selectbox(
    "How often did doctors communicate well with patients?",
    options=answer_options,
    key="doctor_communication"
)

st.subheader("Doctor Explanation")
doctor_explain = st.selectbox(
    "How often did doctors explain things in a way patients could understand?",
    options=answer_options,
    key="doctor_explain"
)

st.subheader("Doctor Listening")
doctor_listen = st.selectbox(
    "How often did doctors listen carefully to patients?",
    options=answer_options,
    key="doctor_listen"
)

st.subheader("Doctor Respect")
doctor_respect = st.selectbox(
    "How often did doctors treat patients with respect?",
    options=answer_options,
    key="doctor_respect"
)

# Button to make prediction
if st.button("Predict Star Rating"):
    # Map survey answers to percentages
    comm_percentages = map_survey_answer_to_percentages(doctor_communication)
    explain_percentages = map_survey_answer_to_percentages(doctor_explain)
    listen_percentages = map_survey_answer_to_percentages(doctor_listen)
    respect_percentages = map_survey_answer_to_percentages(doctor_respect)

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

