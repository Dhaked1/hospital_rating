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

#<--------------------------------------------------------------------------------------------------------------------------------------->
## for nurse survey

try:
    model = load_model('staff_new_cnn.keras')
    with open('staff_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

st.title("Hospital Star Rating Prediction")
st.write("Answer the following questions about staff performance to predict the hospital's star rating.")

# Define answer options
answer_options = ["Always", "Usually", "Sometimes", "Never"]    

st.subheader("staff Communication")
staff_communication = st.selectbox(
    "patient who reported that when reciving new medication the staff communicated what the medication was for?",
    options=answer_options,
    key="staff_communication"
)
st.subheader("staff Explanation")
staff_explain = st.selectbox(
    "patient who reported that staff explained about medicines?",
    options=answer_options,
    key="staff_explain"
)
st.subheader("Nurse discussion about medicine")
staff_discussion = st.selectbox(
    "patient who reported that when receving new medication the staff discussed possible side effects?",
    options=answer_options,
    key="staff_respect"
)

# Button to make prediction
if st.button("Predict Star Rating"):
    # Map survey answers to percentages
    comm_percentages = map_survey_answer_to_percentages(staff_communication)
    explain_percentages = map_survey_answer_to_percentages(staff_explain)
    listen_percentages = map_survey_answer_to_percentages(staff_discussion)

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
        listen_percentages[1]  # H_DOCTOR_LISTEN_U_P
        # H_DOCTOR_RESPECT_U_P
    ]).reshape(1, -1)

    # Scale the input data and reshape for CNN
    try:
        input_scaled = scaler.transform(input_data)
        # Reshape for 1D CNN: (samples, timesteps, features)
        input_scaled = input_scaled.reshape(1, 9, 1)
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
    