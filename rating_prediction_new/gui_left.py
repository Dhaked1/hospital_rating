import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
# Function to map survey answers to percentages
def map_survey_answer_to_percentages(answer_choice):
    """Maps survey responses to percentages based on dataset distribution."""
    if answer_choice == "Yes":
        return [90, 8]  # A_P, U_P, SN_P (high quality, ~5 stars)
    elif answer_choice == "No":
        return [9, 20]  # Neutral default, balanced

#<---------------------------------------------------------------------------------------------------------------------------------------------------------->
## for nurse survey

try:
    model = load_model('left_hospital_cnn.keras')
    with open('left_hospital_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

st.title("Hospital Star Rating Prediction")
st.write("Answer the following questions about cleaneness to predict the hospital's star rating.")

# Define answer options
answer_options = ["Yes", "No"]    

help1 = st.selectbox(
    "patient who reported , Are they discuss weather they would need help after discharge?",
    options=answer_options,
    key="help1"
)
help2 = st.selectbox(
    "patients who reported, are they recive written information about possible symptones to look and for after discharge?",
    options=answer_options,
    key="help2"
)


# Button to make prediction
if st.button("Predict Star Rating"):
    # Map survey answers to percentages
    respect_percentages = map_survey_answer_to_percentages(help1)
    explain_percentages = map_survey_answer_to_percentages(help2)
    


    # Combine percentages into input array
    input_data = np.array([
        respect_percentages[0],  
        respect_percentages[1],     
        explain_percentages[0],  
        explain_percentages[1] 
        
    ]).reshape(1, -1)

    # Scale the input data and reshape for CNN
    try:
        input_scaled = scaler.transform(input_data)
        # Reshape for 1D CNN: (samples, timesteps, features)
        input_scaled = input_scaled.reshape(1, 4, 1)
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
    st.write(f"Doctor Respect - Yes: {respect_percentages[0]}%, No: {respect_percentages[1]}%")
    st.write(f"Doctor Explanation - Yes: {explain_percentages[0]}%, No: {explain_percentages[1]}%")

