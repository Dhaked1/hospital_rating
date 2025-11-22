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

#<---------------------------------------------------------------------------------------------------------------------------------------------------------->
## for nurse survey

try:
    model = load_model('help_new_cnn.keras')
    with open('help_new_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

st.title("Hospital Star Rating Prediction")
st.write("Answer the following questions about cleaneness to predict the hospital's star rating.")

# Define answer options
answer_options = ["Always", "Usually", "Sometimes", "Never"]    

help1 = st.selectbox(
    "patients who repoted that theyrecive help as soon as they wanted?",
    options=answer_options,
    key="help1"
)
help2 = st.selectbox(
    "patients getting help using help button?",
    options=answer_options,
    key="help2"
)
help3 = st.selectbox(
    "patients getting bathroom help?",
    options=answer_options,
    key="help3"
)

# Button to make prediction
if st.button("Predict Star Rating"):
    # Map survey answers to percentages
    respect_percentages = map_survey_answer_to_percentages(help1)
    explain_percentages = map_survey_answer_to_percentages(help2)
    listen_percentages = map_survey_answer_to_percentages(help3)


    # Combine percentages into input array
    input_data = np.array([
        respect_percentages[0],  
        respect_percentages[2],  
        respect_percentages[1],   
        explain_percentages[0],  
        explain_percentages[2],  
        explain_percentages[1],  
        listen_percentages[0],   
        listen_percentages[2],   
        listen_percentages[1]
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
    st.write(f"Doctor Respect - Always: {respect_percentages[0]}%, Usually: {respect_percentages[1]}%, Sometimes/Never: {respect_percentages[2]}%")
    st.write(f"Doctor Explanation - Always: {explain_percentages[0]}%, Usually: {explain_percentages[1]}%, Sometimes/Never: {explain_percentages[2]}%")
    st.write(f"Doctor Listening - Always: {listen_percentages[0]}%, Usually: {listen_percentages[1]}%, Sometimes/Never: {listen_percentages[2]}%")
