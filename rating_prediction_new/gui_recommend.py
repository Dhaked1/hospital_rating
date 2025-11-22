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
        return [99, 8, 2]  # A_P, U_P, SN_P (high quality, ~5 stars)
    elif answer_choice == "Usually":
        return [75, 10, 5]  # Moderate quality, ~4 stars
    elif answer_choice == "Sometimes":
        return [65, 15, 10]  # Lower quality, ~3 stars
    elif answer_choice == "Never":
        return [20, 30, 50]  # Poor quality, ~1-2 stars
    else:
        return [33, 33, 34]  # Neutral default, balanced

# Load the trained model and scaler
try:
    model = load_model('recommend_cnn.keras')
    with open('recommend_scaler.pkl', 'rb') as f:
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
st.subheader("recommendation")
recom = st.selectbox(
    "patient would recommend the hospital?",
    options=answer_options,
    key="recom"
)



# Button to make prediction
if st.button("Predict Star Rating"):
    # Map survey answers to percentages
    comm_percentages = map_survey_answer_to_percentages(recom)
 

    # Combine percentages into input array
    input_data = np.array([
        comm_percentages[0],    # H_COMP_2_A_P
        comm_percentages[2],    # H_COMP_2_SN_P
        comm_percentages[1],    # H_COMP_2_U_P
         
    ]).reshape(1, -1)

    # Scale the input data and reshape for CNN
    try:
        input_scaled = scaler.transform(input_data)
        # Reshape for 1D CNN: (samples, timesteps, features)
        input_scaled = input_scaled.reshape(1, 3, 1)
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
