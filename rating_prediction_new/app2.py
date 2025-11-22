import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
 

## using care models  
# Load CNN model and scaler
model = load_model("care_cnn.keras")
with open("care_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# HCAHPS dropdown options (standard)
response_options = {
    "Always": 4,
    "Usually": 3,
    "Sometimes": 2,
    "Never": 1
}

# Define feature names and corresponding labels
input_features = {
    'H_COMP_7_A': "Pain well controlled",
    'H_COMP_7_D_SD': "Pain control - somewhat disagree",
    'H_COMP_7_SA': "Pain control - strongly agree",
    'H_CT_MED_A': "Staff explained medicine purpose",
    'H_CT_MED_D_SD': "Medicine explanation - somewhat disagree",
    'H_CT_MED_SA': "Medicine explanation - strongly agree",
    'H_CT_PREFER_A': "Staff respected treatment preference",
    'H_CT_PREFER_D_SD': "Treatment preference - somewhat disagree",
    'H_CT_PREFER_SA': "Treatment preference - strongly agree",
    'H_CT_UNDER_A': "Understood care responsibilities",
    'H_CT_UNDER_D_SD': "Care responsibilities - somewhat disagree",
    'H_CT_UNDER_SA': "Care responsibilities - strongly agree"
}

# App title
st.title("🏥 HCAHPS Hospital Rating Predictor (CNN Model)")
st.write("Fill out the patient care survey below to predict the hospital's star rating.")

# Collect inputs
user_input = []
for feature, label in input_features.items():
    response = st.selectbox(f"{label}:", list(response_options.keys()), key=feature)
    user_input.append(response_options[response])

# Predict button
if st.button("Predict Hospital Rating", key="care_rating_button"):
    # Convert to NumPy and scale
    input_array = np.array(user_input).reshape(1, -1)
    scaled_input = scaler.transform(input_array)

    # Reshape for CNN: (batch_size, timesteps, features)
    reshaped_input = scaled_input.reshape(scaled_input.shape[0], scaled_input.shape[1], 1)

    # Predict using CNN model
    prediction = model.predict(reshaped_input)
    predicted_rating = np.argmax(prediction) + 1  # 1-based rating

    st.success(f"⭐ Predicted Hospital Star Rating: **{predicted_rating}** out of 5")

    # Show confidence for each star
    st.subheader("📊 Prediction Probabilities:")
    for i, prob in enumerate(prediction[0], start=1):
        st.write(f"⭐ {i} Star: {prob:.2%}")


### << usoing clean models >> 
# Load the CNN model
model = load_model("clean_cnn.keras")

# Load the scaler
with open("clean_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define input features and corresponding labels
input_features = {
    'H_CLEAN_HSP_A_P': "Hospital room and bathroom were always clean",
    'H_CLEAN_HSP_SN_P': "Hospital was sometimes or never clean",
    'H_CLEAN_HSP_U_P': "Hospital was usually clean"
}

# Define dropdown options (as per HCAHPS)
response_options = {
    "Always": 4,
    "Usually": 3,
    "Sometimes": 2,
    "Never": 1
}

# App Title
st.title("🧼 Hospital Cleanliness Star Rating Predictor")
st.write("Please answer the questions based on your experience. The model will predict the hospital's cleanliness rating (1–5 stars).")

# Collect user inputs
user_input = []
for feature, question in input_features.items():
    response = st.selectbox(f"{question}:", list(response_options.keys()), key=feature)
    user_input.append(response_options[response])

# Prediction button
if st.button("Predict Cleanliness Rating", key="clean_rating_button"):
    # Convert to NumPy and reshape
    input_array = np.array(user_input).reshape(1, -1)

    # Scale the input
    scaled_input = scaler.transform(input_array)

    # Reshape for CNN: (samples, timesteps, features)
    reshaped_input = scaled_input.reshape((scaled_input.shape[0], scaled_input.shape[1], 1))

    # Predict
    prediction = model.predict(reshaped_input)
    predicted_rating = np.argmax(prediction) + 1  # 0-based to 1-based

    # Display results
    st.success(f"🧽 Predicted Cleanliness Star Rating: **{predicted_rating}** out of 5")

    st.subheader("Prediction Confidence:")
    for i, prob in enumerate(prediction[0], start=1):
        st.write(f"⭐ {i} Star: {prob:.2%}")



## << using doctor models >>

# Load trained model and scaler
model = load_model("doctor_new_cnn.keras")
with open("doctor_new_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# HCAHPS dropdown options
response_options = {
    "Always": 4,
    "Usually": 3,
    "Sometimes": 2,
    "Never": 1
}

# Feature labels
input_features = {
    'H_COMP_2_A_P': "Doctors always explained things clearly",
    'H_COMP_2_SN_P': "Doctors sometimes/never explained things clearly",
    'H_COMP_2_U_P': "Doctors usually explained things clearly",
    'H_DOCTOR_EXPLAIN_A_P': "Doctors always explained medications clearly",
    'H_DOCTOR_EXPLAIN_SN_P': "Doctors sometimes/never explained medications clearly",
    'H_DOCTOR_EXPLAIN_U_P': "Doctors usually explained medications clearly",
    'H_DOCTOR_LISTEN_A_P': "Doctors always listened carefully",
    'H_DOCTOR_LISTEN_SN_P': "Doctors sometimes/never listened carefully",
    'H_DOCTOR_LISTEN_U_P': "Doctors usually listened carefully",
    'H_DOCTOR_RESPECT_A_P': "Doctors always showed respect",
    'H_DOCTOR_RESPECT_SN_P': "Doctors sometimes/never showed respect",
    'H_DOCTOR_RESPECT_U_P': "Doctors usually showed respect"
}

st.title("🩺 Doctor Communication Rating Predictor")
st.write("Fill out the form below based on patient experience to predict the doctor communication star rating (1–5).")

# Collect input from user
user_input = []
for feature, label in input_features.items():
    response = st.selectbox(f"{label}:", list(response_options.keys()), key=feature)
    user_input.append(response_options[response])

# Predict rating
if st.button("Predict Hospital Rating", key="doctor_rating_button"):
    # Convert and scale
    input_array = np.array(user_input).reshape(1, -1)
    scaled_input = scaler.transform(input_array)

    # Reshape for CNN: (batch_size, timesteps, features)
    cnn_input = scaled_input.reshape((scaled_input.shape[0], scaled_input.shape[1], 1))

    # Predict
    prediction = model.predict(cnn_input)
    predicted_rating = np.argmax(prediction) + 1  # Ratings are 1-5

    st.success(f"⭐ Predicted Doctor Communication Star Rating: **{predicted_rating}** out of 5")

    # Show prediction confidence
    st.subheader("Prediction Probabilities:")
    for i, prob in enumerate(prediction[0], start=1):
        st.write(f"⭐ {i} Star: {prob:.2%}")


## << using help models >>

# Load trained model and scaler
model = load_model("help_new_cnn.keras")
with open("help_new_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# HCAHPS dropdown options
response_options = {
    "Always": 4,
    "Usually": 3,
    "Sometimes": 2,
    "Never": 1
}

# Feature labels
input_features = {
    'H_BATH_HELP_A_P': "Help with bathing: Always",
    'H_BATH_HELP_SN_P': "Help with bathing: Sometimes/Never",
    'H_BATH_HELP_U_P': "Help with bathing: Usually",
    'H_CALL_BUTTON_A_P': "Call button help: Always",
    'H_CALL_BUTTON_SN_P': "Call button help: Sometimes/Never",
    'H_CALL_BUTTON_U_P': "Call button help: Usually",
    'H_COMP_3_A_P': "Got help soon: Always",
    'H_COMP_3_SN_P': "Got help soon: Sometimes/Never",
    'H_COMP_3_U_P': "Got help soon: Usually"
}

st.title("🧑‍⚕️ Hospital Staff Help Rating Predictor")
st.write("Fill out the form below based on patient feedback to predict the staff help experience star rating (1–5).")

# Collect input from user
user_input = []
for feature, label in input_features.items():
    response = st.selectbox(f"{label}:", list(response_options.keys()), key=feature)
    user_input.append(response_options[response])

# Predict rating
if st.button("Predict Hospital Rating", key="help_rating_button"):
    # Convert and scale
    input_array = np.array(user_input).reshape(1, -1)
    scaled_input = scaler.transform(input_array)

    # Reshape for CNN input
    cnn_input = scaled_input.reshape((scaled_input.shape[0], scaled_input.shape[1], 1))

    # Predict
    prediction = model.predict(cnn_input)
    predicted_rating = np.argmax(prediction) + 1  # Convert from 0-based to 1-5 star

    st.success(f"⭐ Predicted Help Experience Star Rating: **{predicted_rating}** out of 5")

    # Show prediction probabilities
    st.subheader("Prediction Probabilities:")
    for i, prob in enumerate(prediction[0], start=1):
        st.write(f"⭐ {i} Star: {prob:.2%}")

## << using left hospital models >>



# Load model and scaler
model = load_model("left_hospital_cnn.keras")
with open("left_hospital_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define dropdown options
response_options = {
    "Yes": 1,
    "No": 0
}

# Feature labels mapping
input_features = {
    'H_DISCH_HELP_Y_P': "Did the hospital staff talk about help you would need at home?",
    'H_DISCH_HELP_N_P': "Did the hospital staff NOT talk about help you would need at home?",
    'H_SYMPTOMS_Y_P': "Were you given information about symptoms to look out for?",
    'H_SYMPTOMS_N_P': "Were you NOT given information about symptoms to look out for?"
}

st.title("🏥 Discharge Information Rating Predictor")
st.write("Answer the following based on patient discharge experience to predict the star rating for discharge information (1–5).")

# Collect input from user
user_input = []
for feature, question in input_features.items():
    response = st.selectbox(f"{question}", list(response_options.keys()), key=feature)
    user_input.append(response_options[response])

# Predict
if st.button("Predict Hospital Rating", key="left_hospital_rating_button"):
    input_array = np.array(user_input).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    cnn_input = scaled_input.reshape((scaled_input.shape[0], scaled_input.shape[1], 1))

    prediction = model.predict(cnn_input)
    predicted_rating = np.argmax(prediction) + 1

    st.success(f"⭐ Predicted Discharge Information Star Rating: **{predicted_rating}** out of 5")

    # Show probabilities
    st.subheader("Prediction Probabilities:")
    for i, prob in enumerate(prediction[0], start=1):
        st.write(f"⭐ {i} Star: {prob:.2%}")


# << using nurse models >>

# Load trained model and scaler
model = load_model("nurse_new_cnn.keras")
with open("nurse_new_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# HCAHPS dropdown options
response_options = {
    "Always": 4,
    "Usually": 3,
    "Sometimes or Never": 1  # Combined as requested
}

# Feature labels
input_features = {
    'H_COMP_1_A_P': "Nurses always responded when needed",
    'H_COMP_1_SN_P': "Nurses sometimes/never responded when needed",
    'H_COMP_1_U_P': "Nurses usually responded when needed",
    'H_NURSE_EXPLAIN_A_P': "Nurses always explained medications",
    'H_NURSE_EXPLAIN_SN_P': "Nurses sometimes/never explained medications",
    'H_NURSE_EXPLAIN_U_P': "Nurses usually explained medications",
    'H_NURSE_LISTEN_A_P': "Nurses always listened carefully",
    'H_NURSE_LISTEN_SN_P': "Nurses sometimes/never listened carefully",
    'H_NURSE_LISTEN_U_P': "Nurses usually listened carefully",
    'H_NURSE_RESPECT_A_P': "Nurses always treated patients with respect",
    'H_NURSE_RESPECT_SN_P': "Nurses sometimes/never treated patients with respect",
    'H_NURSE_RESPECT_U_P': "Nurses usually treated patients with respect"
}

# App title
st.title("🧑‍⚕️ Nurse Rating Predictor")
st.write("Fill out the survey responses based on patient experience to predict the nurse communication star rating (1–5).")

# Collect input from user
user_input = []
for feature, label in input_features.items():
    response = st.selectbox(f"{label}:", list(response_options.keys()), key=feature)
    user_input.append(response_options[response])

# Predict rating
if st.button("Predict Nurse Rating", key = 'nurse_rating_button'):
    # Convert to array and scale
    input_array = np.array(user_input).reshape(1, -1)
    scaled_input = scaler.transform(input_array)

    # Reshape for CNN: (batch_size, timesteps, features)
    cnn_input = scaled_input.reshape((scaled_input.shape[0], scaled_input.shape[1], 1))

    # Predict
    prediction = model.predict(cnn_input)
    predicted_rating = np.argmax(prediction) + 1  # Ratings are 1-5

    st.success(f"⭐ Predicted Nurse Communication Star Rating: **{predicted_rating}** out of 5")

    # Show prediction confidence
    st.subheader("Prediction Probabilities:")
    for i, prob in enumerate(prediction[0], start=1):
        st.write(f"⭐ {i} Star: {prob:.2%}")

# << using quite models >>


# Load the trained CNN model and scaler
model = load_model("quiet_cnn.keras")
with open("quiet_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Dropdown options typical to HCAHPS surveys
response_options = {
    "Always": 4,
    "Usually": 3,
    "Sometimes/Never": 2  # Combined Sometimes and Never as one option
}

# Input features and labels
input_features = {
    'H_QUIET_HSP_A_P': "Was the hospital environment always quiet?",
    'H_QUIET_HSP_SN_P': "Was the hospital environment sometimes or never quiet?",
    'H_QUIET_HSP_U_P': "Was the hospital environment usually quiet?"
}

st.title("🤫 Hospital Quietness Star Rating Predictor")
st.write("Select the responses based on patient experience to predict the hospital quietness star rating (1 to 5 stars).")

# Collect user inputs
user_input = []
for feature, label in input_features.items():
    response = st.selectbox(label, list(response_options.keys()), key=feature)
    user_input.append(response_options[response])

# Prediction button
if st.button("Predict Quietness Star Rating", key= "quit_rating_button"):
    # Prepare input
    input_array = np.array(user_input).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    cnn_input = scaled_input.reshape((scaled_input.shape[0], scaled_input.shape[1], 1))

    # Predict
    prediction = model.predict(cnn_input)
    predicted_rating = np.argmax(prediction) + 1  # 1 to 5 star rating

    st.success(f"⭐ Predicted Quietness Star Rating: **{predicted_rating}** / 5")

    # Show prediction probabilities for each star
    st.subheader("Prediction Confidence:")
    for i, prob in enumerate(prediction[0], start=1):
        st.write(f"{i} Star: {prob:.2%}")


# << using star rating models>> 

import joblib

# Load model and scaler
model = load_model('hospital_star_rating_cnn.keras')
scaler = joblib.load('star_rating_scaler.pkl')

st.title("Hospital Star Rating Prediction (CNN Model)")

# User input widgets
st.write("Please enter survey ratings below:")

# Using sliders for numeric input between 0 and 100 (adjust range if needed)
rating_0_6 = st.slider("Percentage rating 0-6", min_value=0, max_value=100, value=20)
rating_7_8 = st.slider("Percentage rating 7-8", min_value=0, max_value=100, value=40)
rating_9_10 = st.slider("Percentage rating 9-10", min_value=0, max_value=100, value=40)

# When user clicks Predict button
if st.button("Predict Star Rating"):
    # Prepare input as numpy array
    user_input = np.array([[rating_0_6, rating_7_8, rating_9_10]], dtype=float)

    # Scale input
    scaled_input = scaler.transform(user_input)

    # Reshape for CNN input: (1 sample, 3 features, 1 channel)
    scaled_input_cnn = scaled_input.reshape((1, 3, 1))

    # Predict class probabilities
    preds = model.predict(scaled_input_cnn)
    star_rating_class = np.argmax(preds)  # 0-based class

    # Convert class to star rating (add 1)
    star_rating = star_rating_class + 1

    st.success(f"Predicted Hospital Star Rating: {star_rating} ⭐")

    st.write(f"Class probabilities: {preds[0]}")


#<< using recommend models >>



# Load model and scaler
model = load_model("recommend_cnn.keras")
with open("recommend_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Dropdown options
response_mapping = {
    "Definitely No": 1,
    "Probably No": 2,
    "Probably Yes": 3,
    "Definitely Yes": 4
}

st.title("🏥 Recommend Section: Star Rating Prediction")
st.write("Please fill out the following patient experience ratings to predict the hospital's recommendation star rating.")

# Inputs
dn = st.selectbox("Would you recommend this hospital to friends and family? (Definitely No)", list(response_mapping.keys()))
dy = st.selectbox("Would you recommend this hospital to friends and family? (Definitely Yes)", list(response_mapping.keys()))
py = st.selectbox("Would you recommend this hospital to friends and family? (Probably Yes)", list(response_mapping.keys()))

# Convert inputs to numeric
input_values = [
    response_mapping[dn],
    response_mapping[dy],
    response_mapping[py]
]

# Predict button
if st.button("Predict Star Rating", key = "recommend_rating_button"):
    # Preprocess
    input_array = np.array(input_values).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    scaled_input = scaled_input.reshape(scaled_input.shape[0], scaled_input.shape[1], 1)  # For CNN

    # Predict
    prediction = model.predict(scaled_input)
    predicted_class = np.argmax(prediction) + 1  # Adding 1 to match star rating (1–5)

    st.success(f"⭐ Predicted Hospital Recommendation Star Rating: {predicted_class}")

# <<using staff models>>

# ✅ Load model and scaler
model = load_model("staff_new_cnn.keras")
with open("staff_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ✅ Dropdown options
response_mapping = {
    "Always": 100,
    "Usually": 66,
    "Sometimes": 33,
    "Never": 0
}

# ✅ App UI
st.title("🩺 Staff Section: Star Rating Prediction")
st.write("Please rate your experience with the hospital staff's communication and medication explanations:")

# ✅ Input selections
a = st.selectbox("Staff explained things clearly (Always)", list(response_mapping.keys()))
sn = st.selectbox("Staff explained things clearly (Sometimes or Never)", list(response_mapping.keys()))
u = st.selectbox("Staff explained things clearly (Usually)", list(response_mapping.keys()))

ma = st.selectbox("Staff explained medicine purpose (Always)", list(response_mapping.keys()))
msn = st.selectbox("Staff explained medicine purpose (Sometimes or Never)", list(response_mapping.keys()))
mu = st.selectbox("Staff explained medicine purpose (Usually)", list(response_mapping.keys()))

sa = st.selectbox("Staff explained side effects (Always)", list(response_mapping.keys()))
ssn = st.selectbox("Staff explained side effects (Sometimes or Never)", list(response_mapping.keys()))
su = st.selectbox("Staff explained side effects (Usually)", list(response_mapping.keys()))

# ✅ Convert to input values
input_values = [
    response_mapping[a],
    response_mapping[sn],
    response_mapping[u],
    response_mapping[ma],
    response_mapping[msn],
    response_mapping[mu],
    response_mapping[sa],
    response_mapping[ssn],
    response_mapping[su],
]

# ✅ Predict button
if st.button("Predict Star Rating", key="staff_rating_button"):
    input_array = np.array(input_values).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    scaled_input = scaled_input.reshape(scaled_input.shape[0], scaled_input.shape[1], 1)  # CNN shape

    prediction = model.predict(scaled_input)
    predicted_class = np.argmax(prediction) + 1  # 1 to 5 stars

    st.success(f"⭐ Predicted Staff Star Rating: {predicted_class}")
