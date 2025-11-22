''' streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load models aimportnd scalers
care_model = load_model("care.keras")
with open("care_scaler.pkl", "rb") as f:
    care_scaler = pickle.load(f)

clean_model = load_model("clean.keras")
with open("clean_scaler.pkl", "rb") as f:
    clean_scaler = pickle.load(f)

# Standard HCAHPS response mapping
response_options = {
    "Always": 4,
    "Usually": 3,
    "Sometimes": 2,
    "Never": 1
}

# --- Care-related input fields ---
care_inputs = {
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

# --- Cleanliness-related input fields ---
clean_inputs = {
    'H_CLEAN_HSP_A_P': "Hospital room/bathroom always clean",
    'H_CLEAN_HSP_SN_P': "Cleanliness - sometimes/never",
    'H_CLEAN_HSP_U_P': "Cleanliness - usually"
}

st.title("🏥 HCAHPS Hospital Star Rating Predictor")
st.write("### Fill the survey questions below to get predicted star ratings.")

# --- CARE FORM ---
st.header("🩺 Care Experience")
care_values = []
for key, label in care_inputs.items():
    val = st.selectbox(f"{label}:", list(response_options.keys()), key=key)
    care_values.append(response_options[val])

# --- CLEANLINESS FORM ---
st.header("🧼 Hospital Cleanliness")
clean_values = []
for key, label in clean_inputs.items():
    val = st.selectbox(f"{label}:", list(response_options.keys()), key=key)
    clean_values.append(response_options[val])

# --- PREDICT ---
if st.button("🔍 Predict Ratings"):
    # --- Care Prediction ---
    care_array = np.array(care_values).reshape(1, -1)
    care_scaled = care_scaler.transform(care_array)
    care_pred = care_model.predict(care_scaled)
    care_star = np.argmax(care_pred) + 1

    # --- Cleanliness Prediction ---
    clean_array = np.array(clean_values).reshape(1, -1)
    clean_scaled = clean_scaler.transform(clean_array)
    clean_pred = clean_model.predict(clean_scaled)
    clean_star = np.argmax(clean_pred) + 1

    # --- Display Results ---
    st.success(f"🩺 Predicted **Care-Related Rating**: ⭐ {care_star} / 5")
    st.success(f"🧼 Predicted **Cleanliness Rating**: ⭐ {clean_star} / 5")

    # Optional: Confidence breakdown
    st.subheader("📊 Confidence Breakdown")
    st.write("**Care Rating Probabilities:**")
    for i, prob in enumerate(care_pred[0], start=1):
        st.write(f"⭐ {i} Star: {prob:.2%}")
    st.write("**Cleanliness Rating Probabilities:**")
    for i, prob in enumerate(clean_pred[0], start=1):
        st.write(f"⭐ {i} Star: {prob:.2%}")'''



## nurse model 
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("nurse_new.keras")
with open("nurse_new_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# HCAHPS dropdown options
response_options = {
    "Always": 4,
    "Usually": 3,
    "Sometimes": 2,
    "Never": 1
}

# Nurse-related features and labels
input_features = {
    'H_NURSE_LISTEN_A_P': "Nurses listened carefully (Always)",
    'H_NURSE_LISTEN_U_P': "Nurses listened carefully (Usually)",
    'H_NURSE_LISTEN_SN_P': "Nurses listened carefully (Sometimes/Never)",
    'H_NURSE_RESPECT_A_P': "Nurses treated with respect (Always)",
    'H_NURSE_RESPECT_U_P': "Nurses treated with respect (Usually)",
    'H_NURSE_RESPECT_SN_P': "Nurses treated with respect (Sometimes/Never)",
    'H_NURSE_EXPLAIN_A_P': "Nurses explained clearly (Always)",
    'H_NURSE_EXPLAIN_U_P': "Nurses explained clearly (Usually)",
    'H_NURSE_EXPLAIN_SN_P': "Nurses explained clearly (Sometimes/Never)",
    'H_RESPONDED_A_P': "Help response timely (Always)",
    'H_RESPONDED_U_P': "Help response timely (Usually)",
    'H_RESPONDED_SN_P': "Help response timely (Sometimes/Never)"
}

st.title(" Nurse Communication Star Rating Predictor")
st.write("Please complete the nurse-related questions below to estimate the hospital's star rating for nurse communication.")

# Collect user input
user_input = []
for feature, label in input_features.items():
    selected_option = st.selectbox(f"{label}:", list(response_options.keys()), key=feature)
    user_input.append(response_options[selected_option])

# Predict button
if st.button("🔍 Predict Rating"):
    input_array = np.array(user_input).reshape(1, -1)
    scaled_input = scaler.transform(input_array)

    # Predict
    prediction = model.predict(scaled_input)
    predicted_rating = np.argmax(prediction) + 1

    st.success(f"⭐ Predicted Nurse Communication Rating: **{predicted_rating}** out of 5")

    st.subheader("📊 Prediction Probabilities:")
    for i, prob in enumerate(prediction[0], start=1):
        st.write(f"⭐ {i} Star: {prob:.2%}")
