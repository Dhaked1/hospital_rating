import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
# Load Random Forest model and scaler
with open('combined_rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
# Function to load CNN model and scaler and predict
def predict_cnn_class(model_path, scaler_path, input_values):
    model = load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    scaled_input = scaler.transform(np.array(input_values).reshape(1, -1))
    scaled_input = scaled_input.reshape(scaled_input.shape[0], scaled_input.shape[1], 1)
    prediction = model.predict(scaled_input)
    return np.argmax(prediction) + 1  # Class from 1 to 5

st.title("🏥 Final Hospital Star Rating Prediction")

# Collect inputs for each model
response_mapping = {
    "Always": 4,
    "Usually": 3,
    "Sometimes/Never": 2
}

st.subheader("Staff Communication (Medication and Side Effects)")

staff_input = [
    response_mapping[st.selectbox("How often did staff explain what the medicine was for? (H_COMP_5_A_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("How often did staff speak clearly when explaining meds? (H_COMP_5_SN_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("How often did staff help you understand your medicine? (H_COMP_5_U_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Staff explained new medications clearly? (H_MED_FOR_A_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Staff listened about medication concerns? (H_MED_FOR_SN_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Staff checked if you understood side effects? (H_MED_FOR_U_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Side effects explained clearly? (H_SIDE_EFFECTS_A_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Side effects discussed respectfully? (H_SIDE_EFFECTS_SN_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Side effects discussion understandable? (H_SIDE_EFFECTS_U_P)", list(response_mapping.keys()))]
]
  
# 👨‍⚕️ Doctor Input
st.subheader("Doctor Communication")
doctor_input = [
    response_mapping[st.selectbox("How often did doctors explain things clearly? (H_COMP_2_A_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Did doctors speak clearly when explaining care? (H_COMP_2_SN_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("How often did doctors help you understand your care? (H_COMP_2_U_P)", list(response_mapping.keys()))],
    
    response_mapping[st.selectbox("Doctors explained your health concerns clearly? (H_DOCTOR_EXPLAIN_A_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Doctors explained your diagnosis understandably? (H_DOCTOR_EXPLAIN_SN_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Doctors helped you understand your condition well? (H_DOCTOR_EXPLAIN_U_P)", list(response_mapping.keys()))],
    
    response_mapping[st.selectbox("Doctors listened carefully to you? (H_DOCTOR_LISTEN_A_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Doctors were attentive when you spoke? (H_DOCTOR_LISTEN_SN_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Doctors made sure they understood your concerns? (H_DOCTOR_LISTEN_U_P)", list(response_mapping.keys()))],
    
    response_mapping[st.selectbox("Doctors treated you with courtesy and respect? (H_DOCTOR_RESPECT_A_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Doctors respected your opinions and preferences? (H_DOCTOR_RESPECT_SN_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Doctors behaved respectfully during your care? (H_DOCTOR_RESPECT_U_P)", list(response_mapping.keys()))]
]

# 👩‍⚕️ Nurse Communication
st.subheader("Nurse Communication")
nurse_input = [
    response_mapping[st.selectbox("How often did nurses explain things clearly? (H_COMP_1_A_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Did nurses speak clearly when explaining care? (H_COMP_1_SN_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("How often did nurses help you understand your care? (H_COMP_1_U_P)", list(response_mapping.keys()))],
    
    response_mapping[st.selectbox("Nurses explained your health concerns clearly? (H_NURSE_EXPLAIN_A_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Nurses explained your diagnosis understandably? (H_NURSE_EXPLAIN_SN_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Nurses helped you understand your condition well? (H_NURSE_EXPLAIN_U_P)", list(response_mapping.keys()))],
    
    response_mapping[st.selectbox("Nurses listened carefully to you? (H_NURSE_LISTEN_A_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Nurses were attentive when you spoke? (H_NURSE_LISTEN_SN_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Nurses made sure they understood your concerns? (H_NURSE_LISTEN_U_P)", list(response_mapping.keys()))],
    
    response_mapping[st.selectbox("Nurses treated you with courtesy and respect? (H_NURSE_RESPECT_A_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Nurses respected your opinions and preferences? (H_NURSE_RESPECT_SN_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Nurses behaved respectfully during your care? (H_NURSE_RESPECT_U_P)", list(response_mapping.keys()))]
]

st.subheader("Care Transitions")
care_input = [
    response_mapping[st.selectbox("Staff considered your preferences in discharge planning? (H_COMP_7_A)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Staff discussed discharge needs in detail? (H_COMP_7_D_SD)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Staff clearly explained discharge plans? (H_COMP_7_SA)", list(response_mapping.keys()))],

    response_mapping[st.selectbox("You got help managing your medications after discharge? (H_CT_MED_A)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Medication instructions were discussed clearly? (H_CT_MED_D_SD)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Staff explained how to take new meds safely? (H_CT_MED_SA)", list(response_mapping.keys()))],

    response_mapping[st.selectbox("Staff asked about your care preferences after discharge? (H_CT_PREFER_A)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Preferences were discussed in discharge talks? (H_CT_PREFER_D_SD)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("You felt involved in post-discharge decisions? (H_CT_PREFER_SA)", list(response_mapping.keys()))],

    response_mapping[st.selectbox("You understood the purpose of each medication? (H_CT_UNDER_A)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Instructions were clear and complete? (H_CT_UNDER_D_SD)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("You felt prepared to manage your care? (H_CT_UNDER_SA)", list(response_mapping.keys()))]
]


st.subheader("Cleanliness")
clean_input = [
    response_mapping[st.selectbox("How often was the hospital environment clean? (H_CLEAN_HSP_A_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("How often was the hospital environment sometimes clean? (H_CLEAN_HSP_SN_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("How often was the hospital environment not clean? (H_CLEAN_HSP_U_P)", list(response_mapping.keys()))]
]


st.subheader("Helpfulness of Staff")
help_input = [
    response_mapping[st.selectbox("How often did staff help you with bathing? (H_BATH_HELP_A_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("How often was help with bathing sometimes provided? (H_BATH_HELP_SN_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("How often was help with bathing not provided? (H_BATH_HELP_U_P)", list(response_mapping.keys()))],

    response_mapping[st.selectbox("How often did staff respond quickly when you used the call button? (H_CALL_BUTTON_A_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("How often was staff response to call button sometimes quick? (H_CALL_BUTTON_SN_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("How often was staff response to call button not quick? (H_CALL_BUTTON_U_P)", list(response_mapping.keys()))],

    response_mapping[st.selectbox("How often did staff explain things clearly? (H_COMP_3_A_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("How often was staff explanation sometimes clear? (H_COMP_3_SN_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("How often was staff explanation unclear? (H_COMP_3_U_P)", list(response_mapping.keys()))]
]

st.subheader("Left Hospital")
left_input = [
    response_mapping[st.selectbox("Did you receive help after discharge? (No) (H_DISCH_HELP_N_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Did you receive help after discharge? (Yes) (H_DISCH_HELP_Y_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Were your symptoms explained clearly? (No) (H_SYMPTOMS_N_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Were your symptoms explained clearly? (Yes) (H_SYMPTOMS_Y_P)", list(response_mapping.keys()))]
]


st.subheader("Quietness of Hospital")
quite_input = [
    response_mapping[st.selectbox("How often was the area around your room quiet at night? (Always) (H_QUIET_HSP_A_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("How often was the area around your room quiet at night? (Sometimes) (H_QUIET_HSP_SN_P)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("How often was the area around your room quiet at night? (Never) (H_QUIET_HSP_U_P)", list(response_mapping.keys()))]
]


st.subheader("Hospital Star Rating (0 to 10 scale)")
rate_input = [
    response_mapping[st.selectbox("Rate the hospital 0–6 (H_HSP_RATING_0_6)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Rate the hospital 7–8 (H_HSP_RATING_7_8)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Rate the hospital 9–10 (H_HSP_RATING_9_10)", list(response_mapping.keys()))]
]


st.subheader("Hospital Recommendation")
recommend_input = [
    response_mapping[st.selectbox("Would you recommend the hospital to others? (Definitely No) (H_RECMND_DN)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Would you recommend the hospital to others? (Probably Yes) (H_RECMND_DY)", list(response_mapping.keys()))],
    response_mapping[st.selectbox("Would you recommend the hospital to others? (Definitely Yes) (H_RECMND_PY)", list(response_mapping.keys()))]
]


if st.button("🔍 Predict Final Hospital Star Rating"):
    # Predict class from each CNN
    staff_class = predict_cnn_class("staff_new_cnn.keras", "staff_scaler.pkl", staff_input)
    doctor_class = predict_cnn_class("doctor_new_cnn.keras", "doctor_new_scaler.pkl", doctor_input)
    nurse_class = predict_cnn_class("nurse_new_cnn.keras", "nurse_new_scaler.pkl", nurse_input)
    care_class = predict_cnn_class("care_cnn.keras", "care_scaler.pkl", care_input)
    clean_class = predict_cnn_class("clean_cnn.keras", "clean_scaler.pkl", clean_input)
    help_class = predict_cnn_class("help_new_cnn.keras", "help_new_scaler.pkl", help_input)
    left_class = predict_cnn_class("left_hospital_cnn.keras", "left_hospital_scaler.pkl", left_input)
    quite_class = predict_cnn_class("quiet_cnn.keras", "quiet_scaler.pkl", quite_input)
    rate_class = predict_cnn_class("hospital_star_rating_cnn.keras", "star_rating_scaler.pkl", rate_input)
    recommend_class = predict_cnn_class("recommend_cnn.keras", "recommend_scaler.pkl", recommend_input)
    print(staff_class, doctor_class, nurse_class, care_class, clean_class,
          help_class, left_class, quite_class, rate_class, recommend_class)
    # Prepare RF input
    rf_input = np.array([
        staff_class, doctor_class, nurse_class, care_class, clean_class,
        help_class, left_class, quite_class, rate_class, recommend_class
    ]).reshape(1, -1)
    print(rf_input)
    # Predict final star rating
    final_star = rf_model.predict(rf_input)[0] + 1  # Add 1 to restore 1–5 scale

    st.success(f"🏥 Final Predicted H_STAR_RATING: {final_star} ⭐")