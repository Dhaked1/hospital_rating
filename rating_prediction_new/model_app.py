import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Hospital Star Rating Predictor", page_icon="🏥")

# Load Random Forest model
with open('combined_rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Mapping
response_mapping = {
    "Always": 4,
    "Usually": 3,
    "Sometimes/Never": 2
}

# Load CNN model and make prediction
def predict_cnn_class(model_path, scaler_path, input_values):
    model = load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    scaled_input = scaler.transform(np.array(input_values).reshape(1, -1))
    scaled_input = scaled_input.reshape(scaled_input.shape[0], scaled_input.shape[1], 1)
    prediction = model.predict(scaled_input)
    return np.argmax(prediction) + 1  # Convert to 1-5 stars

# Helper function to collect input
def get_input_section(title, questions):
    with st.expander(title):
        return [response_mapping[st.selectbox(q, list(response_mapping.keys()), key=q)] for q in questions]

st.title("🏥 Final Hospital Star Rating Prediction")
st.write("Please answer the following questions based on your hospital experience.")

# Section Inputs
staff_input = get_input_section("💊 Staff Communication", [
    "Did staff explain what your medicine was for?",
    "Did staff speak clearly about medications?",
    "Did staff help you understand your medication?",
    "Did staff explain new medications clearly?",
    "Did staff listen to your concerns about meds?",
    "Did staff check your understanding of side effects?",
    "Were side effects explained clearly?",
    "Were side effects discussed respectfully?",
    "Was the discussion about side effects easy to understand?"
])

doctor_input = get_input_section("🩺 Doctor Communication", [
    "Did doctors explain things clearly?",
    "Did doctors speak clearly when explaining care?",
    "Did doctors help you understand your care?",
    "Did doctors explain your health concerns clearly?",
    "Did doctors explain your diagnosis understandably?",
    "Did doctors help you understand your condition?",
    "Did doctors listen carefully to you?",
    "Were doctors attentive when you spoke?",
    "Did doctors make sure they understood your concerns?",
    "Did doctors treat you with respect?",
    "Did doctors respect your preferences?",
    "Were doctors courteous during your care?"
])

nurse_input = get_input_section("👩‍⚕️ Nurse Communication", [
    "Did nurses explain things clearly?",
    "Did nurses speak clearly when explaining care?",
    "Did nurses help you understand your care?",
    "Did nurses explain your health concerns clearly?",
    "Did nurses explain your diagnosis understandably?",
    "Did nurses help you understand your condition?",
    "Did nurses listen carefully to you?",
    "Were nurses attentive when you spoke?",
    "Did nurses make sure they understood your concerns?",
    "Did nurses treat you with respect?",
    "Did nurses respect your preferences?",
    "Were nurses courteous during your care?"
])

care_input = get_input_section("🔄 Care Transitions", [
    "Did staff consider your preferences when planning discharge?",
    "Were your discharge needs discussed in detail?",
    "Did staff explain your discharge plan clearly?",
    "Did you get help managing your meds after discharge?",
    "Were medication instructions discussed clearly?",
    "Were you shown how to take new meds safely?",
    "Did staff ask about your care preferences post-discharge?",
    "Were preferences considered during discharge talks?",
    "Were you involved in post-discharge decisions?",
    "Did you understand the purpose of each medication?",
    "Were instructions clear and complete?",
    "Did you feel prepared to manage your care?"
])

clean_input = get_input_section("🧼 Cleanliness", [
    "Was the hospital environment always clean?",
    "Was the hospital environment sometimes clean?",
    "Was the hospital environment not clean?"
])

help_input = get_input_section("🛎️ Staff Helpfulness", [
    "Did staff always help you with bathing?",
    "Did staff sometimes help you with bathing?",
    "Did staff not help with bathing?",
    "Did staff respond quickly to call button?",
    "Was staff response sometimes quick?",
    "Was staff response not quick?",
    "Did staff explain things clearly?",
    "Was staff explanation sometimes clear?",
    "Was staff explanation unclear?"
])

left_input = get_input_section("🚪 After Hospital Stay", [
    "Did you receive help after discharge? (No)",
    "Did you receive help after discharge? (Yes)",
    "Were your symptoms explained clearly? (No)",
    "Were your symptoms explained clearly? (Yes)"
])

quiet_input = get_input_section("🔕 Quietness of Hospital", [
    "Was the hospital area quiet at night? (Always)",
    "Was it quiet sometimes?",
    "Was it never quiet?"
])

rate_input = get_input_section("⭐ Hospital Overall Rating (0-10)", [
    "Would you rate the hospital 0–6?",
    "Would you rate the hospital 7–8?",
    "Would you rate the hospital 9–10?"
])

recommend_input = get_input_section("📣 Hospital Recommendation", [
    "Would you recommend the hospital? (Definitely No)",
    "Would you recommend the hospital? (Probably Yes)",
    "Would you recommend the hospital? (Definitely Yes)"
])

if st.button("🔍 Predict Final Hospital Star Rating"):
    with st.spinner("Analyzing responses and predicting rating..."):

        # Individual CNN predictions
        staff_class = predict_cnn_class("staff_new_cnn.keras", "staff_scaler.pkl", staff_input)
        doctor_class = predict_cnn_class("doctor_new_cnn.keras", "doctor_new_scaler.pkl", doctor_input)
        nurse_class = predict_cnn_class("nurse_new_cnn.keras", "nurse_new_scaler.pkl", nurse_input)
        care_class = predict_cnn_class("care_cnn.keras", "care_scaler.pkl", care_input)
        clean_class = predict_cnn_class("clean_cnn.keras", "clean_scaler.pkl", clean_input)
        help_class = predict_cnn_class("help_new_cnn.keras", "help_new_scaler.pkl", help_input)
        left_class = predict_cnn_class("left_hospital_cnn.keras", "left_hospital_scaler.pkl", left_input)
        quiet_class = predict_cnn_class("quiet_cnn.keras", "quiet_scaler.pkl", quiet_input)
        rate_class = predict_cnn_class("hospital_star_rating_cnn.keras", "star_rating_scaler.pkl", rate_input)
        recommend_class = predict_cnn_class("recommend_cnn.keras", "recommend_scaler.pkl", recommend_input)

        # Combine CNN outputs
        rf_input = np.array([
            staff_class, doctor_class, nurse_class, care_class, clean_class,
            help_class, left_class, quiet_class, rate_class, recommend_class
        ]).reshape(1, -1)

        # Final prediction using RF
        final_star = rf_model.predict(rf_input)[0] + 1

    # Output
    st.success(f"🏥 Final Predicted Star Rating: **{final_star}** ⭐")
    st.markdown("⭐" * final_star)

    st.subheader("📊 Model Predictions Breakdown")
    st.write(f"💊 Staff Communication: {staff_class} ⭐")
    st.write(f"🩺 Doctor Communication: {doctor_class} ⭐")
    st.write(f"👩‍⚕️ Nurse Communication: {nurse_class} ⭐")
    st.write(f"🔄 Care Transitions: {care_class} ⭐")
    st.write(f"🧼 Cleanliness: {clean_class} ⭐")
    st.write(f"🛎️ Staff Helpfulness: {help_class} ⭐")
    st.write(f"🚪 After Hospital Stay: {left_class} ⭐")
    st.write(f"🔕 Quietness: {quiet_class} ⭐")
    st.write(f"⭐ Overall Rating (0–10): {rate_class} ⭐")
    st.write(f"📣 Recommendation: {recommend_class} ⭐")
