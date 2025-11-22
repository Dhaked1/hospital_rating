import streamlit as st
import numpy as np
import pandas as pd # Required for model input structure
import pickle # To load your pre-trained model and scaler

def predict_rating_from_survey_ui():
    """
    Streamlit UI for predicting a patient survey rating using composite scores
    calculated from HCAHPS questions and then a Random Forest model.
    Default option is pre-selected for each question.
    """

    st.set_page_config(page_title="HCAHPS Hospital Survey Rating Predictor (Model-Based)", layout="centered")

    st.title("🏥 HCAHPS Hospital Survey Rating Predictor (Model-Based)")
    st.markdown("""
    Please answer the following questions about your recent hospital stay, based on the official HCAHPS survey questions.
    Each question has a default answer selected.
    Individual category scores will be calculated and then used by a trained Random Forest model to predict the overall hospital star rating.
    """)

    st.markdown("---")

    # Define the mapping from qualitative answers to numerical scores (1-5 scale)
    answer_to_score_map = {
        "Always": 5, "Usually": 4, "Sometimes": 2, "Never": 1,
        "Strongly agree": 5, "Agree": 4, "Disagree": 2, "Strongly disagree": 1,
        "Excellent": 5, "Very good": 4, "Good": 3, "Fair": 2, "Poor": 1,
        "Yes": 5, "No": 1,
        "Definitely yes": 5, "Probably yes": 4, "Probably no": 2, "Definitely no": 1,
        "I never pressed the call button": 3, # Neutral/Non-applicable for Q4
        "I was not given any medication when I left the hospital": 3, # Neutral/Non-applicable for Q22
        # Direct mapping for Q18 (0-10 scale) to a 0-5 scale for calculation consistency
        **{str(i): i / 2 for i in range(11)} # 0 -> 0, 1 -> 0.5, ..., 10 -> 5
    }

    # List of all 29 questions from the HCAHPS Survey PDF
    # 'for_rating': True indicates the question contributes to a rating used by the model.
    questions = [
        {"id": "Q1", "text": "1. During this hospital stay, how often did nurses treat you with courtesy and respect?", "options": ["Always", "Usually", "Sometimes", "Never"], "category": "Nurse Communication"},
        {"id": "Q2", "text": "2. During this hospital stay, how often did nurses listen carefully to you?", "options": ["Always", "Usually", "Sometimes", "Never"], "category": "Nurse Communication"},
        {"id": "Q3", "text": "3. During this hospital stay, how often did nurses explain things in a way you could understand?", "options": ["Always", "Usually", "Sometimes", "Never"], "category": "Nurse Communication"},
        {"id": "Q4", "text": "4. During this hospital stay, after you pressed the call button, how often did you get help as soon as you wanted it?", "options": ["Always", "Usually", "Sometimes", "Never", "I never pressed the call button"], "category": "Nurse Communication"}, # Reverted category as Q4 is now with Nurse Communication
        {"id": "Q5", "text": "5. During this hospital stay, how often did doctors treat you with courtesy and respect?", "options": ["Always", "Usually", "Sometimes", "Never"], "category": "Doctor Communication"},
        {"id": "Q6", "text": "6. During this hospital stay, how often did doctors listen carefully to you?", "options": ["Always", "Usually", "Sometimes", "Never"], "category": "Doctor Communication"},
        {"id": "Q7", "text": "7. During this hospital stay, how often did doctors explain things in a way you could understand?", "options": ["Always", "Usually", "Sometimes", "Never"], "category": "Doctor Communication"},
        {"id": "Q8", "text": "8. During this hospital stay, how often were your room and bathroom kept clean?", "options": ["Always", "Usually", "Sometimes", "Never"], "category": "Cleanliness"},
        {"id": "Q9", "text": "9. During this hospital stay, how often was the area around your room quiet at night?", "options": ["Always", "Usually", "Sometimes", "Never"], "category": "Quietness"},
        {"id": "Q10", "text": "10. During this hospital stay, did you need help from nurses or other hospital staff in getting to the bathroom or in using a bedpan?", "options": ["Yes", "No"], "category": "Responsiveness of Staff (Pre-condition for Q11)"},
        {"id": "Q11", "text": "11. How often did you get help in getting to the bathroom or in using a bedpan as soon as you wanted?", "options": ["Always", "Usually", "Sometimes", "Never"], "category": "Responsiveness of Staff"},
        {"id": "Q12", "text": "12. During this hospital stay, were you given any medicine that you had not taken before?", "options": ["Yes", "No"], "category": "Communication about Medicines (Pre-condition for Q13, Q14)"},
        {"id": "Q13", "text": "13. Before giving you any new medicine, how often did hospital staff tell you what the medicine was for?", "options": ["Always", "Usually", "Sometimes", "Never"], "category": "Communication about Medicines"},
        {"id": "Q14", "text": "14. Before giving you any new medicine, how often did hospital staff describe possible side effects in a way you could understand?", "options": ["Always", "Usually", "Sometimes", "Never"], "category": "Communication about Medicines"},
        {"id": "Q15", "text": "15. After you left the hospital, did you go directly to your own home, to someone else's home, or to another health facility?", "options": ["Own home", "Someone else's home", "Another health facility"], "category": "Demographics/Discharge"},
        {"id": "Q16", "text": "16. During this hospital stay, did doctors, nurses or other hospital staff talk with you about whether you would have the help you needed when you left the hospital?", "options": ["Yes", "No"], "category": "Care Transition"},
        {"id": "Q17", "text": "17. During this hospital stay, did you get information in writing about what symptoms or health problems to look out for after you left the hospital?", "options": ["Yes", "No"], "category": "Care Transition"},
        {"id": "Q18", "text": "18. Using any number from 0 to 10, where 0 is the worst hospital possible and 10 is the best hospital possible, what number would you use to rate this hospital during your stay?", "options": [str(i) for i in range(11)], "category": "Overall Hospital Rating"},
        {"id": "Q19", "text": "19. Would you recommend this hospital to your friends and family?", "options": ["Definitely yes", "Probably yes", "Probably no", "Definitely no"], "category": "Recommendation"},
        {"id": "Q20", "text": "20. During this hospital stay, staff took my preferences and those of my family or caregiver into account in deciding what my health care needs would be when I left.", "options": ["Strongly agree", "Agree", "Disagree", "Strongly disagree"], "category": "Care Transition"},
        {"id": "Q21", "text": "21. When I left the hospital, I had a good understanding of the things I was responsible for in managing my health.", "options": ["Strongly agree", "Agree", "Disagree", "Strongly disagree"], "category": "Care Transition"},
        {"id": "Q22", "text": "22. When I left the hospital, I clearly understood the purpose for taking each of my medications.", "options": ["Strongly agree", "Agree", "Disagree", "Strongly disagree", "I was not given any medication when I left the hospital"], "category": "Care Transition"},
        {"id": "Q23", "text": "23. During this hospital stay, were you admitted to this hospital through the Emergency Room?", "options": ["Yes", "No"], "category": "Demographics/Admission"},
        {"id": "Q24", "text": "24. In general, how would you rate your overall health?", "options": ["Excellent", "Very good", "Good", "Fair", "Poor"], "category": "Demographics/Health"},
        {"id": "Q25", "text": "25. In general, how would you rate your overall mental or emotional health?", "options": ["Excellent", "Very good", "Good", "Fair", "Poor"], "category": "Demographics/Health"},
        {"id": "Q26", "text": "26. What is the highest grade or level of school that you have completed?", "options": ["8th grade or less", "Some high school, but did not graduate", "High school graduate or GED", "Some college or 2-year degree", "4-year college graduate", "More than 4-year college degree"], "category": "Demographics/Education"},
        {"id": "Q27", "text": "27. Are you of Spanish, Hispanic or Latino origin or descent?", "options": ["No, not Spanish/Hispanic/Latino", "Yes, Puerto Rican", "Yes, Mexican, Mexican American, Chicano", "Yes, Cuban", "Yes, other Spanish/Hispanic/Latino"], "category": "Demographics/Ethnicity"},
        {"id": "Q28", "text": "28. What is your race? Please choose one or more.", "options": ["White", "Black or African American", "Asian", "Native Hawaiian or other Pacific Islander", "American Indian or Alaska Native"], "category": "Demographics/Race"},
        {"id": "Q29", "text": "29. What language do you mainly speak at home?", "options": ["English", "Spanish", "Chinese", "Russian", "Vietnamese", "Portuguese", "German", "Tagalog", "Arabic", "Some other language (please print):"], "category": "Demographics/Language"}
    ]

    # Dictionary to store user's selected answers (score for each question)
    selected_question_scores = {}
    # Dictionary to store the raw string answers for skip logic
    selected_answers = {}

    st.header("Your Hospital Experience:")
    for q in questions:
        selected_option = st.selectbox(
            q['text'],
            options=q['options'],
            key=q['id']
        )
        selected_answers[q['id']] = selected_option # Store the raw answer
        score = answer_to_score_map.get(selected_option)
        selected_question_scores[q['id']] = score if score is not None else np.nan

    st.markdown("---")

    # Load the pre-trained model and scaler
    try:
        with open('combined_rf_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        with open('combined_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        st.success("Machine learning model and scaler loaded successfully!")
    except FileNotFoundError:
        st.error("Error: Could not find 'combined_rf_model.pkl' or 'combined_scaler.pkl'. Please ensure they are in the same directory as this script.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        st.stop()

    # Define input columns for the Random Forest model (ORDER IS CRITICAL)
    input_columns = [
        'H_COMP_7_STAR_RATING',     # Care Transitions
        'H_CLEAN_STAR_RATING',      # Cleanliness
        'H_COMP_2_STAR_RATING',     # Doctor Communication (as per your instruction)
        'H_COMP_6_STAR_RATING',     # Communication about Medicines
        'H_COMP_1_STAR_RATING',     # Nurse Communication
        'H_QUIET_STAR_RATING',      # Quietness
        'H_HSP_RATING_STAR_RATING', # Overall Hospital Rating (direct Q18)
        'H_RECMND_STAR_RATING'      # Recommendation
    ]

    # Define mapping from HCAHPS composite to survey questions
    composite_question_map = {
        'H_COMP_1_STAR_RATING': ['Q1', 'Q2', 'Q3', 'Q4'], # Nurse Communication (Q4 is back here as per your request)
        'H_COMP_2_STAR_RATING': ['Q5','Q6','Q7'], # Doctor Communication (as per your instruction)
        'H_COMP_6_STAR_RATING': ['Q13', 'Q14'], # Q12 is a skip for Q13, Q14
        'H_CLEAN_STAR_RATING': ['Q8'], # Cleanliness
        'H_QUIET_STAR_RATING': ['Q9'], # Quietness
        'H_HSP_RATING_STAR_RATING': ['Q18'],
        'H_RECMND_STAR_RATING': ['Q19'],
        'H_COMP_7_STAR_RATING': ['Q16', 'Q17', 'Q20', 'Q21', 'Q22'],
    }

    if st.button("Predict Overall Star Rating"):
        calculated_composites = {}

        # Calculate composite scores
        for composite_name, q_ids in composite_question_map.items():
            scores_for_composite = []
            
            # Special handling for H_COMP_2_STAR_RATING (Doctor Communication)
            if composite_name == 'H_COMP_2_STAR_RATING':
                for q_id in q_ids: # q_ids will be ['Q5', 'Q6', 'Q7']
                    score = selected_question_scores.get(q_id)
                    if not np.isnan(score):
                        scores_for_composite.append(score)
                if scores_for_composite:
                    calculated_composites[composite_name] = np.mean(scores_for_composite)
                else:
                    calculated_composites[composite_name] = 3.0 # Neutral if no valid scores
                    st.warning(f"No valid scores found for {composite_name}. Assigning a neutral score of 3.0.")
                continue # Move to next composite

            # Handle skip logic for Q13, Q14 (Communication about Medicines)
            elif composite_name == 'H_COMP_6_STAR_RATING':
                q12_answer = selected_answers.get('Q12')
                if q12_answer == 'No':
                    calculated_composites[composite_name] = 3.0
                    st.warning(f"Q12 was 'No', so {composite_name} assigned a neutral score of 3.0.")
                    continue
                else:
                    for q_id in q_ids:
                        score = selected_question_scores.get(q_id)
                        if not np.isnan(score):
                            scores_for_composite.append(score)
                
                if scores_for_composite:
                    calculated_composites[composite_name] = np.mean(scores_for_composite)
                else:
                    calculated_composites[composite_name] = 3.0
                    st.warning(f"No valid scores found for {composite_name}. Assigning a neutral score of 3.0.")
                continue
            
            # For all other composites without special skip logic
            for q_id in q_ids:
                score = selected_question_scores.get(q_id)
                if not np.isnan(score):
                    scores_for_composite.append(score)

            if scores_for_composite:
                calculated_composites[composite_name] = np.mean(scores_for_composite)
            else:
                calculated_composites[composite_name] = 3.0
                st.warning(f"No valid scores found for {composite_name}. Assigning a neutral score of 3.0.")

        st.subheader("Calculated Category Ratings (Input for Model):")
        for comp, val in calculated_composites.items():
            st.write(f"- **{comp}:** {val:.2f}")
        st.markdown("---")

        # Prepare input for the Random Forest model
        model_input_features = []
        for col in input_columns:
            # Ensure order matches input_columns, use 3.0 as fallback if something goes wrong
            model_input_features.append(calculated_composites.get(col, 3.0))

        # Reshape for single prediction and scale
        X_new = np.array(model_input_features).reshape(1, -1)
        X_new_scaled = scaler.transform(X_new)

        # Predict the overall rating
        predicted_overall_rating_0_based = rf_model.predict(X_new_scaled)[0]
        # Convert back to 1-5 star rating
        predicted_overall_rating = int(predicted_overall_rating_0_based) + 1

        st.subheader("Predicted Overall Hospital Rating:")
        st.success(f"Based on your answers and the Random Forest model, the predicted overall hospital star rating is: **{predicted_overall_rating} out of 5 stars**")
# This ensures the Streamlit app runs only when the script is executed directly
if __name__ == "__main__":
    predict_rating_from_survey_ui()