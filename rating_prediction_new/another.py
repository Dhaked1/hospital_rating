import streamlit as st
import numpy as np

def predict_rating_from_survey_ui():
    """
    Streamlit UI for predicting a patient survey rating based on a rule-based method,
    using all 29 HCAHPS questions from the provided PDF.
    Default option is pre-selected for each question.
    Questions irrelevant to direct rating are shown but not included in the average.
    """

    st.set_page_config(page_title="HCAHPS Hospital Survey Rating Predictor", layout="centered")

    st.title("🏥 HCAHPS Hospital Survey Rating Predictor (Rule-Based)")
    st.markdown("""
    Please answer the following questions about your recent hospital stay, based on the official HCAHPS survey questions.
    Each question has a default answer selected.
    Some questions are for informational purposes only and will not be included in the calculation of your predicted star rating.
    """)

    st.markdown("---")

    # Define the mapping from qualitative answers to numerical scores (1-5 scale)
    answer_to_score_map = {
        "Always": 5, "Usually": 4, "Sometimes": 2, "Never": 1,
        "Strongly agree": 5, "Agree": 4, "Disagree": 2, "Strongly disagree": 1,
        "Excellent": 5, "Very good": 4, "Good": 3, "Fair": 2, "Poor": 1,
        "Yes": 5, "No": 1,
        "Definitely yes": 5, "Probably yes": 4, "Probably no": 2, "Definitely no": 1,
        "I never pressed the call button": 3, # Neutral/Non-applicable, map to middle
        "I was not given any medication when I left the hospital": 3, # Neutral/Non-applicable
        # Direct mapping for Q18 (0-10 scale) to a 0-5 scale for calculation consistency
        **{str(i): i / 2 for i in range(11)} # 0 -> 0, 1 -> 0.5, ..., 10 -> 5
    }

    # List of all 29 questions from the HCAHPS Survey PDF
    # 'for_rating': True indicates the question contributes to the average rating.
    # 'for_rating': False indicates it's for knowledge/demographics only.
    questions = [
        {
            "id": "Q1", "text": "1. During this hospital stay, how often did nurses treat you with courtesy and respect?",
            "options": ["Always", "Usually", "Sometimes", "Never"], "for_rating": True
        },
        {
            "id": "Q2", "text": "2. During this hospital stay, how often did nurses listen carefully to you?",
            "options": ["Always", "Usually", "Sometimes", "Never"], "for_rating": True
        },
        {
            "id": "Q3", "text": "3. During this hospital stay, how often did nurses explain things in a way you could understand?",
            "options": ["Always", "Usually", "Sometimes", "Never"], "for_rating": True
        },
        {
            "id": "Q4", "text": "4. During this hospital stay, after you pressed the call button, how often did you get help as soon as you wanted it?",
            "options": ["Always", "Usually", "Sometimes", "Never", "I never pressed the call button"], "for_rating": True
        },
        {
            "id": "Q5", "text": "5. During this hospital stay, how often did doctors treat you with courtesy and respect?",
            "options": ["Always", "Usually", "Sometimes", "Never"], "for_rating": True
        },
        {
            "id": "Q6", "text": "6. During this hospital stay, how often did doctors listen carefully to you?",
            "options": ["Always", "Usually", "Sometimes", "Never"], "for_rating": True
        },
        {
            "id": "Q7", "text": "7. During this hospital stay, how often did doctors explain things in a way you could understand?",
            "options": ["Always", "Usually", "Sometimes", "Never"], "for_rating": True
        },
        {
            "id": "Q8", "text": "8. During this hospital stay, how often were your room and bathroom kept clean?",
            "options": ["Always", "Usually", "Sometimes", "Never"], "for_rating": True
        },
        {
            "id": "Q9", "text": "9. During this hospital stay, how often was the area around your room quiet at night?",
            "options": ["Always", "Usually", "Sometimes", "Never"], "for_rating": True
        },
        {
            "id": "Q10", "text": "10. During this hospital stay, did you need help from nurses or other hospital staff in getting to the bathroom or in using a bedpan?",
            "options": ["Yes", "No"], "for_rating": True
        },
        {
            "id": "Q11", "text": "11. How often did you get help in getting to the bathroom or in using a bedpan as soon as you wanted?",
            "options": ["Always", "Usually", "Sometimes", "Never"], "for_rating": True
        },
        {
            "id": "Q12", "text": "12. During this hospital stay, were you given any medicine that you had not taken before?",
            "options": ["Yes", "No"], "for_rating": True
        },
        {
            "id": "Q13", "text": "13. Before giving you any new medicine, how often did hospital staff tell you what the medicine was for?",
            "options": ["Always", "Usually", "Sometimes", "Never"], "for_rating": True
        },
        {
            "id": "Q14", "text": "14. Before giving you any new medicine, how often did hospital staff describe possible side effects in a way you could understand?",
            "options": ["Always", "Usually", "Sometimes", "Never"], "for_rating": True
        },
        {
            "id": "Q15", "text": "15. After you left the hospital, did you go directly to your own home, to someone else's home, or to another health facility?",
            "options": ["Own home", "Someone else's home", "Another health facility"], "for_rating": False
        },
        {
            "id": "Q16", "text": "16. During this hospital stay, did doctors, nurses or other hospital staff talk with you about whether you would have the help you needed when you left the hospital?",
            "options": ["Yes", "No"], "for_rating": True
        },
        {
            "id": "Q17", "text": "17. During this hospital stay, did you get information in writing about what symptoms or health problems to look out for after you left the hospital?",
            "options": ["Yes", "No"], "for_rating": True
        },
        {
            "id": "Q18", "text": "18. Using any number from 0 to 10, where 0 is the worst hospital possible and 10 is the best hospital possible, what number would you use to rate this hospital during your stay?",
            "options": [str(i) for i in range(11)], "for_rating": True
        },
        {
            "id": "Q19", "text": "19. Would you recommend this hospital to your friends and family?",
            "options": ["Definitely yes", "Probably yes", "Probably no", "Definitely no"], "for_rating": True
        },
        {
            "id": "Q20", "text": "20. During this hospital stay, staff took my preferences and those of my family or caregiver into account in deciding what my health care needs would be when I left.",
            "options": ["Strongly agree", "Agree", "Disagree", "Strongly disagree"], "for_rating": True
        },
        {
            "id": "Q21", "text": "21. When I left the hospital, I had a good understanding of the things I was responsible for in managing my health.",
            "options": ["Strongly agree", "Agree", "Disagree", "Strongly disagree"], "for_rating": True
        },
        {
            "id": "Q22", "text": "22. When I left the hospital, I clearly understood the purpose for taking each of my medications.",
            "options": ["Strongly agree", "Agree", "Disagree", "Strongly disagree", "I was not given any medication when I left the hospital"], "for_rating": True
        },
        {
            "id": "Q23", "text": "23. During this hospital stay, were you admitted to this hospital through the Emergency Room?",
            "options": ["Yes", "No"], "for_rating": False
        },
        {
            "id": "Q24", "text": "24. In general, how would you rate your overall health?",
            "options": ["Excellent", "Very good", "Good", "Fair", "Poor"], "for_rating": True
        },
        {
            "id": "Q25", "text": "25. In general, how would you rate your overall mental or emotional health?",
            "options": ["Excellent", "Very good", "Good", "Fair", "Poor"], "for_rating": True
        },
        {
            "id": "Q26", "text": "26. What is the highest grade or level of school that you have completed?",
            "options": ["8th grade or less", "Some high school, but did not graduate", "High school graduate or GED", "Some college or 2-year degree", "4-year college graduate", "More than 4-year college degree"], "for_rating": False
        },
        {
            "id": "Q27", "text": "27. Are you of Spanish, Hispanic or Latino origin or descent?",
            "options": ["No, not Spanish/Hispanic/Latino", "Yes, Puerto Rican", "Yes, Mexican, Mexican American, Chicano", "Yes, Cuban", "Yes, other Spanish/Hispanic/Latino"], "for_rating": False
        },
        {
            "id": "Q28", "text": "28. What is your race? Please choose one or more.",
            "options": ["White", "Black or African American", "Asian", "Native Hawaiian or other Pacific Islander", "American Indian or Alaska Native"], "for_rating": False
        },
        {
            "id": "Q29", "text": "29. What language do you mainly speak at home?",
            "options": ["English", "Spanish", "Chinese", "Russian", "Vietnamese", "Portuguese", "German", "Tagalog", "Arabic", "Some other language (please print):"], "for_rating": False
        }
    ]

    # Dictionary to store user's selected answers
    selected_answers = {}

    st.header("Your Hospital Experience:")
    for q in questions:
        # The first option in q['options'] will now be the default selected value.
        selected_option = st.selectbox(
            q['text'],
            options=q['options'], # Removed the "-- Select an Option --" placeholder
            key=q['id']
        )
        selected_answers[q['id']] = selected_option

    st.markdown("---")

    # Button to calculate and show the rating
    if st.button("Calculate My Predicted Rating"):
        scores_for_rating = []
        knowledge_questions_answered = 0

        for q in questions:
            answer = selected_answers.get(q['id'])
            # All questions will now have a selected_option (default or user-chosen),
            # so the check `if answer != "-- Select an Option --"` is no longer needed.
            if q['for_rating']:
                score = answer_to_score_map.get(answer)
                if score is not None:
                    scores_for_rating.append(score)
                else:
                    st.warning(f"Warning: Could not map score for '{answer}' in question '{q['text']}'. This answer will not be included in the rating calculation.")
            else:
                knowledge_questions_answered += 1 # Just count, not used in avg

        if scores_for_rating:
            final_average_rating = np.mean(scores_for_rating)
            predicted_star_rating = round(final_average_rating)
            # Ensure the star rating is between 1 and 5
            final_star_rating_clipped = np.clip(predicted_star_rating, 1, 5)

            st.subheader("Your Predicted Rating:")
            st.info(f"Based on the questions relevant for rating, your calculated average score is: **{final_average_rating:.2f}**")
            st.success(f"Your predicted patient survey star rating is: **{int(final_star_rating_clipped)} out of 5 stars**")

            st.markdown("---")
            st.subheader("How this rating was calculated:")
            st.write(f"We calculated the average score from {len(scores_for_rating)} questions that directly pertain to the quality of care and patient experience.")
            st.write("Each relevant answer was assigned a numerical score (e.g., 'Always' = 5, 'Never' = 1, '10' for Q18 = 5).")
            if knowledge_questions_answered > 0:
                st.write(f"Questions related to demographics or administrative details (you answered {knowledge_questions_answered} of these) were collected for knowledge but were not included in the rating calculation.")
            st.write("This is a simple, rule-based method, not a machine learning model.")
        else:
            # This else block might be less likely to be hit now, as questions will always have defaults,
            # unless ALL rating-relevant questions' first options somehow don't map to a score.
            st.warning("No scores could be calculated for rating. Please ensure relevant questions have valid default or selected options.")

# This ensures the Streamlit app runs only when the script is executed directly
if __name__ == "__main__":
    predict_rating_from_survey_ui()