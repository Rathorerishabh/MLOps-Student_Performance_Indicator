import streamlit as st
import requests
import json

# Define the MLflow model endpoint URL
url = "http://127.0.0.1:8000/invocations"

# Streamlit app title
st.title("Performance Index Prediction")

# Collect input values for each feature from the user
hours_studied = st.number_input("Hours Studied", min_value=0, max_value=24, step=1)
previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, step=1)
extracurricular_activities = st.number_input("Extracurricular Activities", min_value=0, max_value=1, step=1)
sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=24, step=1)
sample_question_papers_practiced = st.number_input("Sample Question Papers Practiced", min_value=0, max_value=100, step=1)

# Prediction button
if st.button("Predict Performance Index"):
    # Prepare the data in the correct format
    data = {
        "instances": [
            {
                "Hours Studied": hours_studied,
                "Previous Scores": previous_scores,
                "Extracurricular Activities": extracurricular_activities,
                "Sleep Hours": sleep_hours,
                "Sample Question Papers Practiced": sample_question_papers_practiced
            }
        ]
    }
    
    # Send POST request to the model's prediction endpoint
    try:
        response = requests.post(url, data=json.dumps(data), headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            prediction = response.json()  # Extract prediction result
            st.success(f"Predicted Performance Index: {prediction}")
        else:
            st.error(f"Failed to get prediction: {response.text}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
