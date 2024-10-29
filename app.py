import streamlit as st
import requests
import json

# Define the MLflow model endpoint URL
url = "http://127.0.0.1:8000/invocations"

# Set Streamlit page configuration
st.set_page_config(
    page_title="📈 Performance Index Predictor",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS styling
st.markdown(
    """
    <style>
    .main-title { font-size: 40px; color: #4CAF50; font-weight: bold; }
    .subtitle { font-size: 20px; color: #555; margin-top: -20px; }
    .stButton>button { color: white; background-color: #4CAF50; border-radius: 5px; }
    </style>
    """,
    unsafe_allow_html=True
)

# Main title and subtitle
st.markdown('<p class="main-title">📊 Performance Index Prediction App</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Estimate performance based on study habits and activities 📚✨</p>', unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("🔧 Input Features")
hours_studied = st.sidebar.slider("Hours Studied", min_value=0, max_value=24, value=5)
previous_scores = st.sidebar.slider("Previous Scores", min_value=0, max_value=100, value=75)
extracurricular_activities = st.sidebar.radio("Extracurricular Activities", options=[0, 1], index=0)
sleep_hours = st.sidebar.slider("Sleep Hours", min_value=0, max_value=24, value=8)
sample_question_papers_practiced = st.sidebar.slider("Sample Question Papers Practiced", min_value=0, max_value=50, value=10)

# Prediction button
if st.button("🎯 Predict Performance Index"):
    # Show a progress bar while predicting
    with st.spinner("Calculating Performance Index..."):
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
                st.success(f"📊 Predicted Performance Index: **{prediction}**")
            else:
                st.error(f"❌ Failed to get prediction: {response.text}")
        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")

# Footer
st.markdown(
    """
    <hr>
    <div style="text-align: center;">
        <p style="color: #666;">Built with ❤️ by [Rishabh Singh] using Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
