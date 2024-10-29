# 📈 Performance Index Prediction App

### An Interactive Web App to Predict Performance Index Based on Study Habits Using MLFlow 📚✨

This application predicts a student’s *Performance Index* based on key study and lifestyle habits using a trained ML model hosted on an MLflow server. The app is built with Streamlit, providing a user-friendly interface to interact with the model in real time.

---

## 🚀 Features

- **Intuitive Interface**: Input study habits via sliders and radio buttons in the sidebar.
- **Real-time Predictions**: Instantly predicts Performance Index based on given inputs.
- **Interactive Widgets**: Choose values for features like hours studied, previous scores, extracurriculars, and more.
- **Modern Design**: A clean, professional layout with styled headers, footers, and a loading spinner for better UX.

---

## 🧑‍💻 Getting Started

### Prerequisites

- Python 3.7 or higher
- Streamlit
- Requests library

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Rathorerishabh/MLOps-Student_Performnace_Indicator.git
   ```

2. **Install Dependencies**
   Install the required packages from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up MLflow Model Server**
   Ensure that your MLflow model server is running at `http://127.0.0.1:8000/invocations`.

4. **Run the Streamlit App**
   ```bash
   streamlit run app.py
   ```

   The app should automatically open in your web browser at `http://localhost:8501`.

---

## 📝 Usage

1. **Set Feature Values**: Adjust the sliders and select options in the sidebar to input values for:
   - Hours Studied
   - Previous Scores
   - Extracurricular Activities
   - Sleep Hours
   - Sample Question Papers Practiced

2. **Predict Performance**: Click the **Predict Performance Index** button to generate a prediction based on the inputs.

3. **View Prediction**: The app displays the predicted *Performance Index* along with any errors or issues in the request.

---

## ⚙️ Project Structure

- `app.py` - Main application code for Streamlit.
- `requirements.txt` - Required dependencies to run the app.

---

## 💡 Customization

To use your own MLflow model server URL, change the `url` variable in `app.py`:

```python
url = "http://your-model-server-address:8000/invocations"
```

---

## 🛠 Troubleshooting

- **Model Server Issues**: Ensure MLflow is running correctly and accessible at `http://127.0.0.1:8000/invocations`.
- **Failed Requests**: Check the request format if any errors arise from the server response. MLflow models in version 2.0+ require data in a dictionary format.

---

## 🤝 Contributing

Feel free to contribute to this project by opening issues or submitting pull requests. All contributions are welcome!

---

Enjoy using the **Performance Index Prediction App**! Built with ❤️ by Rishabh Singh
