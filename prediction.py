import requests
import json

# Define the endpoint URL
url = "http://127.0.0.1:8000/invocations"

# Prepare your input data in the correct format using "instances" with feature names
data = {
    "instances": [
        {
            input("Hours Studied"): 7,
            "Previous Scores": 99,
            "Extracurricular Activities": 1,
            "Sleep Hours": 9,
            "Sample Question Papers Practiced": 1
        }
    ]
}

# Send the POST request
response = requests.post(url, data=json.dumps(data), headers={"Content-Type": "application/json"})

# Check the response
if response.status_code == 200:
    prediction = response.json()  # assuming the response is in JSON format
    print("Predicted Performance Index:", prediction)
else:
    print("Failed to get prediction:", response.text)
