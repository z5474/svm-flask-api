from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# ✅ Load the trained SVM model
svm_model = joblib.load("fsvm_model.pkl")

@app.route('/')
def home():
    return "Welcome to the SVM Model Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    df = pd.read_excel(file)  # Read the uploaded Excel file

    # ✅ Ensure input matches model requirements
    if df.shape[1] != 11680:
        return jsonify({"error": "Incorrect number of columns in dataset"}), 400

    predictions = svm_model.predict(df)

    return jsonify({"predictions": predictions.tolist()})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
