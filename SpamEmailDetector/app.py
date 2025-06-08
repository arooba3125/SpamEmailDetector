from flask import Flask, render_template, request
import joblib
from preprocess import clean_text
import numpy as np

app = Flask(__name__)
model = joblib.load("model/spam_classifier.pkl")
feature_union = joblib.load("model/feature_union.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    cleaned = clean_text(message)
    
    # Transform using the feature union
    vect_msg = feature_union.transform([cleaned])
    
    # Make prediction
    prediction = model.predict(vect_msg)
    probability = model.predict_proba(vect_msg)[0]
    
    # Prepare result with confidence
    if prediction[0] == 1:
        result = f"Spam ({(probability[1]*100):.1f}% confidence)"
    else:
        result = f"Not Spam ({(probability[0]*100):.1f}% confidence)"
    
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)