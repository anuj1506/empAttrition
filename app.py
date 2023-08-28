import json
import pickle
from flask import Flask

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
model = pickle.load(open('clf1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict():
    data = request.form.to_dict()
    features = np.array([float(x) for x in data.values()]).reshape(1, -1)
    
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]  # Probability of attrition
    
    if prediction[0] == 0:
        result = "Not Likely"
    else:
        result = "Likely"
    
    return render_template("home.html", prediction_text="Prediction: {} (Probability: {:.2f})".format(result, probability))

if __name__ == "__main__":
    app.run(debug=True)

