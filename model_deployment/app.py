from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from preprocessing import preprocess

app = Flask(__name__)

# Load the model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json()

    # Preprocess the data
    processed_data = preprocess(data)  # Note the unpacking of the return value
    # print(processed_data)
    
    # # Make predictions
    predictions = model.predict(processed_data)
    probability = model.predict_proba(processed_data)
    # print(probability)

    # # # Return the predictions
    # return jsonify(predictions.tolist(), probability)
    predictions_list = predictions.tolist()
    probability_list = probability.tolist()

    # Return the predictions and probabilities as a JSON response
    response = {
        "predictions": predictions_list,
        "probabilities": probability_list
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
