from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
app = Flask(__name__)

with open('my_churn_prediction_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)
with open('onehot_encoder.pkl', 'rb') as model_file2:
    ohe = pickle.load(model_file2)
@app.route('/')
def home():
    return """
    <html>
        <body>
            <h1>Welcome to my Churn Prediction API</h1>
            <p>Click <a href="/predict">here</a> to go to the prediction page.</p>
        </body>
    </html>
    """
bill_per_gb_bins = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
bill_subscription_product_bins = [27.69, 267.006, 503.952, 740.898, 977.844, 1214.79, 1451.736, 1688.682, 1925.628, 2162.574, 2399.52]
avg_usage_per_month_bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
age_bins = [18, 30, 40, 50, 60, 70]
def calculate_bin(value, bins):
    for i in range(len(bins)-1):
        if value >= bins[i] and value < bins[i+1]:
            return f'({bins[i]}, {bins[i+1]}]'
@app.route('/predict', methods=['GET', 'POST'])
def predict_churn():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
    # Getting the inputs from the request
        age = request.json.get('Age')
        monthly_bill = request.json.get('Monthly_Bill')
        total_usage_gb = request.json.get('Total_Usage_GB')
        bill_per_gb = monthly_bill / total_usage_gb
        subscription_length_months = request.json.get('Subscription_Length_Months')
        bill_subscription_product = monthly_bill * subscription_length_months
        avg_usage_per_month = total_usage_gb / subscription_length_months
        bill_per_gb_bin = calculate_bin(bill_per_gb, bill_per_gb_bins)
        bill_subscription_product_bin = calculate_bin(bill_subscription_product, bill_subscription_product_bins)
        avg_usage_per_month_bin = calculate_bin(avg_usage_per_month, avg_usage_per_month_bins)
        age_bin = calculate_bin(age, age_bins)
        location = request.json.get('Location')
        gender = request.json.get('Gender')
        features = [
        age_bin, 
        location, 
        gender, 
        bill_per_gb_bin, 
        bill_subscription_product_bin, 
        avg_usage_per_month_bin]
        features = np.array(features).reshape(1, -1)
        features_encoded = ohe.transform(features).toarray()
        prediction = loaded_model.predict(features_encoded)
        return {'prediction': prediction.tolist()}

if __name__ == '__main__':
    app.run(debug=True)

