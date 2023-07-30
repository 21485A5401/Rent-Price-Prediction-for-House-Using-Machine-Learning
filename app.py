# app.py

from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

app = Flask(__name__)

# Load the trained Random Forest model
from rent import rf_model, label_encoder, original_feature_names

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the uploaded data and preprocess it
        data = pd.DataFrame(request.form, index=[0])

        # Convert city to lowercase before encoding
        data['city'] = data['city'].apply(lambda x: x.lower())

        # Ensure the column ordering is consistent with the original feature names
        data = data[original_feature_names]

        # Use OrdinalEncoder for label encoding
        data_encoded = label_encoder.transform(data)

        # Make prediction using the trained model
        prediction = rf_model.predict(data_encoded)
        result = f"Predicted Rent Price: {prediction[0]:.2f} INR"

        return render_template('house.html', prediction=result)

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)