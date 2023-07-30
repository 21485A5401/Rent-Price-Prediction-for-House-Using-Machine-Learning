# rent.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder

# Load the dataset
data = pd.read_csv('99acres_data.csv')

# Preprocess the data
# (Add any necessary preprocessing steps here)

# Drop any rows with missing values
data.dropna(inplace=True)


# Define the features and target variable
X = data.drop('monthly_rant', axis=1)
y = data['monthly_rant']

# Define the original feature names in the order they appear in the dataset
original_feature_names = X.columns.tolist()

# Initialize and fit the OrdinalEncoder for label encoding
label_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X = label_encoder.fit_transform(X)

# Create and train the RandomForestRegressor model
rf_model = RandomForestRegressor()
rf_model.fit(X, y)
