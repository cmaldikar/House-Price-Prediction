import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
file_path = 'data/realtor-data.csv'  # Adjust the path if necessary
data = pd.read_csv(file_path)

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Encode categorical variables
data = pd.get_dummies(data, columns=['city', 'state'])

# Select features and target
features = ['bed', 'bath', 'acre_lot', 'house_size'] + [col for col in data.columns if col.startswith('city_') or col.startswith('state_')]
X = data[features]
y = data['price']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model, scaler, and feature list
joblib.dump(model, 'models/house_price_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(features, 'models/features.pkl')

print("Model training complete and saved.")
