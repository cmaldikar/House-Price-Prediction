from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model, scaler, and features
model = joblib.load('models/house_price_model.pkl')
scaler = joblib.load('models/scaler.pkl')
features = joblib.load('models/features.pkl')

# Load the dataset
file_path = 'data/realtor-data.csv'
data = pd.read_csv(file_path)

@app.route('/')
def home():
    # Extract unique cities and states
    cities = data['city'].unique()
    states = data['state'].unique()
    return render_template('index.html', cities=cities, states=states)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    bed = int(request.form['bed'])
    bath = int(request.form['bath'])
    acre_lot = float(request.form['acre_lot'])
    house_size = int(request.form['house_size'])
    city = request.form['city']
    state = request.form['state']

    # Create a feature vector based on input
    input_data = {'bed': [bed], 'bath': [bath], 'acre_lot': [acre_lot], 'house_size': [house_size]}
    
    # Add city and state one-hot encoded columns
    for feature in features:
        if feature.startswith('city_'):
            input_data[feature] = [1 if feature == f'city_{city}' else 0]
        elif feature.startswith('state_'):
            input_data[feature] = [1 if feature == f'state_{state}' else 0]

    # Convert to DataFrame and scale
    input_df = pd.DataFrame(input_data)
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)
    prediction_text = f'Estimated House Price: ${prediction[0]:,.2f}'

    return render_template('index.html', prediction_text=prediction_text, cities=data['city'].unique(), states=data['state'].unique())

if __name__ == '__main__':
    app.run(debug=True)
