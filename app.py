from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# ----------------------------
# Load model & sample data
# ----------------------------
MODEL_PATH = 'random_forest_model_forplant.joblib'
DATA_PATH = 'plant_health_data.csv'
model = None
data_sample = None

FEATURE_COLUMNS = [
    'Soil_Moisture', 'Ambient_Temperature', 'Soil_Temperature', 'Humidity',
    'Light_Intensity', 'Soil_pH', 'Nitrogen_Level', 'Phosphorus_Level',
    'Potassium_Level', 'Chlorophyll_Content', 'Electrochemical_Signal'
]

def load_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"✅ Model loaded successfully from {MODEL_PATH}")
            return True
        else:
            print(f"❌ Model file not found: {MODEL_PATH}")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

def load_sample_data():
    global data_sample
    try:
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            data_sample = df[FEATURE_COLUMNS].describe()
            print(f"✅ Sample data loaded from {DATA_PATH}")
            return True
        else:
            print(f"⚠️ Sample data file not found: {DATA_PATH}")
            return False
    except Exception as e:
        print(f"❌ Error loading sample data: {str(e)}")
        return False

# ----------------------------
# Utility functions
# ----------------------------
def validate_sensor_data(data):
    try:
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
        missing_cols = set(FEATURE_COLUMNS) - set(df.columns)
        if missing_cols:
            return None, f"Missing required columns: {missing_cols}"
        feature_data = df[FEATURE_COLUMNS]
        if feature_data.isnull().any().any():
            return None, "Missing values detected in sensor data"
        return feature_data.values, None
    except Exception as e:
        return None, f"Data validation error: {str(e)}"

def interpret_prediction(prediction, probabilities=None):
    try:
        if isinstance(prediction, (list, np.ndarray)):
            pred_value = prediction[0]
        else:
            pred_value = prediction
        status_mapping = {
            'High Stress': {'status': 'high_stress', 'message': '🚨 High Stress Detected'},
            'Moderate Stress': {'status': 'moderate_stress', 'message': '⚠️ Moderate Stress Detected'},
            'Low Stress': {'status': 'low_stress', 'message': '🟡 Low Stress Detected'},
            'Healthy': {'status': 'healthy', 'message': '🌱 Plant is Healthy'}
        }
        if str(pred_value) in status_mapping:
            return status_mapping[str(pred_value)]
        elif pred_value in [0, 1, 2, 3]:
            status_list = ['Healthy', 'Low Stress', 'Moderate Stress', 'High Stress']
            return status_mapping[status_list[int(pred_value)]]
        else:
            return {'status': 'unknown', 'message': f'Unknown prediction: {pred_value}'}
    except Exception as e:
        return {'status': 'error', 'message': f'Error: {str(e)}'}

# ----------------------------
# Routes
# ----------------------------
@app.route('/')
def index():
    return jsonify({"message": "🌱 Plant Health Monitor Backend running on Render!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No sensor data provided'}), 400
        features, error = validate_sensor_data(data)
        if error:
            return jsonify({'error': error}), 400
        prediction = model.predict(features)
        status_info = interpret_prediction(prediction)
        return jsonify({
            "success": True,
            "prediction": str(prediction[0]),
            "status": status_info
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/sample-data', methods=['GET'])
def get_sample_data():
    try:
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            samples = df.tail(5)[FEATURE_COLUMNS].to_dict('records')
            return jsonify({'success': True, 'samples': samples})
        else:
            return jsonify({'success': False, 'error': 'CSV not found'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'data_available': data_sample is not None,
        'timestamp': datetime.now().isoformat()
    })

# ----------------------------
# Main entry
# ----------------------------
if __name__ == '__main__':
    load_model()
    load_sample_data()
    app.run(debug=True, host='0.0.0.0', port=5000)
