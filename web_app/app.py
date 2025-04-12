from src.data.data_cleaner import DataCleaner
from src.data.data_loader import DataLoader
from src.models.model_trainer import ModelTrainer
from src.utils.public_imports import *

from flask import render_template

import pandas as pd

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)


# Add error handling
@app.errorhandler(500)
def handle_500_error(e):
    return "Internal Server Error", 500


@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}")
    return "An unexpected error occurred", 500


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict')
def predict():
    # Placeholder for prediction logic
    try:
        # Simulate a prediction
        prediction = "Prediction result"
        return render_template('predict.html', prediction=prediction)
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return "Error during prediction", 500


CORS(app)

if __name__ == '__main__':
    try:
        # For development only
        data_loader = DataLoader('data/raw/insurance.csv')
        data_cleaner = DataCleaner()
        model_trainer = ModelTrainer()

        # Load and clean data
        raw_data = data_loader.load_data()
        cleaned_data = data_cleaner.clean_data(raw_data)

        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Error starting the Flask app: {e}")
        exit(1)
