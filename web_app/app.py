from flask import Flask, render_template, request
import logging

from models.ml_model import MLModel
from src.models.model_trainer import ModelTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
model = MLModel()

trained_model = None
try:
    # Load the trained model
    trained_model = ModelTrainer().load_model()
    logger.info("Model loaded successfully.")
    # prediction logic
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])  # This line is crucial - methods=['POST']
def predict():
    mdl = MLModel()
    mdl.process_dataset()
    try:
        if request.method == 'POST':
            agency_type = request.form['agency_type']
            if not agency_type:
                return render_template('index.html', error="Please enter an agency type.")
            mdl.load_model()
            # Your prediction logic here
            logger.info(f"Received prediction request for agency type: {agency_type}")
            prediction = mdl.predict(agency_type)
            return render_template('predict.html', prediction=prediction)
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        return render_template('error.html', error=str(e)), 500


if __name__ == '__main__':
    app.run(debug=True)
