from flask import Flask, request, render_template, get_flashed_messages
from models.ml_model import MLModel
from utils.data_verification import verify_data_structure
import pandas as pd
import logging

app = Flask(__name__)
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

data_path = 'data/dataset.csv'


@app.route('/')
def index():
    try:
        mdl = MLModel(data_path, 'Claim')
        mdl.train()
        return render_template('index.html')
    except Exception as e:
        app.logger.error(f"Error in index route: {str(e)}")
        return render_template('error.html',
                               error="An error occurred while processing the request. Please check the logs.",
                               details=str(e))


@app.route('/predict', methods=['POST'])
def predict():
    pass


if __name__ == '__main__':
    app.run(debug=True)