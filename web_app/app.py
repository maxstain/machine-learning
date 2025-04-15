from flask import Flask, request, render_template, get_flashed_messages
from models.ml_model import MLModel
from utils.data_verification import verify_data_structure
import pandas as pd
import logging

app = Flask(__name__)
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route('/')
def index():
    try:
        mdl = MLModel()
        df = mdl.clean_dataset(mdl.process_dataset())
        mdl.train_model(df)
        return render_template('index.html')
    except Exception as e:
        app.logger.error(f"Error in index route: {str(e)}")
        return render_template('error.html',
                               error="An error occurred while processing the request. Please check the logs.",
                               details=str(e))


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not verify_data_structure():
            return render_template('error.html',
                                   error="Data directory structure is incomplete. Please check the logs.")

        if request.method == 'POST':
            agency_type = request.form.get('agency_type')

            if not agency_type:
                return render_template('error.html',
                                       error="Agency type is required")

            mdl = MLModel()
            df = mdl.process_dataset()
            mdl.train_model(df)

            if df is None:
                return render_template('error.html',
                                       error="Could not process the dataset. Please check the logs.")

            if not mdl.train_model(df):
                return render_template('error.html',
                                       error="Could not train the model. Please check the logs.")

            try:
                input_data = pd.DataFrame({'agency_type': [agency_type]})
                input_encoded = pd.get_dummies(input_data)

                for col in mdl.feature_names:
                    if col not in input_encoded.columns:
                        input_encoded[col] = 0

                prediction = mdl.model.predict(input_encoded[mdl.feature_names])
                result = mdl.label_encoder.inverse_transform(prediction)[0]

                return render_template('result.html', prediction=result)

            except Exception as e:
                app.logger.error(f"Prediction error: {str(e)}")
                return render_template('error.html',
                                       error="Error making prediction. Please check the input format.")

    except Exception as e:
        app.logger.error(f"Unhandled exception: {str(e)}")
        return render_template('error.html',
                               error="An unexpected error occurred. Please check the logs.")


if __name__ == '__app__':
    app.run(debug=True)