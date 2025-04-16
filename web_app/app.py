from flask import Flask, request, render_template, get_flashed_messages
from models.ml_model import MLModel
from utils.data_verification import verify_data_structure
import pandas as pd
import logging

app = Flask(__name__)
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_input_features(input_data, model_features):
    # First create the initial DataFrame with the input data
    initial_df = pd.get_dummies(input_data)

    # Create a DataFrame with all required features initialized to 0
    all_features = pd.DataFrame(0,
                                index=[0],
                                columns=model_features)

    # Update the values for features that exist in the input
    for col in initial_df.columns:
        if col in all_features.columns:
            all_features[col] = initial_df[col]

    return all_features


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

            if request.is_json:
                # Handle JSON input
                input_data = request.get_json()
                if not input_data:
                    return render_template('error.html',
                                           error="No JSON data provided")
            else:
                # Handle form input
                input_data = request.form.to_dict()

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

            if not mdl.model:
                return render_template('error.html',
                                       error="Model is not trained. Please check the logs.")

            try:
                # Convert input to DataFrame
                input_data = pd.DataFrame([request.json])

                # Prepare features all at once
                input_encoded = prepare_input_features(input_data, mdl.feature_names)

                for feature in mdl.feature_names:
                    if feature not in input_encoded.columns:
                        input_encoded[feature] = 0

                missing_cols = [col for col in mdl.feature_names if col not in input_encoded.columns]
                if missing_cols:
                    missing_data = pd.DataFrame(0, index=input_encoded.index, columns=missing_cols)
                    input_encoded = pd.concat([input_encoded, missing_data], axis=1)

                input_encoded = input_encoded[mdl.feature_names]

                if input_encoded[mdl.feature_names].isnull().values.any():
                    return render_template('error.html', error="Invalid input data. Please check the format.")

                # Make prediction
                prediction = mdl.model.predict(input_encoded)
                result = mdl.label_encoder.inverse_transform(prediction)[0]

                return render_template('result.html', prediction=result)

            except Exception as e:
                app.logger.error(f"Prediction error: {str(e)}")
                return render_template('error.html',
                                       error="Error making prediction. Please check the input format.")
        return None

    except Exception as e:
        app.logger.error(f"Unhandled exception: {str(e)}")
        return render_template('error.html',
                               error="An unexpected error occurred. Please check the logs.")


if __name__ == '__main__':
    app.run(debug=True)