"""
Routes for the Insurance Claim Prediction application.
This module contains the Flask routes for the web application.
"""

from flask import Blueprint, request, render_template
from models.ml_model import MLModel
from utils.data_verification import verify_data_structure, validate_input
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create blueprint
main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Route for the home page."""
    try:
        mdl = MLModel()
        df = mdl.clean_dataset(mdl.process_dataset())
        mdl.train_model(df)
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        return render_template('error.html',
                               error="An error occurred while processing the request. Please check the logs.",
                               details=str(e))

@main_bp.route('/predict', methods=['POST'])
def predict():
    """Route for making predictions."""
    try:
        if not verify_data_structure():
            return render_template('error.html',
                                   error="Data directory structure is incomplete. Please check the logs.")

        if request.method == 'POST':
            # Get agency_type from form or JSON
            if request.is_json:
                # Handle JSON input
                json_data = request.get_json()
                if not json_data:
                    return render_template('error.html',
                                           error="No JSON data provided")
                agency_type = json_data.get('agency_type')
                input_data = json_data
            else:
                # Handle form input
                agency_type = request.form.get('agency_type')
                input_data = request.form.to_dict()

            # Validate agency_type
            try:
                validate_input(agency_type)
            except ValueError as e:
                return render_template('error.html', error=str(e))

            # Initialize and train the model (only once)
            mdl = MLModel()
            df = mdl.process_dataset()

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
                input_df = pd.DataFrame([input_data])

                # Store agency_type for later use
                agency_type_value = input_df['agency_type'].iloc[0] if 'agency_type' in input_df.columns else None

                # Rename 'agency_type' to 'Agency Type' to match training data format
                if 'agency_type' in input_df.columns:
                    input_df['Agency Type'] = input_df['agency_type']
                    input_df = input_df.drop('agency_type', axis=1)

                # Process the input data the same way as during training
                # One-hot encode categorical columns (excluding Agency Type)
                # First, validate Agency Type
                if 'Agency Type' in input_df.columns and mdl.label_encoder is not None:
                    agency_type_value = input_df['Agency Type'].iloc[0]
                    try:
                        # Validate that the Agency Type value is one that the model knows about
                        if agency_type_value not in mdl.label_encoder.classes_:
                            logger.error(f"Invalid Agency Type value: {agency_type_value}")
                            return render_template('error.html',
                                                   error=f"Invalid 'Agency Type' value: {agency_type_value}. Must be one of {list(mdl.label_encoder.classes_)}")
                        logger.info(f"Valid Agency Type value: {agency_type_value}")
                    except Exception as e:
                        logger.error(f"Error validating Agency Type: {str(e)}")
                        return render_template('error.html',
                                               error=f"Error validating Agency Type: {str(e)}")

                # Remove Agency Type from input_df since we don't need it for prediction
                # (we're predicting agency_type_encoded)
                if 'Agency Type' in input_df.columns:
                    input_df = input_df.drop('Agency Type', axis=1)

                # One-hot encode the categorical columns
                categorical_columns = [col for col in input_df.columns if col in ['Agency', 'Distribution Channel',
                                                                                  'Product Name', 'Destination',
                                                                                  'Gender']]
                if categorical_columns:
                    input_df = pd.get_dummies(input_df, columns=categorical_columns, prefix_sep='_')

                # Create a DataFrame with all required features initialized to 0
                all_features = pd.DataFrame(0, index=[0], columns=mdl.feature_names)

                # Update the values for features that exist in the input
                for col in input_df.columns:
                    if col in all_features.columns:
                        all_features[col] = input_df[col]

                # Make prediction
                prediction = mdl.model.predict(all_features)
                result = mdl.label_encoder.inverse_transform(prediction)[0]

                return render_template('predict.html', prediction=result)

            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                return render_template('error.html',
                                       error="Error making prediction. Please check the input format.")
        return None

    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        return render_template('error.html',
                               error="An unexpected error occurred. Please check the logs.")