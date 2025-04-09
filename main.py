from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import logging

logging.basicConfig(level=logging.INFO)


class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_clean_data(self):
        df = pd.read_csv(self.file_path)
        # Cleaning steps
        df.dropna(inplace=True)  # Remove rows with missing values
        df = pd.get_dummies(df, columns=['Agency', 'Agency.Type', 'Distribution.Channel',
                                         'Product.Name', 'Destination', 'Gender'], drop_first=True)
        return df


class ModelTrainer:
    def __init__(self, df):
        self.df = df
        self.model = None
        self.training_columns = None

    def train_model(self):
        # Split features and target
        X = self.df.drop('Claim.Status', axis=1)
        y = self.df['Claim.Status']

        # Save training columns for later prediction
        self.training_columns = X.columns.tolist()
        with open('training_columns.pkl', 'wb') as f:
            joblib.dump(self.training_columns, f)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate and save
        y_pred = self.model.predict(X_test)
        print("Model evaluation:")
        print(classification_report(y_test, y_pred))

        joblib.dump(self.model, 'model.pkl')


def create_app():
    app = Flask(__name__)

    # Load model and training columns
    model_path = 'model.pkl'
    columns_path = 'training_columns.pkl'

    if os.path.exists(model_path) and os.path.exists(columns_path):
        model = joblib.load(model_path)
        with open(columns_path, 'rb') as f:
            training_columns = joblib.load(f)
    else:
        model = None
        training_columns = []

    @app.route('/predict', methods=['POST'])
    def predict():
        logging.info("Received prediction request")
        if not model:
            return jsonify({'error': 'Model not found. Train the model first.'}), 400

        try:
            # Get raw text from the form
            raw_data = request.form['data']

            # Parse the text as JSON
            import json
            try:
                parsed_data = json.loads(raw_data)
            except json.JSONDecodeError:
                return jsonify({'error': 'Invalid JSON format'}), 400

            # Convert parsed JSON to DataFrame
            data = pd.DataFrame([parsed_data])

            # Preprocess and predict (same logic as before)
            data = pd.get_dummies(data, columns=['Agency', 'Agency.Type', 'Distribution.Channel',
                                                 'Product.Name', 'Destination', 'Gender'], drop_first=True)

            # Add missing columns and match training structure
            missing_cols = set(training_columns) - set(data.columns)
            for col in missing_cols:
                data[col] = 0
            data = data[training_columns]

            # Predict and return result
            prediction = model.predict(data)
            return jsonify({'prediction': int(prediction[0])})
        except Exception as e:
            return jsonify({'error': f'An error occurred: {str(e)}'}), 400

    @app.route('/')
    def home():
        return render_template('index.html')

    return app


if __name__ == '__main__':
    # Train the model
    data_processor = DataProcessor('dataset.csv')
    df = data_processor.load_and_clean_data()

    model_trainer = ModelTrainer(df)
    model_trainer.train_model()

    # Run the app
    app = create_app()
    app.run(debug=True)
