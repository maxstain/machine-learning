# main.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_clean_data(self):
        df = pd.read_csv(self.file_path)
        # Basic cleaning
        return df


class ModelTrainer:
    def __init__(self, df):
        self.df = df
        self.model = None

    def train_model(self):
        # Prepare your data
        X = self.df.drop('target_column', axis=1)  # Replace 'target_column' with your actual target
        y = self.df['target_column']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model (replace with your actual model)
        self.model = YourModel()  # Replace YourModel with actual model class
        self.model.fit(X_train, y_train)

        # Evaluate and save
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy}")

        joblib.dump(self.model, 'model.pkl')


def create_app():
    app = Flask(__name__)
    model = joblib.load('model.pkl')

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.get_json(force=True)
            data_df = pd.DataFrame([data])
            prediction = model.predict(data_df)
            return jsonify({'prediction': int(prediction[0])})
        except Exception as e:
            return jsonify({'error': str(e)}), 400

    @app.route('/')
    def home():
        return render_template('index.html')

    return app


if __name__ == '__main__':
    # Train the model
    data_processor = DataProcessor('PROJET6/dataset.csv')
    df = data_processor.load_and_clean_data()

    model_trainer = ModelTrainer(df)
    model_trainer.train_model()

    # Run the app
    app = create_app()
    app.run(debug=True)
