import json
import os
from datetime import datetime
from src.utils.public_imports import *

import pandas as pd


def _load_data():
    """
    Load the processed training data from a CSV file.

    Returns:
        X: Features
        y: Labels
    """
    try:
        df = pd.read_csv('data/processed/processed_data.csv')
        X = df.drop('target', axis=1)
        y = df['target']
        return X, y
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None


class ModelTrainer:
    def __init__(self, model_path='models/'):
        self.data_path = os.path.join('data', 'processed', 'processed_data.csv')
        # Verify the path exists
        if not os.path.exists(self.data_path):
            print(f"Data file not found at: {self.data_path}")
            print(f"Current working directory: {os.getcwd()}")
        self.model_path = model_path
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'adaboost': AdaBoostClassifier(
                n_estimators=100,
                learning_rate=1.0,
                random_state=42
            )
        }
        self.best_model = None
        self.best_score = 0
        self.best_model_name = None

    def perform_cross_validation(self, model, X, y):
        if X is None or y is None:
            print("Cannot perform cross-validation with None values")
            return 0.0

        try:
            scores = cross_val_score(model, X, y, cv=5)
            return np.mean(scores)
        except Exception as e:
            print(f"Cross-validation error: {str(e)}")
            return 0.0

    def train_and_evaluate(self, X_train, y_train):
        if X_train is None or y_train is None:
            print("Training data is None. Cannot proceed with model training.")
            return None

        if self.best_model_name == 'xgboost':
            self.best_model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif self.best_model_name == 'adaboost':
            self.best_model = AdaBoostClassifier(
                n_estimators=100,
                learning_rate=1.0,
                random_state=42
            )

        if self.best_model is None:
            print("Model initialization failed")
            return None

        try:
            self.best_model.fit(X=X_train, y=y_train)
            return self.best_model
        except Exception as e:
            print(f"Error during model training: {str(e)}")
            return None

    def _evaluate_on_test_set(self, X_test, y_test):
        """
        Evaluate the best model on the test set

        Args:
            X_test: Test features
            y_test: Test labels
        """
        y_pred = self.best_model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        print("\nTest Set Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(class_report)

    def _save_model(self):
        """
        Save the best model to disk
        :return:
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.model_path, exist_ok=True)

            # Save model
            model_file = os.path.join(self.model_path, f'{self.best_model_name}_model.pkl')
            joblib.dump(self.best_model, model_file)

            # Save metadata
            metadata = {
                'model_name': self.best_model_name,
                'cv_score': self.best_score,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'parameters': self.best_model.get_params()
            }

            metadata_file = os.path.join(self.model_path, f'{self.best_model_name}_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4)

            print(f"\nModel saved successfully to {model_file}")

        except Exception as e:
            print(f"Error saving model: {str(e)}")

    def load_data(self):
        """
        Load the processed training data from a CSV file.

        Returns:
            X: Features
            y: Labels
        """
        try:
            df = pd.read_csv('data/processed/processed_data.csv')
            X = df.drop('target', axis=1)
            y = df['target']
            return X, y
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None, None

    def load_model(self):
        """
        Load a saved model

        Returns:
            loaded_model: The loaded model
        """
        try:
            model_file = os.path.join(self.model_path, f'model.pkl')
            loaded_model = joblib.load(model_file)
            return loaded_model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None

    def predict(self, X):
        """
        Make predictions using the best model

        Args:
            X: Features to predict

        Returns:
            predictions: Model predictions
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet!")
        return self.best_model.predict(X)

    def train_model(self):
        """
        Train the model using the training data and save it to disk.

        Return: void
        """

        # Load training data
        X_train, y_train = _load_data()

        # Train and evaluate models
        best_model = self.train_and_evaluate(X_train, y_train)

        # Save the best model
        self._save_model()

        return best_model

    def load_dataset(self):
        """
        Load the dataset from a CSV file.

        Returns:
            df: The loaded dataset
        """
        try:
            df = pd.read_csv('data/raw/dataset.csv')
            return df
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return None

    def preprocess_dataset(self):
        """
        Preprocess the dataset.

        Returns:
            df: The preprocessed dataset
        """
        try:
            df = self.load_dataset()
            df = df.dropna()
            df = df.drop_duplicates()
            df = df.drop(['agency_name', 'agency_id'], axis=1)
            df = df.drop(['agency_name', 'agency_id'], axis=1)
            df = df.drop(['agency_name', 'agency_id'], axis=1)
            # Convert target to binary
            df['target'] = df['target'].apply(lambda x: 1 if x == 'Yes' else 0)
            return df
        except Exception as e:
            print(f"Error preprocessing dataset: {str(e)}")
            return None

    def save_dataset(self):
        """
        Save the preprocessed dataset to a CSV file.

        Returns:
            void
        """
        try:
            df = self.preprocess_dataset()
            df.to_csv('data/processed/processed_data.csv', index=False)
        except Exception as e:
            print(f"Error saving dataset: {str(e)}")
            return None
