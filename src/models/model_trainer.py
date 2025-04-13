import json
import os
from datetime import datetime
from src.utils.public_imports import *


class ModelTrainer:
    def __init__(self, model_path='models/'):
        """
        Initialize the ModelTrainer with a set of models and path for saving

        Args:
            model_path (str): Path to save trained models
        """
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

    def train_and_evaluate(self, X_train, y_train, X_test=None, y_test=None):
        """
        Train and evaluate multiple models using cross-validation

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (optional)
            y_test: Test labels (optional)

        Returns:
            best_model: The best performing model
        """
        results = {}

        for name, model in self.models.items():
            try:
                # Perform cross-validation
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                mean_score = scores.mean()
                std_score = scores.std()

                results[name] = {
                    'mean_cv_score': mean_score,
                    'std_cv_score': std_score
                }

                print(f"{name}:")
                print(f"CV Score: {mean_score:.4f} (+/- {std_score:.4f})")

                if mean_score > self.best_score:
                    self.best_score = mean_score
                    self.best_model = model
                    self.best_model_name = name

            except Exception as e:
                print(f"Error training {name}: {str(e)}")

        # Train the best model on full training data
        print(f"\nBest model: {self.best_model_name} (CV Score: {self.best_score:.4f})")
        self.best_model.fit(X_train, y_train)

        # If test data is provided, evaluate on it
        if X_test is not None and y_test is not None:
            self._evaluate_on_test_set(X_test, y_test)

        # Save the model
        self._save_model()

        return self.best_model

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
        """Save the best model to disk"""
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
