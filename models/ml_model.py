import joblib
from src.models.model_trainer import ModelTrainer


class MLModel:
    def __init__(self):
        self.model = None
        self.trainer = ModelTrainer()
        self.features = None
        self.target = None
        self.agency_type = None
        self.trainer.train_model()
        self.load_model()

    def process_dataset(self):
        """
        Load and process the dataset then save it to a csv called processed_data.csv.

        This function is called when the user submits the form in the web app.
        It handles the entire pipeline of loading, preprocessing, and saving the dataset.
        It also handles any exceptions that may occur during this process.

        Returns:
            None
        """
        try:
            # Save the dataset
            self.trainer.save_dataset()
            # Load the dataset
            self.trainer.load_dataset()
        except Exception as e:
            raise Exception(f"Error processing dataset: {str(e)}")

    def load_model(self):
        try:
            self.model = joblib.load('models/model.pkl')
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def predict(self, features):
        try:
            return self.model.predict([features])
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")
