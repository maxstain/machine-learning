import joblib
from src.models.model_trainer import ModelTrainer


class MLModel:
    def __init__(self):
        self.model = None
        self.trainer = ModelTrainer()
        # TODO: TBD
        self.trainer.train_model()
        self.load_model()

    def load_model(self):
        try:
            self.model = joblib.load('model.pkl')
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def predict(self, features):
        try:
            return self.model.predict([features])
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")
