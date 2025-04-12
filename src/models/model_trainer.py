from src.utils.public_imports import *


class ModelTrainer:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'xgboost': xgb.XGBClassifier(random_state=42),
            'adaboost': AdaBoostClassifier(random_state=42)
        }
        self.best_model = None
        self.best_score = 0

    def train_and_evaluate(self, X_train, y_train):
        """Train and evaluate multiple models"""
        for name, model in self.models.items():
            scores = cross_val_score(model, X_train, y_train, cv=5)
            mean_score = scores.mean()

            print(f"{name} CV Score: {mean_score:.4f}")

            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_model = model

        # Train the best model on full training data
        self.best_model.fit(X_train, y_train)
        return self.best_model
