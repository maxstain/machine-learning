# utils.py
import joblib
import os


def load_model_components(model_dir='models'):
    """Load model, scaler, transformer, and threshold from the specified directory."""
    model = joblib.load(os.path.join(model_dir, 'xgboost_claim_model.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'standard_scaler.pkl'))
    transformer = joblib.load(os.path.join(model_dir, 'yeo_johnson_transformer.pkl'))

    with open(os.path.join(model_dir, 'xgb_threshold.txt'), 'r') as f:
        threshold = float(f.read())

    return model, scaler, transformer, threshold


def apply_threshold(probs, threshold):
    """Convert probabilities to binary predictions using a threshold."""
    return (probs >= threshold).astype(int)
