# xgb_claim_predictor.py
import pandas as pd
from src.utils import load_model_components, apply_threshold
from sklearn.preprocessing import LabelEncoder

# --- Load Saved Components ---
xgb, scaler, pt, threshold = load_model_components()


# --- Prediction Function ---
def predict_claim(input_df):
    """
    Expects a DataFrame with raw numeric and categorical features,
    including columns: Duration, Net Sales, Commission (in value), Age, etc.
    Returns prediction labels and probabilities.
    """
    # Drop target column if present
    if 'Claim' in input_df.columns:
        input_df = input_df.drop(columns='Claim')

    # Apply transformations to numeric features
    numeric_cols = ['Duration', 'Net Sales', 'Commission (in value)', 'Age']
    input_df[numeric_cols] = pt.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Encode categorical features (must match training mapping)
    categorical_cols = ['Agency Type', 'Distribution Channel', 'Product Name', 'Destination', 'Gender']
    for col in categorical_cols:
        le = LabelEncoder()
        input_df[col] = le.fit_transform(input_df[col])

    # Predict probabilities
    y_probs = xgb.predict_proba(input_df)[:, 1]

    # Apply threshold
    predictions = apply_threshold(y_probs, threshold)

    return predictions, y_probs


# --- Example usage ---
if __name__ == "__main__":
    # Example input (replace with real test data)
    df_test = pd.read_csv("data/dataset.csv")

    preds, probs = predict_claim(df_test)
    df_test['Predicted Claim'] = preds
    df_test['Claim Probability'] = probs

    df_test.to_csv("data/model_outputs/predicted_claims_output.csv", index=False)
    print("Predictions saved to predicted_claims_output.csv")
