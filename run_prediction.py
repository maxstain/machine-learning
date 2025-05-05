# run_prediction.py
import pandas as pd
import os
from src.xgb_claim_predictor import predict_claim

# Load test dataset
df_test = pd.read_csv('data/dataset.csv')

# Make predictions
preds, probs = predict_claim(df_test)

df_test['Predicted Claim'] = preds
df_test['Claim Probability'] = probs

# Ensure output directory exists
os.makedirs('data/model_outputs', exist_ok=True)

# Save results
df_test.to_csv('data/model_outputs/predicted_claims_output.csv', index=False)
print("Prediction complete. Output saved to data/model_outputs/predicted_claims_output.csv")
