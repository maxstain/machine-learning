# Claim Prediction Using XGBoost

This project predicts insurance claims using an XGBoost classifier. The model pipeline includes preprocessing (
Yeo-Johnson transformation, scaling), handling class imbalance with SMOTE, and optimizing prediction thresholds.

## Created by:

- Firas Chabchoub
- Chaima Abbassi
- Ahmed Belhaj Amor
- Rached Ben Rhouma

## 📁 Project Structure

```
xgboost_claim_project/
├── data/
│   ├── example_claim_test.csv         # Example test data
│   └── model_outputs/                 # Output predictions
├── models/
│   ├── xgboost_claim_model.pkl        # Trained XGBoost model
│   ├── standard_scaler.pkl            # StandardScaler
│   ├── yeo_johnson_transformer.pkl    # PowerTransformer
│   └── xgb_threshold.txt              # Optimal classification threshold
├── src/
│   └── xgb_claim_predictor.py         # Prediction logic
├── run_prediction.py                  # Run script for demonstration
└── requirements.txt                   # Python package dependencies
```

## 🚀 How to Run

1. Make sure your test data is located at `data/example_claim_test.csv`
2. From the project root directory, run:

```bash
python run_prediction.py
```

3. Predictions will be saved to `data/model_outputs/predicted_claims_output.csv`

## 🔍 Notes

* Model input should contain columns matching the original training features.
* Categorical encoding should match training preprocessing.
* Output includes both prediction labels and claim probabilities.

## 🧩 Requirements

Install required packages:

```bash
pip install -r requirements.txt
```

## 📚 License

MIT License or adapt based on your university submission requirements.
