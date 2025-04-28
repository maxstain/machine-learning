import logging
import os
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBModel


class MLModel:
    def __init__(self, data_path: str, target_column: str):
        self.data_path = data_path
        self.target_column = target_column
        self.model = XGBModel()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.df = self.data_cleaning()
        self.load_data()

    def data_cleaning(self) -> pd.DataFrame:
        # Numerical features to scale
        numerical_features = ['Duration', 'Net Sales', 'Commission (in value)', 'Age']

        # Categorical features to include
        categorical_features = ['Agency Type', 'Distribution Channel', 'Product Name', 'Destination', 'Claim']

        logging.info("Cleaning data...")
        data = pd.read_csv(self.data_path)
        data.drop_duplicates(inplace=True)
        data.dropna(inplace=True)
        data = data.astype({
            'Agency Type': 'string',
            'Distribution Channel': 'string',
            'Product Name': 'string',
            'Claim': 'string',
            'Duration': 'float64',
            'Destination': 'string',
            'Net Sales': 'float64',
            'Commission (in value)': 'float64'
        })
        # Scale numerical features
        scaler = MinMaxScaler()
        scaled_numerical = scaler.fit_transform(data[numerical_features])
        # Create a DataFrame for scaled numerical features
        df_scaled_numerical = pd.DataFrame(scaled_numerical, columns=numerical_features)
        # Combine scaled numerical and categorical features
        df_final = pd.concat([df_scaled_numerical, data[categorical_features].reset_index(drop=True)], axis=1)
        # Encode categorical features
        label_encoder = LabelEncoder()
        for column in categorical_features:
            df_final[column] = label_encoder.fit_transform(df_final[column])
        # Save the cleaned data
        df_final.to_csv(self.data_path, index=False)
        logging.info(f"Data cleaned and saved to {self.data_path}")
        return df_final

    def load_data(self):
        data = pd.read_csv(self.data_path)
        X = data.drop(self.target_column, axis=1)
        y = data[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X):
        return self.model.predict(X)