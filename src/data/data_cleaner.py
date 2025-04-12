from src.utils.public_imports import *


class DataCleaner:
    def __init__(self):
        self.label_encoders = {}

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the insurance data"""
        # Create a copy of the dataframe
        df_cleaned = df.copy()

        # Handle missing values
        df_cleaned = df_cleaned.dropna()

        # Convert duration to numeric (assuming it's in days)
        df_cleaned['Duration'] = pd.to_numeric(df_cleaned['Duration'], errors='coerce')

        # Convert Net Sales and Commission to numeric
        df_cleaned['Net Sales'] = pd.to_numeric(df_cleaned['Net Sales'], errors='coerce')
        df_cleaned['Commission'] = pd.to_numeric(df_cleaned['Commission'], errors='coerce')

        # Encode categorical variables
        categorical_columns = ['Agency', 'Agency.Type', 'Distribution.Channel',
                               'Product.Name', 'Destination', 'Gender']

        for column in categorical_columns:
            self.label_encoders[column] = LabelEncoder()
            df_cleaned[column] = self.label_encoders[column].fit_transform(df_cleaned[column])
        df_cleaned.to_csv('data/cleaned/cleaned_data.csv', index=False)
        df_cleaned = pd.read_csv('data/cleaned/cleaned_data.csv')
        df_cleaned = df_cleaned.dropna()

        return df_cleaned
