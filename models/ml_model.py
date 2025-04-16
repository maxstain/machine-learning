import logging
import os
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_raw_data():
    """
    Attempts to load data from multiple possible locations and creates the necessary directories
    """
    possible_paths = [
        os.path.join('data', 'raw', 'dataset.csv'),
    ]

    data_dir = os.path.join('data', 'raw')
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    for path in possible_paths:
        try:
            if os.path.exists(path):
                print(f"Found data file at: {path}")
                return pd.read_csv(path)
        except Exception as e:
            print(f"Could not load data from {path}: {str(e)}")
            continue


def load_processed_data():
    """
    Attempts to load processed data from multiple possible locations and creates the necessary directories
    """
    possible_paths = [
        os.path.join('data', 'processed', 'processed_data.csv'),
        os.path.join(os.getcwd(), '..', 'data', 'processed', 'processed_data.csv'),
        os.path.join(os.getcwd(), 'data', 'processed_data.csv'),
    ]

    data_dir = os.path.join('data', 'processed')
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    for path in possible_paths:
        try:
            if os.path.exists(path):
                print(f"Found processed data file at: {path}")
                return pd.read_csv(path)
        except Exception as e:
            print(f"Could not load processed data from {path}: {str(e)}")
            continue

    raise FileNotFoundError(
        f"Could not find processed_data.csv in any of the following locations:\n"
        f"{chr(10).join(possible_paths)}\n"
        f"Current working directory is: {os.getcwd()}"
    )


class MLModel:
    def __init__(self):
        print("Initializing MLModel")
        self.model = None
        self.feature_names = None
        self.label_encoder = None
        self.data_path = None

    def clean_dataset(self, df):
        """
        Cleans the dataset by removing duplicates and handling missing values
        :return: cleaned DataFrame
        """
        if df is None:
            raise ValueError("No data available for cleaning")
        try:
            df = df.dropna()
            df = df.drop_duplicates()
            return df

        except Exception as e:
            print(f"Error cleaning dataset: {str(e)}")
            return None

    def process_dataset(self):
        try:
            # Try to load processed data first
            try:
                df = load_processed_data()
                logging.info("Loaded pre-processed data")

                # Initialize label encoder
                self.label_encoder = LabelEncoder()

                # Ensure agency_type_encoded exists
                if 'agency_type_encoded' not in df.columns:
                    # If not, we need to encode it
                    if 'Agency Type' in df.columns:
                        df['agency_type_encoded'] = self.label_encoder.fit_transform(df['Agency Type'])
                    else:
                        raise ValueError("Neither 'agency_type_encoded' nor 'Agency Type' found in processed data")
                else:
                    # If agency_type_encoded exists but we still need to initialize the label encoder
                    if 'Agency Type' in df.columns:
                        self.label_encoder.fit(df['Agency Type'])
                    else:
                        # Create a mapping of encoded values back to original values
                        unique_encoded = df['agency_type_encoded'].unique()
                        # Use default values if we can't determine the original values
                        original_values = ['airlines', 'travel_agency', 'direct'][:len(unique_encoded)]
                        self.label_encoder.fit(original_values)

                # Set feature names for training (including Claim as a feature, excluding Agency Type)
                self.feature_names = [col for col in df.columns if col != 'agency_type_encoded' and col != 'Agency Type']

                return df
            except FileNotFoundError:
                logging.info("No processed data found, processing raw data...")

            # If processed data not found, load and process raw data
            df = load_raw_data()

            if df is None or df.empty:
                raise ValueError("The loaded dataset is empty or could not be loaded.")

            required_columns = ['Agency', 'Agency Type', 'Distribution Channel', 'Product Name',
                                'Claim', 'Duration', 'Destination', 'Net Sales',
                                'Commission (in value)', 'Gender', 'Age']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Encode categorical columns - only encode Agency Type once
            self.label_encoder = LabelEncoder()
            df['agency_type_encoded'] = self.label_encoder.fit_transform(df['Agency Type'])

            # One-hot encode other categorical columns (excluding Agency Type which is already encoded)
            df = pd.get_dummies(df, columns=['Agency', 'Distribution Channel',
                                             'Product Name', 'Destination', 'Gender'],
                                prefix_sep='_')

            # Set feature names for training (including Claim as a feature, excluding Agency Type)
            self.feature_names = [col for col in df.columns if col != 'agency_type_encoded' and col != 'Agency Type']

            # Log the processed columns
            logging.info(f"Processed dataset columns: {self.feature_names}")

            # Save the processed data
            processed_path = os.path.join(os.getcwd(), 'data', 'processed', 'processed_data.csv')
            df.to_csv(processed_path, index=False)
            logging.info(f"Processed dataset saved successfully at {processed_path}")

            return df

        except Exception as e:
            logging.error(f"Error processing dataset: {str(e)}")
            return None

    def train_model(self, df):
        try:
            if df is None:
                raise ValueError("No data available for training")

            # Use the processed feature names
            X = df[self.feature_names]
            y = df['agency_type_encoded']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            self.model = RandomForestClassifier(random_state=42)
            self.model.fit(X_train, y_train)

            # Validate the model
            y_pred = self.model.predict(X_test)
            accuracy = (y_pred == y_test).mean()
            print(f"Model training completed successfully. Accuracy: {accuracy:.2f}")
            return True

        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False