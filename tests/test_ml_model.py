import unittest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import shutil
import tempfile

# Add the project root to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ml_model import MLModel


class TestMLModel(unittest.TestCase):
    """Test cases for the MLModel class"""

    def setUp(self):
        """Set up test fixtures"""
        self.model = MLModel()

        # Create a simple test dataframe
        self.test_df = pd.DataFrame({
            'Agency': ['Agency1', 'Agency2', 'Agency3'],
            'Agency Type': ['airlines', 'travel_agency', 'direct'],
            'Distribution Channel': ['Channel1', 'Channel2', 'Channel3'],
            'Product Name': ['Product1', 'Product2', 'Product3'],
            'Claim': [0, 1, 0],
            'Duration': [10, 20, 30],
            'Destination': ['Dest1', 'Dest2', 'Dest3'],
            'Net Sales': [100, 200, 300],
            'Commission (in value)': [10, 20, 30],
            'Gender': ['M', 'F', 'M'],
            'Age': [25, 35, 45]
        })

        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.raw_dir = os.path.join(self.temp_dir, 'data', 'raw')
        self.processed_dir = os.path.join(self.temp_dir, 'data', 'processed')
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        # Save test data to the temporary directory
        self.test_df.to_csv(os.path.join(self.raw_dir, 'dataset.csv'), index=False)

    def tearDown(self):
        """Clean up after tests"""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test that the model initializes correctly"""
        self.assertIsNone(self.model.model)
        self.assertIsNone(self.model.feature_names)
        self.assertIsNone(self.model.label_encoder)
        self.assertIsNone(self.model.data_path)

    def test_clean_dataset(self):
        """Test the clean_dataset method"""
        # Create a dataframe with duplicates and NaN values
        df = self.test_df.copy()
        df = pd.concat([df, df.iloc[[0]]])  # Add a duplicate row
        df.loc[3, 'Age'] = None  # Add a NaN value

        # Clean the dataset
        cleaned_df = self.model.clean_dataset(df)

        # Check that duplicates and NaN values were removed
        self.assertEqual(len(cleaned_df), 3)
        self.assertFalse(cleaned_df.isna().any().any())

    def test_clean_dataset_with_none(self):
        """Test the clean_dataset method with None input"""
        with self.assertRaises(ValueError):
            self.model.clean_dataset(None)

    def test_process_dataset(self):
        """Test the process_dataset method"""
        # Mock the load_raw_data and load_processed_data functions
        original_load_raw_data = self.model.process_dataset.__globals__['load_raw_data']
        original_load_processed_data = self.model.process_dataset.__globals__['load_processed_data']

        try:
            # Make load_processed_data raise FileNotFoundError to force using load_raw_data
            self.model.process_dataset.__globals__['load_processed_data'] = lambda: exec('raise FileNotFoundError("Test")')
            self.model.process_dataset.__globals__['load_raw_data'] = lambda: self.test_df

            # Process the dataset
            processed_df = self.model.process_dataset()

            # Check that the processed dataframe has the expected columns
            self.assertIsNotNone(processed_df)
            self.assertIn('agency_type_encoded', processed_df.columns)
            self.assertIsNotNone(self.model.feature_names)
            self.assertIsNotNone(self.model.label_encoder)

            # Check that one-hot encoding was applied
            self.assertTrue(any(col.startswith('Agency_') for col in processed_df.columns))
            self.assertTrue(any(col.startswith('Distribution Channel_') for col in processed_df.columns))

            # Check that Agency Type was not one-hot encoded (only label encoded)
            self.assertFalse(any(col.startswith('Agency Type_') for col in processed_df.columns))

        finally:
            # Restore the original functions
            self.model.process_dataset.__globals__['load_raw_data'] = original_load_raw_data
            self.model.process_dataset.__globals__['load_processed_data'] = original_load_processed_data

    def test_train_model(self):
        """Test the train_model method"""
        # Create a simple processed dataframe for testing
        processed_df = pd.DataFrame({
            'Duration': [10, 20, 30],
            'Net Sales': [100, 200, 300],
            'Commission (in value)': [10, 20, 30],
            'Age': [25, 35, 45],
            'Claim': [0, 1, 0],
            'Agency_Agency1': [1, 0, 0],
            'Agency_Agency2': [0, 1, 0],
            'Agency_Agency3': [0, 0, 1],
            'Distribution Channel_Channel1': [1, 0, 0],
            'Distribution Channel_Channel2': [0, 1, 0],
            'Distribution Channel_Channel3': [0, 0, 1],
            'Product Name_Product1': [1, 0, 0],
            'Product Name_Product2': [0, 1, 0],
            'Product Name_Product3': [0, 0, 1],
            'Destination_Dest1': [1, 0, 0],
            'Destination_Dest2': [0, 1, 0],
            'Destination_Dest3': [0, 0, 1],
            'Gender_F': [0, 1, 0],
            'Gender_M': [1, 0, 1],
            'agency_type_encoded': [0, 1, 2]
        })

        # Set feature names to all columns except agency_type_encoded
        self.model.feature_names = [col for col in processed_df.columns if col != 'agency_type_encoded']

        # Create a label encoder
        from sklearn.preprocessing import LabelEncoder
        self.model.label_encoder = LabelEncoder()
        self.model.label_encoder.classes_ = np.array(['airlines', 'travel_agency', 'direct'])

        # Train the model
        result = self.model.train_model(processed_df)

        # Check that training was successful
        self.assertTrue(result)
        self.assertIsNotNone(self.model.model)

        # Check that the model can make predictions
        X = processed_df[self.model.feature_names]
        predictions = self.model.model.predict(X)
        self.assertEqual(len(predictions), len(processed_df))

    def test_train_model_with_none(self):
        """Test the train_model method with None input"""
        result = self.model.train_model(None)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()