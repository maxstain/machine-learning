import unittest
import pandas as pd
import os
import sys
from pathlib import Path

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


if __name__ == '__main__':
    unittest.main()