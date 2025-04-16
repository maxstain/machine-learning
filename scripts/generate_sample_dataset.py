"""
This script generates a sample dataset for the Insurance Claim Prediction application.
It creates a CSV file with the required structure and sample data.

Run this script to create a sample dataset in the data/raw directory.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the project root to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_app.config import get_config

def generate_sample_dataset(num_samples=100, output_path=None):
    """
    Generate a sample dataset with the required structure.
    
    Args:
        num_samples (int): Number of samples to generate
        output_path (str): Path where the dataset will be saved
    """
    # Get configuration
    config = get_config()
    
    # Use the configured path if none is provided
    if output_path is None:
        output_path = config.RAW_DATA_FILE
    
    # Create random data
    np.random.seed(42)  # For reproducibility
    
    # Define possible values for categorical variables
    agencies = ['Agency' + str(i) for i in range(1, 11)]
    agency_types = ['airlines', 'travel_agency', 'direct']
    channels = ['Online', 'Phone', 'Agent', 'Branch', 'Partner']
    products = ['Basic Insurance', 'Premium Coverage', 'Full Protection', 'Travel Plus', 'Business Trip']
    destinations = ['Europe', 'North America', 'Asia', 'Africa', 'Australia', 'South America']
    genders = ['M', 'F']
    
    # Generate random data
    data = {
        'Agency': np.random.choice(agencies, num_samples),
        'Agency Type': np.random.choice(agency_types, num_samples),
        'Distribution Channel': np.random.choice(channels, num_samples),
        'Product Name': np.random.choice(products, num_samples),
        'Claim': np.random.choice([0, 1], num_samples, p=[0.8, 0.2]),  # 20% chance of claim
        'Duration': np.random.randint(1, 31, num_samples),  # 1-30 days
        'Destination': np.random.choice(destinations, num_samples),
        'Net Sales': np.random.uniform(500, 3000, num_samples).round(2),
        'Commission (in value)': np.random.uniform(50, 300, num_samples).round(2),
        'Gender': np.random.choice(genders, num_samples),
        'Age': np.random.randint(18, 81, num_samples)  # 18-80 years
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Sample dataset with {num_samples} rows created at: {output_path}")
    
    # Print the first few rows
    print("\nFirst 5 rows of the dataset:")
    print(df.head())

if __name__ == "__main__":
    # Ensure the data directory exists
    config = get_config()
    data_dir = Path(config.RAW_DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate the sample dataset
    generate_sample_dataset()
    
    print("\nYou can now run the scripts/test_setup.py script to verify your setup.")