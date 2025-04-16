"""
This script tests if your setup is working correctly by:
1. Checking if the required directories and files exist
2. Attempting to load and process the dataset
3. Training a model on the dataset
4. Reporting success or any errors encountered

Run this script after setting up your environment to verify everything is working.
"""

import os
import logging
from models.ml_model import MLModel
from utils.data_verification import verify_data_structure

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_setup():
    """Test if the setup is working correctly."""
    try:
        # Step 1: Check if the required directories and files exist
        logger.info("Checking data directory structure...")
        if not verify_data_structure():
            logger.error("Data directory structure is incomplete.")
            logger.info("Please ensure you have the following structure:")
            logger.info("data/")
            logger.info("├── raw/")
            logger.info("│   └── dataset.csv    # Raw input data")
            logger.info("└── processed/")
            logger.info("    └── processed_data.csv  # Will be created if not exists")
            return False
        
        logger.info("Data directory structure is correct.")
        
        # Step 2: Attempt to load and process the dataset
        logger.info("Attempting to load and process the dataset...")
        mdl = MLModel()
        df = mdl.process_dataset()
        
        if df is None:
            logger.error("Could not process the dataset.")
            return False
        
        logger.info("Dataset processed successfully.")
        
        # Step 3: Train a model on the dataset
        logger.info("Attempting to train a model...")
        if not mdl.train_model(df):
            logger.error("Could not train the model.")
            return False
        
        logger.info("Model trained successfully.")
        
        # Step 4: Report success
        logger.info("Setup test completed successfully!")
        logger.info("You can now run the web application with: python web_app/app.py")
        return True
        
    except Exception as e:
        logger.error(f"Error during setup test: {str(e)}")
        return False

if __name__ == "__main__":
    test_setup()