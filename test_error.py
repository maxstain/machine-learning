from models.ml_model import MLModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # Initialize the model
    mdl = MLModel()
    
    # Process the dataset
    df = mdl.process_dataset()
    
    if df is None:
        logger.error("Could not process the dataset")
        exit(1)
    
    # Train the model
    if not mdl.train_model(df):
        logger.error("Could not train the model")
        exit(1)
    
    logger.info("Model trained successfully")
    
except Exception as e:
    logger.error(f"Error: {str(e)}")
    exit(1)