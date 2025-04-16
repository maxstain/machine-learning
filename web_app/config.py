"""
Configuration settings for the Insurance Claim Prediction application.
This module contains configuration classes for different environments.
"""

import os
from pathlib import Path

class Config:
    """Base configuration class."""
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-for-development-only')
    DEBUG = False
    TESTING = False
    
    # Application settings
    PROJECT_ROOT = Path(__file__).parent.parent.absolute()
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, 'dataset.csv')
    PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'processed_data.csv')
    
    # Model settings
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
    MODEL_FILE = os.path.join(MODEL_DIR, 'model.pkl')
    
    # Logging settings
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True
    
class ProductionConfig(Config):
    """Production configuration."""
    # In production, SECRET_KEY should be set as an environment variable
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
    # Use more secure settings in production
    # For example, you might want to use a different database
    # or set up more restrictive security policies

# Dictionary of configuration environments
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config(config_name=None):
    """
    Get the configuration object for the specified environment.
    
    Args:
        config_name: Name of the configuration environment
        
    Returns:
        Configuration object
    """
    if not config_name:
        config_name = os.environ.get('FLASK_ENV', 'default')
    return config.get(config_name, config['default'])