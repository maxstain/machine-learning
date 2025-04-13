from public_imports import *


def validate_input(agency_type):
    valid_types = ['airlines', 'travel_agency', 'direct']
    if agency_type.lower() not in valid_types:
        raise ValueError(f"Invalid agency type. Must be one of {valid_types}")


def setup_logging():
    # Logging configuration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    return logger
