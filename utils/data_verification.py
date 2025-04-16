from utils.public_imports import *


def validate_input(agency_type):
    valid_types = ['airlines', 'travel_agency', 'direct']
    if agency_type.lower() not in valid_types:
        raise ValueError(f"Invalid agency type. Must be one of {valid_types}")


def setup_logging():
    # Logging configuration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    return logger


def verify_data_structure():
    """
    Verifies the data directory structure and provides information about missing elements
    """
    base_dir = os.getcwd()
    required_structure = {
        'data': {
            'processed': ['processed_data.csv'],
            'raw': ['dataset.csv']
        }
    }

    missing_elements = []

    def check_structure(current_path, structure):
        for key, value in structure.items():
            path = os.path.join(current_path, key)
            if not os.path.exists(path):
                missing_elements.append(f"Missing directory: {path}")
            elif isinstance(value, dict):
                check_structure(path, value)
            elif isinstance(value, list):
                for file in value:
                    file_path = os.path.join(path, file)
                    if not os.path.exists(file_path):
                        missing_elements.append(f"Missing file: {file_path}")

    check_structure(base_dir, required_structure)

    if missing_elements:
        print("The following elements are missing:")
        for element in missing_elements:
            print(f"- {element}")
        return False
    return True