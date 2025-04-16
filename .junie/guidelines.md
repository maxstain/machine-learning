# Machine Learning Project Guidelines

This document provides essential information for developers working on this machine learning project.

## Build/Configuration Instructions

### Environment Setup

1. **Python Environment**: This project requires Python 3.8+ and uses a virtual environment.

   ```powershell
   # Create a virtual environment
   python -m venv .venv
   
   # Activate the virtual environment
   # On Windows:
   .\.venv\Scripts\Activate
   
   # On Unix/MacOS:
   # source .venv/bin/activate
   ```

2. **Dependencies**: Install required packages using pip:

   ```powershell
   pip install -r requirements.txt
   ```

### Data Structure

The project expects a specific data directory structure:

```
data/
├── raw/
│   └── dataset.csv    # Raw input data
└── processed/
    └── processed_data.csv  # Processed data (created by the application)
```

- Create these directories if they don't exist
- Place your raw dataset in `data/raw/dataset.csv`
- The app will verify this structure using `verify_data_structure()` from `utils/data_verification.py`

### Required Dataset Columns

The dataset must include the following columns:
- 'Agency'
- 'Agency Type'
- 'Distribution Channel'
- 'Product Name'
- 'Claim'
- 'Duration'
- 'Destination'
- 'Net Sales'
- 'Commission (in value)'
- 'Gender'
- 'Age'

### Running the Application

The app is a Flask web app that can be started with:

```powershell
python web_app/app.py
```

This will start the development server at http://127.0.0.1:5000/

## Testing Information

### Test Structure

Tests are organized in the `tests/` directory with filenames prefixed with `test_`.

### Running Tests

Individual tests can be run using the Python unittest module:

```powershell
# Run a specific test file
python -m tests.test_ml_model

# Run all tests
python -m unittest discover tests
```

### Adding New Tests

1. Create a new test file in the `tests/` directory with the naming convention `test_*.py`
2. Import the unittest module and the components you want to test
3. Create a class that inherits from `unittest.TestCase`
4. Add test methods that start with `test_`
5. Use assertions to verify expected behavior

Example:

```python
import unittest
from models.ml_model import MLModel

class TestNewFeature(unittest.TestCase):
    def setUp(self):
        # Setup code runs before each test
        self.model = MLModel()
        
    def test_feature_behavior(self):
        # Test code
        result = self.model.some_method()
        self.assertEqual(result, expected_value)
        
if __name__ == '__main__':
    unittest.main()
```

### Test Data

For testing, you can create synthetic data frames as shown in the example test:

```python
test_df = pd.DataFrame({
    'Agency': ['Agency1', 'Agency2', 'Agency3'],
    'Agency Type': ['airlines', 'travel_agency', 'direct'],
    # Include all required columns
})
```

## Additional Development Information

### Code Style

- The project follows PEP 8 style guidelines
- Functions and methods include docstrings in the following format:
  ```python
  def function_name(param1, param2):
      """
      Brief description of the function
      
      :param param1: Description of param1
      :param param2: Description of param2
      :return: Description of return value
      """
  ```

### Project Structure

- `models/`: Contains machine learning model definitions
- `utils/`: Utility functions and common imports
- `web_app/`: Flask web app
- `data/`: Data directory (raw and processed)
- `notebooks/`: Jupyter notebooks for analysis
- `tests/`: Test files

### Key Parts

1. **MLModel Class** (`models/ml_model.py`):
   - Handles data processing, cleaning, and model training
   - Use RandomForestClassifier for prediction
   - Requires specific columns in the input data

2. **Data Verification** (`utils/data_verification.py`):
   - Validates input data
   - Verifies directory structure
   - Sets up logging

3. **Web Application** (`web_app/app.py`):
   - Flask app for model interaction
   - Provides endpoints for prediction
   - Handles errors and input validation

### Error Handling

- The app uses Python's exception handling with specific error messages
- Flask routes include try-except blocks to catch and display errors
- Logging is configured in both the web app and model components

### Model Training and Prediction

- The model predicts 'agency_type_encoded' based on various features
- Training data is split 80/20 (train/test)
- One-hot encoding is used for categorical variables