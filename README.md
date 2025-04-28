# Insurance Claim Prediction Application

This application predicts agency types based on various insurance-related features using machine learning.

#### Created by:

- CHABCHOUB Firas
- ABBASSI Chaima
- BELHAJ AMOR Ahmed
- RHOUMA Rached

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone or download this repository to your local machine.

2. Create a virtual environment:
   ```
   python -m venv .venv
   ```

3. Activate the virtual environment:
    - On Windows:
      ```
      .\.venv\Scripts\Activate
      ```
    - On macOS/Linux:
      ```
      source .venv/bin/activate
      ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Data Setup

The application requires specific data files to be present:

1. Ensure you have the following directory structure:
   ```
   data/
   ├── raw/
   │   └── dataset.csv    # Raw input data
   └── processed/
       └── processed_data.csv  # Processed data (created by the application)
   ```

2. Place your raw dataset in `data/raw/dataset.csv`. The dataset must include the following columns:
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

   Here's a sample of how your dataset should look:

   ```
   Agency,Agency Type,Distribution Channel,Product Name,Claim,Duration,Destination,Net Sales,Commission (in value),Gender,Age
   Agency1,airlines,Channel1,Product1,0,10,Destination1,1000,100,M,30
   Agency2,travel_agency,Channel2,Product2,1,15,Destination2,1500,150,F,35
   Agency3,direct,Channel3,Product3,0,20,Destination3,2000,200,M,40
   Agency4,airlines,Channel1,Product2,1,12,Destination1,1200,120,F,45
   Agency5,travel_agency,Channel2,Product1,0,18,Destination2,1800,180,M,50
   ```

   Note:
    - 'Agency Type' must be one of: 'airlines', 'travel_agency', or 'direct'
    - 'Claim' should be 0 (No) or 1 (Yes)
    - 'Gender' should be 'M' or 'F'

   Alternatively, you can generate a sample dataset by running:
   ```
   python scripts/generate_sample_dataset.py
   ```
   This will create a dataset with 100 random samples in the correct format.

## Running the Application

1. Make sure your virtual environment is activated.

2. Verify your setup is working correctly:
   ```
   python scripts/test_setup.py
   ```
   This script will check if your data directory structure is correct, process the dataset, and train a model.

3. Start the Flask web application:
   ```
   python run.py
   ```

4. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

## Using the Application

### Making Predictions

1. Fill out the form on the homepage with the required information:
    - **Agency**: The name of the insurance agency
    - **Agency Type**: Select from Airlines, Travel Agency, or Direct
    - **Distribution Channel**: The channel through which the insurance was sold
    - **Product Name**: The name of the insurance product
    - **Claim**: Whether a claim was made (Yes/No)
    - **Duration**: The duration of the insurance policy
    - **Destination**: The travel destination
    - **Net Sales**: The net sales amount
    - **Commission**: The commission value
    - **Gender**: The gender of the insured person
    - **Age**: The age of the insured person

2. Click the "Predict" button to submit the form.

3. The application will process your input and display the prediction result.

### Understanding the Results

The prediction result shows the predicted agency type based on the input features. The possible agency types are:

- Airlines
- Travel Agency
- Direct

## Troubleshooting

If you encounter any issues:

1. Check that all required data files are present in the correct locations.
2. Ensure all required Python packages are installed.
3. Check the application logs for error messages.
4. Make sure all input fields are filled out correctly when making predictions.

## Running Tests

To run the tests for this application:

```
python -m unittest discover tests
```

## Quick Start Guide for First-Time Users

If you're new to this application, follow these steps to get started quickly:

1. **Set up your environment**:
   ```
   python -m venv .venv
   .\.venv\Scripts\Activate
   pip install -r requirements.txt
   ```

2. **Generate a sample dataset**:
   ```
   python scripts/generate_sample_dataset.py
   ```

3. **Verify your setup**:
   ```
   python scripts/test_setup.py
   ```

4. **Start the web application**:
   ```
   python run.py
   ```

5. **Open your browser** and go to `http://127.0.0.1:5000/`

6. **Fill out the form** with insurance details and click "Predict"

7. **View the prediction result** showing the predicted agency type

## Project Structure

The project is organized into the following directories:

- **data/**: Contains the raw and processed data
    - **raw/**: Raw input data (dataset.csv)
    - **processed/**: Processed data (processed_data.csv)
- **models/**: Contains the machine learning model implementation
    - **ml_model.py**: Main ML model class
- **scripts/**: Contains utility scripts
    - **generate_sample_dataset.py**: Script to generate a sample dataset
    - **test_setup.py**: Script to test the setup
- **tests/**: Contains test modules
    - **test_ml_model.py**: Tests for the ML model
- **utils/**: Contains utility functions
    - **data_verification.py**: Functions for verifying data structure and input
    - **public_imports.py**: Common imports used across the project
- **web_app/**: Contains the Flask web application
    - **__init__.py**: Application factory
    - **config.py**: Configuration settings
    - **errors.py**: Error handling
    - **routes.py**: Route handlers
    - **static/**: Static files (CSS, JavaScript, images)
    - **templates/**: HTML templates
- **run.py**: Entry point for the application

## Additional Information

- The application uses a Random Forest Classifier for predictions.
- The model is trained on the provided dataset each time the application starts.
- The processed data is saved to `data/processed/processed_data.csv` for future use.
- For API usage, you can send POST requests to `/predict` with JSON data containing the required fields.
- The application follows the Flask application factory pattern for better organization and testability.
- Configuration is separated from code using environment-specific configuration classes.