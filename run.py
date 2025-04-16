"""
Entry point for the Insurance Claim Prediction application.
This script creates and runs the Flask application.
"""

import os
from web_app import create_app

# Create the application instance
app = create_app(os.environ.get('FLASK_ENV', 'development'))

if __name__ == '__main__':
    # Run the application
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))