"""
Error handling for the Insurance Claim Prediction application.
This module contains error handlers and custom exceptions.
"""

from flask import render_template, Blueprint
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create blueprint
errors_bp = Blueprint('errors', __name__)

class ModelError(Exception):
    """Exception raised for errors in the model."""
    pass

class DataError(Exception):
    """Exception raised for errors in the data."""
    pass

class InputError(Exception):
    """Exception raised for errors in the input."""
    pass

@errors_bp.app_errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    logger.error(f"404 error: {error}")
    return render_template('error.html', error="Page not found"), 404

@errors_bp.app_errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"500 error: {error}")
    return render_template('error.html', error="An internal error occurred. Please try again later."), 500

@errors_bp.app_errorhandler(ModelError)
def model_error(error):
    """Handle model errors."""
    logger.error(f"Model error: {error}")
    return render_template('error.html', error=f"Model error: {str(error)}"), 500

@errors_bp.app_errorhandler(DataError)
def data_error(error):
    """Handle data errors."""
    logger.error(f"Data error: {error}")
    return render_template('error.html', error=f"Data error: {str(error)}"), 500

@errors_bp.app_errorhandler(InputError)
def input_error(error):
    """Handle input errors."""
    logger.error(f"Input error: {error}")
    return render_template('error.html', error=f"Input error: {str(error)}"), 400

def init_app(app):
    """Initialize error handling for the application."""
    app.register_blueprint(errors_bp)
    
    # Set up logging
    if not app.debug:
        # In production, you might want to log to a file
        handler = logging.FileHandler('app.log')
        handler.setLevel(logging.INFO)
        app.logger.addHandler(handler)
    
    return app