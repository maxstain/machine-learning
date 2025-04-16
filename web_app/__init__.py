"""
Web application package for the Insurance Claim Prediction application.
This package contains the Flask web application and related components.
"""

from flask import Flask
from web_app.config import get_config

def create_app(config_name=None):
    """
    Application factory function that creates and configures the Flask application.

    Args:
        config_name: Name of the configuration environment

    Returns:
        Flask application instance
    """
    app = Flask(__name__)

    # Configure the application
    config = get_config(config_name)
    app.config.from_object(config)

    # Register blueprints
    from web_app.routes import main_bp
    app.register_blueprint(main_bp)

    # Register error handlers
    from web_app.errors import init_app as init_errors
    init_errors(app)

    # Initialize logging
    import logging
    logging.basicConfig(
        level=getattr(logging, app.config.get('LOG_LEVEL', 'INFO')),
        format=app.config.get('LOG_FORMAT', '%(asctime)s - %(levelname)s - %(message)s')
    )

    return app