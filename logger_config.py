import logging
import os
from datetime import datetime

# Create a logs directory if it doesn't exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Log file name with date-time format
log_filename = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

def setup_logger(name):
    """
    Set up a logger accessible across multiple files.
    Args:
        name (str): Name of the logger, usually `__name__` from the module.
    Returns:
        logging.Logger: Configured logger.
    """
    # Define the logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set the lowest level to capture all logs

    # File handler for DEBUG and higher logs
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)

    # Console handler for INFO and higher logs
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # Add handlers to the logger (avoid duplicates)
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
