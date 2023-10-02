import os
import logging
import datetime

# Set the logging level to capture errors and above
def setup_logger(run_id, run_name):
    logging.basicConfig(level=logging.ERROR)

    # Define a custom logging format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create a logging folder if it doesn't exist
    log_folder = "logs"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    # Create a logger
    LOGGER = logging.getLogger(__name__)

    # Create a file handler to save log messages to a file
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_folder, f"{run_name}_{run_id}.log'")
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    LOGGER.addHandler(file_handler)

    return LOGGER
