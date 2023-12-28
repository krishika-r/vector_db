# logger.py
import logging
from logging import FileHandler
import yaml
from utils import config_init, load_config
import os


def setup_logger(data_config_path = "../../configs/data_config.yaml"):

    data_config = load_config(data_config_path)
    
    log_path =  data_config["path"]["logger"]["path"]
    log_file =  data_config["path"]["logger"]["file_name"]

    

    
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    print(log_path)
    log_file_path = os.path.join(log_path, log_file)
    
    print(log_file_path)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a file handler and set the level to debug
    fh = FileHandler(log_file_path, 'w+')
    fh.setLevel(logging.DEBUG)

    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    return logger

# Call the setup_logger function to get a logger instance
logger = setup_logger()
