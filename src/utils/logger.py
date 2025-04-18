import logging  
  
  
def create_logger(logger_name: str = __name__):  
    """  
    Create a logger for the module
    """  
    logger = logging.getLogger(logger_name)  
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()  
    file_handler = logging.FileHandler(f"./logs/{logger_name}")
    formatter = logging.Formatter(  
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"  
    )  
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    logger.propagate = False

    return logger
