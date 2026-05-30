import logging
import sys

def setup_logger(name="DoTPFN", level=logging.INFO):
    """Sets up a clean formatted console logger for DoTPFN."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
        
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    
    logger.addHandler(ch)
    return logger
