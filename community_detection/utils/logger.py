import logging
from pathlib import Path


def init_logger(log_path: Path, level=logging.INFO):
    """Initialize the logger to log messages to a specified file.

    Args:
        log_path (Path): Path to the log file.
        level: Logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    # Avoid duplicate handlers
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path) for h in logger.handlers):
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

    return logger


def log_header(title: str, width: int = 100):
    """
    Log a header message with surrounding bars for emphasis.

    Args:
        title (str): Title to log.
        width (int): Width of the header bar (default: 100).
    """
    logger = logging.getLogger()
    bar = "=" * width
    logger.info(f"\n{bar}\n{title.center(width)}\n{bar}")


def log_substep(message: str):
    """
    Log a substep message with an arrow prefix.

    Args:
        message (str): Substep message to log.  
    """
    logger = logging.getLogger()
    logger.info(f"--> {message}")
