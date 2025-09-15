import sys

from loguru import logger

# Configure the logger for the entire package
# By setting it up here, any module that imports `logger` from this file
# will get the same, pre-configured instance.


def setup_logging(level="INFO", colorize=True):
    """
    Configures the global logger for the application.

    This function removes the default loguru handler and adds a new one with
    a custom format to provide consistent, readable, and informative logs
    across the package.

    Args:
        level (str): The minimum logging level to output (e.g., "INFO", "DEBUG").
        colorize (bool): Whether to use colors in the log output.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        ),
        colorize=colorize,
    )


# Apply the default configuration when the module is imported.
# This ensures logging is ready to use without explicit setup in other modules.
setup_logging()

# The configured logger is now ready to be imported by other modules.
# Example usage in other files:
# from py_name_entity_recognition.observability.logging import logger
# logger.info("This is a configured log message.")
