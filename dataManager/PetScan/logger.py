import logging
import sys
from colorlog import ColoredFormatter

def setup_logger(name: str = "app", level: int = logging.INFO) -> logging.Logger:
    """
    Configure un logger coloré pour affichage dans le terminal.

    :param name: Nom du logger.
    :param level: Niveau de log à afficher.
    :return: logger object, ready to print
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)

        formatter = ColoredFormatter(
            fmt="%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG":    "cyan",
                "INFO":     "green",
                "WARNING":  "yellow",
                "ERROR":    "red",
                "CRITICAL": "bold_red",
            }
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
