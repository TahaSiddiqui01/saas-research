"""Lightweight logging helper to provide consistent, configurable logging across the project."""
import logging
import os


def setup_logger(name: str = "saas_research", level: str | None = None) -> logging.Logger:
    """Create and configure a logger.

    Level can be provided as a string (e.g., "INFO", "DEBUG") or read from
    the environment variable LOG_LEVEL. Defaults to INFO.
    """
    log_level = level or os.getenv("LOG_LEVEL", "INFO")
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    if not logger.handlers:
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)

    return logger
