"""Utility functions for WronAI."""

__all__ = ["download_file", "setup_logging"]

import logging
import os
import sys
import types
from pathlib import Path
from typing import Optional, Union

import requests
from loguru import logger


def download_file(
    url: str, output_path: Union[str, Path], chunk_size: int = 8192
) -> Path:
    """
    Download a file from a URL to the specified path.

    Args:
        url: URL of the file to download
        output_path: Path where the file will be saved
        chunk_size: Size of chunks to download at a time

    Returns:
        Path to the downloaded file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {url} to {output_path}")

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)

    return output_path


def setup_logging(
    level: Union[str, int] = "INFO", log_file: Optional[Union[str, Path]] = None
) -> None:
    """
    Configure logging with loguru.

    Args:
        level: Logging level (e.g., "INFO", "DEBUG")
        log_file: Optional path to log file
    """
    # Remove default handler
    logger.remove()

    # Add console handler
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stderr,
        level=level,
        format=log_format,
    )

    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            level=level,
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )

    # Set log level for other loggers
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Disable noisy loggers
    for name in ["transformers", "datasets", "httpx"]:
        logging.getLogger(name).setLevel(logging.WARNING)


class InterceptHandler(logging.Handler):
    """Intercept standard logging messages and forward them to loguru.

    This handler forwards all standard logging messages to the loguru logger.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record.

        Args:
            record: The log record to emit
        """
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated
        frame = sys._getframe(6)  # type: ignore[attr-defined]
        depth = 6
        # Skip frames from the logging module
        while True:
            if not hasattr(frame, "f_code") or not hasattr(frame.f_code, "co_filename"):
                break
            if frame.f_code.co_filename != logging.__file__:
                break
            if not hasattr(frame, "f_back"):
                break
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )
