"""One consistent log format for scripts and notebooks."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def get_logger(name: str) -> logging.Logger:
    global _configured
    if not _configured:
        logging.basicConfig(level=logging.INFO, format=_FORMAT, datefmt="%H:%M:%S")
        _configured = True
    return logging.getLogger(name)
