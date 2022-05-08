#!/usr/bin/env python3

from __future__ import annotations

from . import spectrum
from . import theoretical_lines
__all__ = ['spectrum', 'theoretical_lines']

import logging


class CustomFormatter(logging.Formatter):
    green = "\x1b[32m"
    blue = "\x1b[34m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    fmt = "%(asctime)s %(name)s %(levelname)s - %(message)s"
    fmt = "%(levelname)s - %(filename)s:%(lineno)d %(funcName)s() - %(message)s"

    FORMATS = {
        logging.DEBUG: green + fmt + reset,
        logging.INFO: blue + fmt + reset,
        logging.WARNING: yellow + fmt + reset,
        logging.ERROR: red + fmt + reset,
        logging.CRITICAL: bold_red + fmt + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def basicConfig(level=logging.INFO):
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(CustomFormatter())
    logging.basicConfig(handlers=[ch], force=True)


stream = logging.StreamHandler()
stream.setFormatter(CustomFormatter())
logging.basicConfig(handlers=[stream], force=True)
logger = logging.getLogger(__name__)
