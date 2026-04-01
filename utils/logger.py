import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name);
    if logger.handlers:
        return logger;

    logger.setLevel(logging.INFO);
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    );

    ch = logging.StreamHandler(sys.stdout);
    ch.setFormatter(fmt);
    logger.addHandler(ch);

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True);
        fh = logging.FileHandler(log_file, encoding="utf-8");
        fh.setFormatter(fmt);
        logger.addHandler(fh);

    return logger;
