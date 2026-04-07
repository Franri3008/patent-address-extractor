import logging
from pathlib import Path

import tqdm


class _TqdmStream(logging.StreamHandler):
    """Log handler that writes via tqdm.write() to avoid disrupting progress bars."""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.tqdm.write(self.format(record));
        except Exception:
            self.handleError(record);


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name);
    if logger.handlers:
        return logger;

    logger.setLevel(logging.INFO);
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    );

    ch = _TqdmStream();
    ch.setFormatter(fmt);
    logger.addHandler(ch);

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True);
        fh = logging.FileHandler(log_file, encoding="utf-8");
        fh.setFormatter(fmt);
        logger.addHandler(fh);

    return logger;
