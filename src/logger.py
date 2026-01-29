import logging
import os

def setup_debug_logger(
    log_dir: str,
    log_name: str = "google_search.log",
    level=logging.DEBUG,
):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)

    logger = logging.getLogger(log_name)
    logger.setLevel(level)
    logger.propagate = False 

    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(level)

        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s"
        )
        fh.setFormatter(formatter)

        logger.addHandler(fh)

    return logger
