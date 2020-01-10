import logging
import pathlib
from typing import Union


def initialize_logger(output_dir: Union[None, pathlib.Path, str] = None) -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_formatter = logging.Formatter('%(asctime)s | %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    if output_dir is not None:
        logs_dir = pathlib.Path(output_dir).expanduser() / 'logs'
        logs_dir.mkdir()
        file_handler = logging.FileHandler(logs_dir / 'train_info.log')
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
