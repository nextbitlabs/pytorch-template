import logging
import pathlib
from typing import Union


def initialize_logger(output_dir: Union[pathlib.Path, str]) -> pathlib.Path:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_formatter = logging.Formatter('%(asctime)s | %(message)s')

    logs_dir = pathlib.Path(output_dir).expanduser() / 'logs'
    logs_dir.mkdir()
    file_handler = logging.FileHandler(logs_dir / 'train_info.log')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    return logs_dir
