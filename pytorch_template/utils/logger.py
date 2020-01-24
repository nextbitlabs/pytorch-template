import logging
import pathlib
from typing import Union, Optional, Dict

from torch.utils.tensorboard import SummaryWriter


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


def log_dictionary(data: Dict, writer: Optional[SummaryWriter] = None) -> None:
    for k, v in data.items():
        logging.info(f'{k}: {v}')
        if writer is not None:
            writer.add_text(str(k), str(v))
