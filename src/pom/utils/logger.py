import os
import logging
import sys
import torch.distributed as dist

from enum import Enum

logger_initialized = {}


class CHATBOT(int, Enum):
    BOT = 22
    YOU = 21


def get_dist_info():
    initialized = dist.is_initialized()
    world_size = dist.get_world_size() if initialized else 1
    rank = dist.get_rank() if initialized else 0
    return rank, world_size


def get_logger(name='__main__', file_path=None, master_only=True):
    rank, _ = get_dist_info()
    name = f"{name}:{rank}"
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger

    if rank == 0:
        log_level = logging.INFO
    else:
        if master_only:
            log_level = logging.WARNING
        else:
            log_level = logging.INFO

    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    format_str = f'[rank{rank}]:[%(asctime)s] [%(levelname)s] - %(message)s'
    formatter = logging.Formatter(format_str, "%Y-%m-%d %H:%M:%S")

    if file_path:
        if rank == 0:
            os.makedirs(file_path, exist_ok=True)
            file_handler = logging.FileHandler(f"{file_path}/result.log", 'w')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(stream=sys.stdout)

    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)
    logger.setLevel(log_level)

    logger_initialized[name] = True

    return logger


def get_chat_logger(name='__main__', file_path=None):
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger

    log_level = logging.INFO

    logging.addLevelName(levelName='Chatbot', level=CHATBOT.BOT)
    logging.addLevelName(levelName='You', level=CHATBOT.YOU)

    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    format_str = f'[%(asctime)s]\n[%(levelname)s]:\t%(message)s\n' + '=' * 80
    formatter = logging.Formatter(format_str, "%Y-%m-%d %H:%M:%S")

    if file_path:
        os.makedirs(file_path, exist_ok=True)
        file_handler = logging.FileHandler(f"{file_path}/result.log", 'w')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(stream=sys.stdout)

    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)
    logger.setLevel(log_level)

    logger_initialized[name] = True

    return logger
