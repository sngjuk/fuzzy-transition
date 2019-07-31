'''
Author : Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>
'''
import argparse
import logging

__all__ = ['set_logger', 'get_args_parser']


def set_logger(context, verbose=False):

    logger = logging.getLogger(context)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        '%(levelname)-.1s:' + context + ':[%(filename).3s:%(funcName).3s:%(lineno)3d]:%(message)s',
        datefmt='%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(console_handler)
    return logger


def get_args_parser():
    parser = argparse.ArgumentParser(description='arguments for FuzzyServer')

    parser.add_argument('-m', '--model_path', type=str, help='Path of trained model, e.g ./model.bin')
    parser.add_argument('-p', '--port', type=str, default='5555', help='opening port number')
    parser.add_argument('-t', '--thread_num', type=int, default=4, help='number of thread on server.')
    args = parser.parse_args()

    return args
