#!/usr/bin/env

import os
import logging

def initialise_logger():

    logger_name = "tensorflow"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # create file handler that logs debug and higher level messages
    path = os.path.join(os.path.dirname(os.getcwd()) + "\\logs\\" + logger_name + ".log")
    fh = logging.FileHandler(path)
    fh.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s:%(levelname)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger