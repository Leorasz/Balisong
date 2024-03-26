import logging
from Balisong import logError

def logError(error):
    logging.error(error)
    raise ValueError(error)