import logging

logging.basicConfig(
    filename="error_log.log",
    level=logging.ERROR,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

def logError(error):
    logging.error(error)
    raise ValueError(error)

def cleanText(text):
    bads = [" ", "'", "\"", "\n", "."]
    while text[0] in bads:
        text = text[1:]
    while text[-1] in bads:
        text = text[:-1]
    return text