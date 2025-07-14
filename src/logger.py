import logging

# Configure basic logging to console
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def info(*args):
    logging.info(*args)
