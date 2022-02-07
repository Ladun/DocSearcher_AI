
import time


class TimeMeasure:
    def __init__(self, logger, prefix=""):
        self.time = 0
        self.logger = logger
        self.prefix = prefix

    def set_prefix(self, prefix):
        self.prefix = prefix

    def __enter__(self):
        self.time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.info(f"{self.prefix}{time.time() - self.time}")
