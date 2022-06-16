import logging


class Logger:
    def __init__(
        self,
        log_mode = None,
        log_formatter = None
    ):
        super(Logger, self).__init__()
        self.log_dir = None
        self.log_mode = log_mode
        self.log_formatter = log_formatter
    
    def get_logger(self, log_name):
        logger = logging.getLogger(log_name)
        logger.setLevel(self.log_mode)
        file_hand = logging.FileHandler(f'{self.log_dir}/file.log')
        formatter = logging.Formatter(f'{self.log_formatter}')
        file_hand.setFormatter(formatter)
        logger.addHandler(file_hand)
        return logger
    