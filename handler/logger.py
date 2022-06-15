import logging

from pathlib import Path
from datetime import datetime

class Logger:
    def __init__(
        self,
        log_dir: str = 'saved',
        log_mode = None,
        log_formatter = None
    ):
        super(Logger, self).__init__()
        time = datetime.now().strftime(r'%y%m%d%H%M')
        self.log_dir = f'{log_dir}/{time}'
        if not Path(self.log_dir).exists():
            Path(self.log_dir).mkdir(parents=True)
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
    