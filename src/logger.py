import os
import logging
import sys
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(log_dir, exist_ok=True)
LOG_FILE_PATH = os.path.join(log_dir, LOG_FILE)
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format='%(asctime)s -%(lineno)d- %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
if __name__ == "__main__":
    logging.info("Logger test running")
print("Logger executed")