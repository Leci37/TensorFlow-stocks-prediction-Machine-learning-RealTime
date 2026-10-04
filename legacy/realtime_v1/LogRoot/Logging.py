#-*-coding:utf-8-*-

# FORMAT_LOG = '%(asctime)s [%(threadName)s] %(name)s - [%(levelname)-5.5s] - %(funcName)s() - %(message)s'
# FILE_LOG_NAME = 'Log.log'
# LOG_LEVEL = logging.DEBUG

import logging.config
from os import path
LOGGING_CONFIG = 'LogRoot/logging.conf'



def singleton(cls):
    instances = {}
    def get_instance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    return get_instance()

@singleton
class Logger():
    def __init__(self):
        logging.config.fileConfig(LOGGING_CONFIG)
        self.logr = logging.getLogger('root')





