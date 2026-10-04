import logging.config
import os.path

from useless.XTBApi.__version__ import __version__

# LOGGING_CONFIG = r"C:\Users\Luis\Desktop\LecTrade\LecTrade\LogRoot\logging.conf"
# logging.config.fileConfig(LOGGING_CONFIG, disable_existing_loggers = False)

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'deafult': {
            'format':
                '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'deafult',
        },
        'rotating': {
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'formatter': 'deafult',
            'filename': os.path.join(
                os.path.dirname(__file__), 'logs/logfile.log'),
            'when': 'midnight',
            'backupCount': 3
        }
    },
    'loggers': {
        '': {
            'handlers': ['console'],
            'level': 'CRITICAL',
            'propagate': True
        },
        'XTBApi': {
            'handlers': ['rotating'],
            'level': 'DEBUG'
        }
    }
})
