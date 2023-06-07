# -*- coding utf-8 -*-

"""
XTBApi.exceptions
~~~~~~~

Exception module
"""

import logging

LOGGER = logging.getLogger('root') # ('XTBApi.exceptions')


class CommandFailed(Exception):
    """when a command fail"""
    def __init__(self, response):
        self.msg = "command failed"
        self.err_code = response['errorCode']
        super().__init__(self.msg)


class NotLogged(Exception):
    """when not logged"""
    def __init__(self):
        self.msg = "Not logged, please log in"
        LOGGER.exception(self.msg)
        super().__init__(self.msg)


class SocketError(Exception):
    """when socket is already closed
    may be the case of server internal error"""
    def __init__(self):
        self.msg = "SocketError, mey be an internal error"
        LOGGER.error(self.msg)
        super().__init__(self.msg)


class TransactionRejected(Exception):
    """transaction rejected error"""
    def __init__(self, status_code):
        self.status_code = status_code
        self.msg = "transaction rejected with error code {}".format(
            status_code)
        LOGGER.error(self.msg)
        super().__init__(self.msg)
