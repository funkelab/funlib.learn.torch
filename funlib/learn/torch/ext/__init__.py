import logging
import sys
import traceback

logger = logging.getLogger(__name__)


class NoSuchModule(object):
    def __init__(self, name):
        self.__name = name
        self.__traceback_str = traceback.format_tb(sys.exc_info()[2])
        errtype, value = sys.exc_info()[:2]
        self.__exception = errtype(value)

    def __getattr__(self, item):
        raise self.__exception

try:
    import mlpack
except ImportError as e:
    mlpack = NoSuchModule('mlpack')
