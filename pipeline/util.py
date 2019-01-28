import os
import time
from contextlib import contextmanager
from loguru import logger as LOG


@contextmanager
def TLOG(name):
    t0 = time.time()
    LOG.info(f'{name} begin')
    try:
        yield
    finally:
        t1 = time.time()
        LOG.info('---- end {} cost {:.3f}s', name, t1 - t0)


try:
    profile
except NameError:
    def profile(f):
        return f


def get_config(name, default=None):
    value = os.getenv(name)
    if not value:
        value = default
    return value
