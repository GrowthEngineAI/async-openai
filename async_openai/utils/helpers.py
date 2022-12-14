
import inspect
from datetime import datetime, timedelta

from typing import Dict
from lazyops.utils.helpers import timed, timer, is_coro_func

__all__ = [
    'is_naive',
    'total_seconds',
    'remove_trailing_slash',
    'full_name',
    'merge_dicts',
    'is_coro_func',
    'timed',
    'timer',
]


def merge_dicts(x: Dict, y: Dict):
    z = x.copy()
    z.update(y)
    return z


def full_name(func, follow_wrapper_chains=True):
    """
    Return full name of `func` by adding the module and function name.

    If this function is decorated, attempt to unwrap it till the original function to use that
    function name by setting `follow_wrapper_chains` to True.
    """
    if follow_wrapper_chains: func = inspect.unwrap(func)
    return f'{func.__module__}.{func.__qualname__}'

def is_naive(dt: datetime):
    """Determines if a given datetime.datetime is naive."""
    return dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None


def total_seconds(delta: timedelta):
    """Determines total seconds with python < 2.7 compat."""
    # http://stackoverflow.com/questions/3694835/python-2-6-5-divide-timedelta-with-timedelta
    return (delta.microseconds + (delta.seconds + delta.days * 24 * 3600) * 1e6) / 1e6


def remove_trailing_slash(host: str):
    return host[:-1] if host.endswith("/") else host
