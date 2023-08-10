
import inspect
import aiohttpx
from datetime import datetime, timedelta

from typing import Dict, Optional, Iterator, AsyncIterator, Union
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
    'parse_stream',
    'aparse_stream',
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
    """
    Removes trailing slash from a host if it exists.
    """
    return host[:-1] if host.endswith("/") else host


def parse_stream_line_bytes(line: bytes) -> Optional[str]:
    """
    Parse a line from a Server-Sent Events stream.
    """
    if line:
        if line.strip() == b"data: [DONE]":
            # return here will cause GeneratorExit exception in urllib3
            # and it will close http connection with TCP Reset
            return None
        if line.startswith(b"data: "):
            line = line[len(b"data: "):]
            return line.decode("utf-8")
        else:
            return None
    return None


def parse_stream_line_string(line: str) -> Optional[str]:
    """
    Parse a line from a Server-Sent Events stream.
    """
    if line:
        if line.strip() == "data: [DONE]":
            # return here will cause GeneratorExit exception in urllib3
            # and it will close http connection with TCP Reset
            return None
        return line[len("data: "):] if line.startswith("data: ") else None
    return None

def parse_stream_line(line: Union[str, bytes]) -> Optional[str]:
    """
    Parse a line from a Server-Sent Events stream.
    """
    if isinstance(line, bytes):
        return parse_stream_line_bytes(line)
    elif isinstance(line, str):
        return parse_stream_line_string(line)
    else:
        raise TypeError("line must be str or bytes")


def parse_stream(response: aiohttpx.Response) -> Iterator[str]:
    """
    Parse a Server-Sent Events stream.
    """
    for line in response.iter_lines():
        _line = parse_stream_line(line)
        if _line is not None:
            yield _line

async def aparse_stream(response: aiohttpx.Response) -> AsyncIterator[str]:
    """
    Parse a Server-Sent Events stream.
    """
    async for line in response.aiter_lines():
        _line = parse_stream_line(line)
        if _line is not None:
            yield _line