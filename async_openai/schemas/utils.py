from typing import Dict, Union, Optional
from async_openai.schemas.types.exceptions import APIError, TryAgain, RateLimitError


def fatal_exception(exc):
    if isinstance(exc, (APIError, TryAgain)):
        # retry on server errors and client errors
        # with 429 status code (rate limited),
        # don't retry on other client errors
        return (400 <= exc.exc.status_code < 500) and exc.exc.status_code != 429
    else:
        # retry on all other errors (eg. network)
        return False


def build_proxies(
    proxy: Optional[Union[str, Dict]] = None,
):
    if proxy is None:
        return None
    elif isinstance(proxy, str):
        return {"http://": proxy, "https://": proxy}
    elif isinstance(proxy, dict):
        return proxy.copy()
    else:
        raise TypeError("Proxy must be a string or a dictionary")


