import inspect
from re import sub


def is_coro_func(obj, func_name: str = None):
    """
    This is probably in the library elsewhere but returns bool
    based on if the function is a coro
    """
    try:
        if inspect.iscoroutinefunction(obj): return True
        if inspect.isawaitable(obj): return True
        if func_name and hasattr(obj, func_name) and inspect.iscoroutinefunction(getattr(obj, func_name)):
            return True
        return bool(hasattr(obj, '__call__') and inspect.iscoroutinefunction(obj.__call__))
    except Exception:
        return False

def to_camel_case(text: str):
    """Convert a snake str to camel case."""
    components = text.split("_")
    # We capitalize the first letter of each component except the first one
    # with the 'title' method and join them together.
    return components[0] + "".join(x.title() for x in components[1:])

def to_snake_case(text: str):
    """
    multiMaster -> multi_master
    """
    return '_'.join(
        sub('([A-Z][a-z]+)', r' \1',
        sub('([A-Z]+)', r' \1',
        text.replace('-', ' '))).split()).lower()

def to_snake_case_args(text: str):
    """
    multiMaster -> multi-master
    """
    return '-'.join(
        sub('([A-Z][a-z]+)', r' \1',
        sub('([A-Z]+)', r' \1',
        text.replace('-', ' '))).split()).lower()
