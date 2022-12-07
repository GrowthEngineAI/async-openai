"""
Base Types from Typing
"""

__all__ = (
    'PathLike',
    'TYPE_CHECKING',
    'List', 'Dict', 'AnyStr', 'Any', 'Set',
    'Optional', 'Union', 'Tuple', 'Mapping', 'Sequence', 'TypeVar', 'Type',
    'Callable', 'Coroutine', 'Generator', 'AsyncGenerator', 'IO', 'Iterable', 'Iterator', 'AsyncIterator',
    'cast', 'overload',
    'Final', 'Literal',
    'Data', 'AnyMany', 'TextMany', 'TextList',
    'DictList', 'DictMany', 'DictAny', 'DictAny',
    'aobject',
)

import sys

from os import PathLike
from typing import TYPE_CHECKING
from typing import List, Dict, AnyStr, Any, Set
from typing import Optional, Union, Tuple, Mapping, Sequence, TypeVar, Type
from typing import Callable, Coroutine, Generator, AsyncGenerator, IO, Iterable, Iterator, AsyncIterator
from typing import cast, overload


if sys.version_info >= (3, 8):
    from typing import Final, Literal
else:
    from typing_extensions import Final, Literal

Data = TypeVar('Data', str, List[str], Dict[str, Union[str, List[str]]])
AnyMany = TypeVar('AnyMany', Any, List[Any])

TextMany = TypeVar('TextMany', str, List[str])
TextList = List[str]

DictList = List[Dict[str, Any]]
DictMany = TypeVar('DictMany', Dict[str, Any], List[Dict[str, Any]])
DictAny = Dict[str, Any]
DictText = Dict[str, str]
NoneType = type(None)



class aobject(object):
    """Inheriting this class allows you to define an async __init__.

    So you can create objects by doing something like `await MyClass(params)`
    """
    async def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        await instance.__init__(*args, **kwargs)
        return instance

    async def __init__(self, *args, **kwargs):
        pass