from pydantic import Field
from pydantic import BaseModel as _BaseModel
from typing import Any

from async_openai.types.formatting import to_snake_case

class BaseModel(_BaseModel):
    class Config:
        extra = 'allow'
        arbitrary_types_allowed = True
        alias_generator = to_snake_case

    def get(self, name, default: Any = None):
        return getattr(self, name, default)


class Schema(BaseModel):

    class Config:
        extra = 'allow'
        arbitrary_types_allowed = True
        

    def get(self, name, default: Any = None):
        return getattr(self, name, default)


__all__ = [
    'Field',
    'BaseModel',
    'Schema',
]