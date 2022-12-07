
## Deprecated

from typing import List, Any, Optional, Union, Dict, Type
from async_openai.types import BaseModel
from async_openai.schemas.types.base import BaseResult

__all__ = [
    'EngineRequest',
    'EngineData',
    'EngineResult',
]

class EngineRequest(BaseModel):
    engine_id: Optional[str]

class EngineData(BaseModel):
    id: Optional[str]
    object: Optional[str]
    owner: Optional[str]
    ready: Optional[bool]

class EngineResult(BaseResult):
    data: Optional[Union[EngineData, List[EngineData]]]
    _data_model: Optional[Type[EngineData]] = EngineData
    _request: Optional[EngineRequest] = None