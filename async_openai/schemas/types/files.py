
from typing import List, Any, Optional, Union, Dict, Type
from async_openai.schemas.types.base import BaseResult, FileData, FileRequest

__all__ = [
    'FileRequest',
    'FileData',
    'FileResult',
]

class FileResult(BaseResult):
    data: Optional[Union[FileData, List[FileData]]]
    _data_model: Optional[Type[FileData]] = FileData
    _request: Optional[FileRequest] = None

    @property
    def metadata_fields(self):
        return [
            'created',
            'object',
            # 'data',
        ]





