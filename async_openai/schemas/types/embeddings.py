from typing import List, Any, Optional, Union, Dict, Type
from async_openai.types import BaseModel, lazyproperty
from async_openai.schemas.types.base import BaseResult, Method, BaseEndpoint

__all__ = [
    'EmbeddingData',
    'EmbeddingRequest',
    'EmbeddingResult',
]


class EmbeddingData(BaseModel):
    object: Optional[str] = 'embedding'
    embedding: Optional[List[float]] = []
    index: Optional[int] = 0

class EmbeddingRequest(BaseModel):

    model: Optional[str]
    input: Optional[Union[List[Any], Any]]
    user: Optional[str] = None

    @property
    def create_embeddings_endpoint(self) -> BaseEndpoint:
        return BaseEndpoint(
            method = Method.POST,
            url = '/embeddings',
            data = self.dict(
                exclude_none = True
            )
        )


class EmbeddingResult(BaseResult):
    data: Optional[Union[EmbeddingData, List[EmbeddingData]]]
    _data_model: Optional[Type[EmbeddingData]] = EmbeddingData
    _request: Optional[EmbeddingRequest] = None

    @property
    def metadata_fields(self):
        return [
            'object',
            # 'data',
            'usage',
        ]


