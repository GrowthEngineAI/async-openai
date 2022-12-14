from typing import Optional, Type, Any, Union, List
from lazyops.types import validator, lazyproperty

from async_openai.types.options import OpenAIModel, OpenAIModelType
from async_openai.types.resources import BaseResource
from async_openai.types.responses import BaseResponse
from async_openai.types.routes import BaseRoute


__all__ = [
    'EmbeddingData',
    'EmbeddingObject',
    'EmbeddingResponse',
    'EmbeddingRoute',
]


class EmbeddingData(BaseResource):
    object: Optional[str] = 'embedding'
    embedding: Optional[List[float]] = []
    index: Optional[int] = 0

class EmbeddingObject(BaseResource):
    model: Optional[Union[str, OpenAIModel, Any]] = OpenAIModelType.curie
    input: Optional[Union[List[Any], Any]]
    user: Optional[str] = None

    @validator('model', pre=True, always=True)
    def validate_model(cls, v) -> OpenAIModel:
        """
        Validate the model
        """
        return OpenAIModel(value = v, mode = 'embedding')


class EmbeddingResponse(BaseResponse):
    data: Optional[List[EmbeddingData]]
    data_model: Optional[Type[BaseResource]] = EmbeddingData

    @lazyproperty
    def embeddings(self) -> List:
        """
        Returns the text for the response
        object
        """
        if self.data:
            return [data.embedding for data in self.data] if self.data else []
        return None


class EmbeddingRoute(BaseRoute):
    input_model: Optional[Type[BaseResource]] = EmbeddingObject
    response_model: Optional[Type[BaseResource]] = EmbeddingResponse

    @lazyproperty
    def api_resource(self):
        return 'embeddings'


