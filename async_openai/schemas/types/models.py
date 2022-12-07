
from async_openai.types import BaseModel, lazyproperty
from typing import List, Any, Optional, Union, Type
from async_openai.schemas.types.base import Permission, BaseResult, Method, BaseEndpoint

__all__ = [
    'ModelData',
    'ModelRequest',
    'ModelResult',
]

class ModelData(BaseModel):
    id: str
    owned_by: str
    permission: List[Permission] = []
    root: str
    parent: Optional[str]
    object: str = 'model'

    @property
    def openai_id(self):
        return self.id

class ModelRequest(BaseModel):
    model: Optional[str]

    @lazyproperty
    def list_models_endpoint(self) -> BaseEndpoint:
        return BaseEndpoint(
            method = Method.GET,
            url ='/models',
        )

    @lazyproperty
    def retrieve_model_endpoint(self) -> BaseEndpoint:
        return BaseEndpoint(
            method = Method.GET,
            url = f'/models/{self.model}'
        )
    

class ModelResult(BaseResult):
    data: Optional[Union[ModelData, List[ModelData]]]
    _data_model: Optional[Type[ModelData]] = ModelData
    _request: Optional[ModelRequest] = None

    @property
    def metadata_fields(self):
        return [
            'object',
            # 'data',
        ]



