import datetime
from typing import Optional, Type, List, Dict, Any
from lazyops.types import lazyproperty, Field

from async_openai.types.resources import BaseResource, Permission
from async_openai.types.responses import BaseResponse
from async_openai.types.routes import BaseRoute


__all__ = [
    'ModelData',
    'ModelObject',
    'ModelResponse',
    'ModelRoute',
]


class ModelData(BaseResource):
    id: str
    status: Optional[str] = None
    owned_by: Optional[str] = None
    created: Optional[datetime.datetime] = None
    updated_at: Optional[datetime.datetime] = None
    created_at: Optional[datetime.datetime] = None
    permission: Optional[List[Permission]] = Field(default_factory = list)
    root: Optional[str] = None
    parent: Optional[str] = None
    object: Optional[str] = 'model'
    capabilities: Optional[Dict[str, Any]] = Field(default_factory = dict)
    lifecycle_status: Optional[str] = None
    deprecation: Optional[Dict[str, Any]] = Field(default_factory = dict)


    @lazyproperty
    def model_age(self) -> Optional[datetime.datetime]:
        """
        Returns how long ago the model was created
        """
        if self.created:
            return datetime.datetime.now(tz = datetime.timezone.utc) - self.created


class ModelObject(BaseResource):
    model: Optional[str]


class ModelResponse(BaseResponse):
    data: Optional[List[ModelData]]
    data_model: Optional[Type[BaseResource]] = ModelData

    @lazyproperty
    def model_list(self) -> List[str]:
        return [model.id for model in self.data] if self.data and isinstance(self.data, list) else []


class ModelRoute(BaseRoute):
    input_model: Optional[Type[BaseResource]] = ModelObject
    response_model: Optional[Type[BaseResource]] = ModelResponse

    @lazyproperty
    def api_resource(self):
        return 'models'
    
    @lazyproperty
    def create_enabled(self):
        return False
    
    @lazyproperty
    def list_enabled(self):
        return True
    
    @lazyproperty
    def get_enabled(self):
        return True

    def retrieve(
        self, 
        resource_id: str, 
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Retrieve a Single Model by Resource ID

        :param resource_id: The ID of the Resource to GET
        :param params: Optional Query Parameters
        """
        return super().retrieve(resource_id = resource_id, params = params, **kwargs)
    
    async def async_retrieve(
        self,
        resource_id: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Retrieve a Single Model by Resource ID

        :param resource_id: The ID of the Resource to GET
        :param param
        """
        return await super().async_retrieve(resource_id = resource_id, params = params, **kwargs)


    def list(
        self, 
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ModelResponse:
        """
        List all available Models

        :param params: Optional Query Parameters
        """
        return super().list(params = params, **kwargs)
    
    async def async_list(
        self, 
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ModelResponse:
        """
        List all available Models

        :param params: Optional Query Parameters
        """
        return await super().async_list(params = params, **kwargs)

    



