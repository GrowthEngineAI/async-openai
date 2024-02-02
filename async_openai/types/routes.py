import json
import aiohttpx
import backoff
import functools
from lazyops.types import BaseModel, Field, lazyproperty
from lazyops.utils import ObjectEncoder
from typing import Dict, Optional, Any, List, Type, Callable, Union

from async_openai.utils.logs import logger
from async_openai.utils.config import get_settings, get_default_headers, get_max_retries, OpenAISettings, AzureOpenAISettings
from async_openai.types.errors import fatal_exception, error_handler
from async_openai.types.resources import BaseResource, FileResource
from async_openai.types.responses import BaseResponse

__all__ = [
    'BaseRoute',
    'RESPONSE_SUCCESS_CODES',
]


"""
Route Class
"""

RESPONSE_SUCCESS_CODES = [
    200, 
    201, 
    202, 
    204
]

_retry_wrapper: Optional[Callable] = None

def get_retry_wrapper(
    max_retries: int
):
    """
    Creates the retryable wrapper
    """
    global _retry_wrapper
    if _retry_wrapper is None:
        _retry_wrapper = functools.partial(
            backoff.on_exception,
            backoff.expo, 
            exception = Exception, 
            giveup = fatal_exception
        )
    return _retry_wrapper(max_tries = max_retries + 1)


class BaseRoute(BaseModel):
    client: aiohttpx.Client
    name: str # Client Name
    # headers: Dict[str, str] = Field(default_factory = get_default_headers)
    success_codes: Optional[List[int]] = RESPONSE_SUCCESS_CODES
    
    input_model: Optional[Type[BaseResource]] = None
    response_model: Optional[Type[BaseResponse]] = None

    # Options
    timeout: Optional[int] = None
    debug_enabled: Optional[bool] = False
    on_error: Optional[Callable] = None
    ignore_errors: Optional[bool] = False
    disable_retries: Optional[bool] = None
    max_retries: Optional[int] = None
    retry_function: Optional[Callable] = None # Allow for customized retry functions

    settings: Optional[Union[OpenAISettings, AzureOpenAISettings]] = Field(default_factory = get_settings)
    is_azure: Optional[bool] = None
    azure_model_mapping: Optional[Dict[str, str]] = None

    client_callbacks: Optional[List[Callable]] = None

    @lazyproperty
    def api_resource(self):
        """
        Returns the API Resource
        """
        return ''
    
    """
    Enable or Disable Routes
    """
    @lazyproperty
    def create_enabled(self):
        return True
    
    @lazyproperty
    def create_batch_enabled(self):
        return False
    
    @lazyproperty
    def get_enabled(self):
        return False

    @lazyproperty
    def list_enabled(self):
        return False

    @lazyproperty
    def update_enabled(self):
        return False
    
    @lazyproperty
    def delete_enabled(self):
        return False


    @lazyproperty
    def download_enabled(self):
        return False
    
    @lazyproperty
    def upload_enabled(self):
        return False
    
    @lazyproperty
    def exclude_null(self) -> bool:
        return True
    

    @lazyproperty
    def usage_enabled(self):
        return False
    
    def get_resource_url(self, data: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        """
        Returns the Resource URL from the Response Data
        """
        if self.is_azure is None: self.is_azure = isinstance(self.settings, AzureOpenAISettings)
        if not self.is_azure: return self.api_resource
        data = data or {}
        # logger.info(f"Data: {data}")
        base_endpoint = '/openai'
        deployment = data.get('deployment', data.get('model', kwargs.get('deployment', kwargs.get('model'))))
        if deployment is not None:
            base_endpoint += f'/deployments/{deployment}'
        base_endpoint += f'/{self.api_resource}'
        if api_version := data.get('api_version', kwargs.get('api_version', self.settings.api_version)):
            base_endpoint += f'?api-version={api_version}'
        return base_endpoint


    def encode_data(self, data: Dict[str, Any]) -> str:
        """
        Encodes the data
        """
        # if self.is_azure:
        #     _ = data.pop('model', None)
        #     _ = data.pop('deployment', None)
        return json.dumps(
            data,
            cls = ObjectEncoder
        )

    def create(
        self, 
        input_object: Optional[Type[BaseResource]] = None,
        headers: Optional[Dict[str, str]] = None,
        parse_stream: Optional[bool] = True,
        timeout: Optional[int] = None,
        **kwargs
    ):
        """
        Create a Resource

        :param input_object: Input Object to Create
        """
        if not self.create_enabled:
            raise NotImplementedError(f'Create is not enabled for {self.api_resource}')
        
        if input_object is None: 
            input_object, kwargs = self.input_model.create_resource(
                resource = self.input_model,
                **kwargs
            )
        # data = get_pyd_dict(input_object, exclude_none = self.exclude_null)
        data = input_object.dict(exclude_none = self.exclude_null)
        api_response = self._send(
            method = 'POST',
            url = self.get_resource_url(data = data, **kwargs),
            data = self.encode_data(data),
            headers = headers,
            timeout = timeout if timeout is not None else self.timeout,
            stream = input_object.get('stream'),
            **kwargs
        )
        data = self.handle_response(api_response)
        return self.prepare_response(data, input_object = input_object, parse_stream = parse_stream)
    

    async def async_create(
        self, 
        input_object: Optional[Type[BaseResource]] = None,
        headers: Optional[Dict[str, str]] = None,
        parse_stream: Optional[bool] = True,
        timeout: Optional[int] = None,
        **kwargs
    ):
        """
        Create a Resource

        :param input_object: Input Object to Create
        """
        if not self.create_enabled:
            raise NotImplementedError(f'Create is not enabled for {self.api_resource}')
        
        if input_object is None:
            input_object, kwargs = self.input_model.create_resource(
                resource = self.input_model,
                **kwargs
            )
        # data = get_pyd_dict(input_object, exclude_none = self.exclude_null)
        data = input_object.dict(exclude_none = self.exclude_null)
        api_response = await self._async_send(
            method = 'POST',
            url = self.get_resource_url(data = data, **kwargs),
            data = self.encode_data(data),
            headers = headers,
            timeout = timeout if timeout is not None else self.timeout,
            stream = input_object.get('stream'),
            **kwargs
        )
        # if input_object.get('stream'):
        #     await api_response.aread()
        data = self.handle_response(api_response)
        return await self.aprepare_response(data, input_object = input_object, parse_stream = parse_stream)
    
    acreate = async_create
    
    def batch_create(
        self, 
        input_object: Optional[Type[BaseResource]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Batch Create Resources

        :param input_object: Input Object to Create
        :param return_response: Return the Response Object
        """
        if not self.create_batch_enabled:
            raise NotImplementedError(f'Batch Create is not enabled for {self.api_resource}')
        
        if input_object is None:
            input_object, kwargs = self.input_model.create_resource(
                resource = self.input_model,
                **kwargs
            )

        api_resource = f'{self.api_resource}/batch'
        data = json.dumps(
            # get_pyd_dict(input_object, exclude_none = self.exclude_null),
            input_object.dict(exclude_none = self.exclude_null), 
            cls = ObjectEncoder
        )
        api_response = self._send(
            method = 'POST',
            url = self.get_resource_url(data = data),
            data = data,
            headers = headers,
            timeout = self.timeout,
            **kwargs
        )
        resp = self.handle_response(api_response)
        return self.prepare_response(resp, input_object = input_object)

    async def async_batch_create(
        self, 
        input_object: Optional[Type[BaseResource]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Batch Create Resources

        :param input_object: Input Object to Create
        :param return_response: Return the Response Object
        """
        if not self.create_batch_enabled:
            raise NotImplementedError(f'Batch Create is not enabled for {self.api_resource}')
        
        if input_object is None:
            input_object, kwargs = self.input_model.create_resource(
                resource = self.input_model,
                **kwargs
            )

        api_resource = f'{self.api_resource}/batch'
        data = json.dumps(
            # get_pyd_dict(input_object, exclude_none = self.exclude_null),
            input_object.dict(exclude_none = self.exclude_null), 
            cls = ObjectEncoder
        )
        api_response = await self._async_send(  
            method = 'POST',
            url = api_resource,
            data = data,
            headers = headers,
            timeout = self.timeout,
            **kwargs
        )
        resp = self.handle_response(api_response)
        return await self.aprepare_response(resp, input_object = input_object)
    
    abatch_create = async_batch_create

    
    def retrieve(
        self, 
        resource_id: str, 
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Type[BaseResource]:
        """
        GET a Single Resource

        :param resource_id: The ID of the Resource to GET
        :param params: Optional Query Parameters
        """
        if not self.get_enabled:
            raise NotImplementedError(f'Get is not enabled for {self.api_resource}')
        
        api_resource = f'{self.api_resource}/{resource_id}'
        api_response = self._send(
            method = 'GET',
            url = api_resource, 
            params = params,
            headers = headers,
            **kwargs
        )
        data = self.handle_response(api_response)
        return self.prepare_response(data)
    
    async def async_retrieve(
        self,
        resource_id: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Type[BaseResource]:
        """
        GET a Single Resource

        :param resource_id: The ID of the Resource to GET
        :param params: Optional Query Parameters
        """
        if not self.get_enabled:
            raise NotImplementedError(f'Get is not enabled for {self.api_resource}')
        
        api_resource = f'{self.api_resource}/{resource_id}'
        api_response = await self._async_send(
            method = 'GET',
            url = api_resource, 
            params = params,
            headers = headers,
            timeout = self.timeout,
            **kwargs
        )
        data = self.handle_response(api_response)
        return self.prepare_response(data)
    
    aretrieve = async_retrieve

    def get(
        self, 
        resource_id: str, 
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Type[BaseResource]:
        """
        GET a Single Resource

        :param resource_id: The ID of the Resource to GET
        :param params: Optional Query Parameters
        """
        return self.retrieve(resource_id = resource_id, params = params, headers = headers, **kwargs)
    
    async def async_get(
        self,
        resource_id: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Type[BaseResource]:
        """
        GET a Single Resource

        :param resource_id: The ID of the Resource to GET
        :param params: Optional Query Parameters
        """
        return await self.async_retrieve(resource_id = resource_id, params = params, headers = headers, **kwargs)

    aget = async_get
    
    def list(
        self, 
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Union[List[Type[BaseResource]], Dict[str, Any]]]:
        """
        GET all available objects of Resource

        :param params: Optional Query Parameters
        
        :return: Dict[str, Union[List[Type[BaseResource]], Dict[str, Any]]]
        """
        if not self.list_enabled:
            raise NotImplementedError(f'List is not enabled for {self.api_resource}')
        
        api_response = self._send(
            method = 'GET',
            url = self.get_resource_url(data = None, **kwargs),
            params = params,
            headers = headers,
            timeout = self.timeout,
            **kwargs
        )
        data = self.handle_response(api_response)
        return self.prepare_response(data)
        #return self.prepare_index_response(data)

    async def async_list(
        self, 
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Union[List[Type[BaseResource]], Dict[str, Any]]]:
        """
        GET all available objects of Resource

        :param params: Optional Query Parameters

        :return: Dict[str, Union[List[Type[BaseResource]], Dict[str, Any]]]
        """
        if not self.list_enabled:
            raise NotImplementedError(f'List is not enabled for {self.api_resource}')
        
        api_response = await self._async_send(
            method = 'GET',
            url = self.get_resource_url(data = None, **kwargs),
            params = params,
            headers = headers,
            timeout = self.timeout,
            **kwargs
        )
        data = self.handle_response(api_response)
        return await self.aprepare_response(data)
    
    alist = async_list

    def get_all(
        self, 
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Union[List[Type[BaseResource]], Dict[str, Any]]]:
        """
        GET all available objects of Resource

        :param params: Optional Query Parameters
        
        :return: Dict[str, Union[List[Type[BaseResource]], Dict[str, Any]]]
        """
        return self.retrieve(params = params, **kwargs)

    async def async_get_all(
        self, 
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Union[List[Type[BaseResource]], Dict[str, Any]]]:
        """
        GET all available objects of Resource

        :param params: Optional Query Parameters

        :return: Dict[str, Union[List[Type[BaseResource]], Dict[str, Any]]]
        """
        return await self.async_retrieve(params = params, **kwargs)
    
    aget_all = async_get_all

    def delete(
        self, 
        resource_id: str,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        DELETE a Resource

        :param resource_id: The ID of the Resource to DELETE
        """
        if not self.delete_enabled:
            raise NotImplementedError(f'Delete is not enabled for {self.api_resource}')
        
        api_resource = f'{self.api_resource}/{resource_id}'
        api_response = self._send(
            method = 'DELETE',
            url = api_resource,
            headers = headers,
            timeout = self.timeout,
            **kwargs
        )
        data = self.handle_response(api_response)
        return self.prepare_response(data)
    
    async def async_delete(
        self, 
        resource_id: str,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        DELETE a Resource

        :param resource_id: The ID of the Resource to DELETE
        """
        if not self.delete_enabled:
            raise NotImplementedError(f'Delete is not enabled for {self.api_resource}')
        
        api_resource = f'{self.api_resource}/{resource_id}'
        api_response = await self._async_send(
            method = 'DELETE',
            url = api_resource,
            headers = headers,
            timeout = self.timeout,
            **kwargs
        )
        data = self.handle_response(api_response)
        return self.prepare_response(data)

    adelete = async_delete

    def update(
        self, 
        input_object: Optional[Type[BaseResource]] = None,
        resource_id: str = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Update a Resource

        :param input_object: Input Object to Update
        :param resource_id: The ID of the Resource to Update
        """
        if not self.update_enabled:
            raise NotImplementedError(f'Update is not enabled for {self.api_resource}')
        
        if input_object is None:
            input_object, kwargs = self.input_model.create_resource(
                resource = self.input_model,
                **kwargs
            )
        
        api_resource = self.api_resource

        resource_id = resource_id or input_object.resource_id
        if resource_id is not None:
            api_resource = f'{api_resource}/{resource_id}'

        data = json.dumps(
            input_object.dict(exclude_none = self.exclude_null), 
            cls = ObjectEncoder
        )
        api_response = self._send(
            method = 'PUT',
            url = api_resource,
            data = data,
            headers = headers,

            timeout = self.timeout,
            **kwargs
        )
        data = self.handle_response(api_response)
        return self.prepare_response(data, input_object = input_object)

    async def async_update(
        self, 
        input_object: Optional[Type[BaseResource]] = None,
        resource_id: str = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Update a Resource

        :param input_object: Input Object to Update
        :param resource_id: The ID of the Resource to Update
        """
        if not self.update_enabled:
            raise NotImplementedError(f'Update is not enabled for {self.api_resource}')
        
        if input_object is None: 
            input_object, kwargs = self.input_model.create_resource(
                resource = self.input_model,
                **kwargs
            )
        
        api_resource = self.api_resource
        resource_id = resource_id or input_object.resource_id
        if resource_id is not None:
            api_resource = f'{api_resource}/{resource_id}'
        data = json.dumps(
            input_object.dict(exclude_none = self.exclude_null), 
            cls = ObjectEncoder
        )
        api_response = await self._async_send(
            method = 'PUT',
            url = api_resource,
            data = data,
            headers = headers,
            timeout = self.timeout,
            **kwargs
        )
        data = self.handle_response(api_response)
        return self.prepare_response(data, input_object = input_object)

    aupdate = async_update

    """
    Extra Methods
    """

    def exists(
        self,
        resource_id: str,
        **kwargs
    ) -> bool:
        """
        See whether a Resource Exists

        :param resource_id: The ID of the Resource to Valid
        """
        try:
            return self.get(resource_id = resource_id, **kwargs)
        except Exception:
            return False
    
    async def async_exists(
        self,
        resource_id: str,
        **kwargs
    ) -> bool:
        """
        See whether a Resource Exists

        :param resource_id: The ID of the Resource to Valid
        """
        try:
            return await self.async_get(resource_id = resource_id, **kwargs)
        except Exception:
            return False
        
    aexists = async_exists
    
    def upsert(
        self,
        resource_id: str,
        input_object: Optional[Type[BaseResource]] = None,
        update_existing: bool = False, 
        overwrite_existing: bool = False,
        **kwargs
    ):
        """
        Upsert a Resource
        Validates whether the Resource exists, and if it does, updates it. 
        If it doesn't, creates it.
        If update_existing is True, it will always update the Resource
        If overwrite_existing is True, it will re-create the Resource

        :resource_id: The ID of the Resource to Upsert
        :param input_object: Input Object to Upsert
        :param update_existing (bool): Whether to update the Resource if it exists
        :overwrite_existing (bool): Whether to overwrite the Resource if it exists
        """
        resource = self.exists(resource_id = resource_id, **kwargs)
        if resource is not None:
            if update_existing:
                return self.update(input_object = input_object, identifier = resource_id, **kwargs)
            if overwrite_existing:
                self.delete(resource_id = resource_id, **kwargs)
                return self.create(input_object = input_object, **kwargs)
            return resource
        return self.create(input_object = input_object, **kwargs)
    
    async def async_upsert(
        self,
        resource_id: str,
        input_object: Optional[Type[BaseResource]] = None,
        update_existing: bool = False, 
        overwrite_existing: bool = False,
        **kwargs
    ):
        """
        Upsert a Resource
        Validates whether the Resource exists, and if it does, updates it. 
        If it doesn't, creates it.
        If update_existing is True, it will always update the Resource
        If overwrite_existing is True, it will re-create the Resource

        :resource_id: The ID of the Resource to Upsert
        :param input_object: Input Object to Upsert
        :param update_existing (bool): Whether to update the Resource if it exists
        :overwrite_existing (bool): Whether to overwrite the Resource if it exists
        """
        resource = await self.async_exists(resource_id = resource_id, **kwargs)
        if resource is not None:
            if update_existing:
                return self.async_update(input_object = input_object, identifier = resource_id, **kwargs)
            if overwrite_existing:
                await self.async_delete(resource_id = resource_id, **kwargs)
                return await self.async_create(input_object = input_object, **kwargs)
            return resource
        return await self.async_create(input_object = input_object, **kwargs)

    aupsert = async_upsert

    def upload(
        self, 
        input_object: Optional[Type[FileResource]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """
        Upload a Resource

        :param input_object: Input Object to Create
        """
        if not self.upload_enabled:
            raise NotImplementedError(f'Upload is not enabled for {self.api_resource}')
        
        if input_object is None: 
            input_object, kwargs = self.input_model.create_resource(
                resource = self.input_model,
                **kwargs
            )
        
        # headers = self.headers.copy()
        headers = headers or {}
        headers['Content-Type'] = 'multipart/form-data'

        api_resource = self.api_resource
        api_response = self._send(
            method = 'POST',
            url = api_resource,
            files = input_object.get_params(**kwargs),
            headers = headers,
            timeout = self.timeout,
            **kwargs
        )
        data = self.handle_response(api_response)
        return self.prepare_response(data, input_object = input_object)
        
    async def async_upload(
        self, 
        input_object: Optional[Type[FileResource]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """
        Upload a Resource

        :param input_object: Input Object to Create
        """
        if not self.upload_enabled:
            raise NotImplementedError(f'Upload is not enabled for {self.api_resource}')

        if input_object is None: 
            input_object, kwargs = self.input_model.create_resource(
                resource = self.input_model,
                **kwargs
            )
        
        # headers = self.headers.copy()
        headers = headers or {}
        headers['Content-Type'] = 'multipart/form-data'

        api_resource = self.api_resource
        api_response = await self._async_send(
            method = 'POST',
            url = api_resource,
            files = input_object.get_params(**kwargs),
            headers = headers,
            timeout = self.timeout,
            **kwargs
        )
        data = self.handle_response(api_response)
        return self.prepare_response(data, input_object = input_object)

    aupload = async_upload

    def download(
        self, 
        resource_id: str,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Download a Resource

        :param resource_id: The ID of the Resource to Download
        """
        if not self.download_enabled:
            raise NotImplementedError(f'Download is not enabled for {self.api_resource}')
        
        api_resource = f'{self.api_resource}/{resource_id}/download'
        api_response = self._send(
            method = 'POST',
            url = api_resource, 
            headers = headers,
            timeout = self.timeout,
            **kwargs
        )
        data = self.handle_response(api_response)
        return self.prepare_response(data)

    async def async_download(
        self, 
        resource_id: str,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Download a Resource

        :param resource_id: The ID of the Resource to Download
        """
        if not self.download_enabled:
            raise NotImplementedError(f'Download is not enabled for {self.api_resource}')
        
        api_resource = f'{self.api_resource}/{resource_id}/download'
        api_response = await self._async_send(
            method = 'POST',
            url = api_resource, 
            headers = headers,
            timeout = self.timeout,
            **kwargs
        )
        data = self.handle_response(api_response)
        return self.prepare_response(data)

    adownload = async_download

    def prepare_response(
        self, 
        data: aiohttpx.Response,
        input_object: Optional[Type[BaseResource]] = None,
        response_object: Optional[Type[BaseResource]] = None,
        parse_stream: Optional[bool] = True,
        **kwargs
    ):
        """
        Prepare the Response Object
        
        :param data: The Response Data
        :param response_object: The Response Object
        """
        response_object = response_object or self.response_model
        if response_object:
            response = response_object.prepare_response(data, input_object = input_object, parse_stream = parse_stream)
            self.handle_callbacks(response, **kwargs)
            return response
        raise NotImplementedError('Response model not defined for this resource.')

    async def aprepare_response(
        self, 
        data: aiohttpx.Response,
        input_object: Optional[Type[BaseResource]] = None,
        response_object: Optional[Type[BaseResource]] = None,
        parse_stream: Optional[bool] = True,
        **kwargs
    ):
        """
        Prepare the Response Object
        
        :param data: The Response Data
        :param response_object: The Response Object
        """
        response_object = response_object or self.response_model
        if response_object:
            response = await response_object.aprepare_response(data, input_object = input_object, parse_stream = parse_stream)
            self.handle_callbacks(response, **kwargs)
            return response
        raise NotImplementedError('Response model not defined for this resource.')

    def handle_callbacks(
        self,
        response_object: BaseResource,
        **kwargs
    ):
        """
        Handle the Callbacks for the Response as a Background Task

        This is useful for when you want to run a background task after a response is received

        The callback should be a function that takes the response object as the first argument
        """
        if self.client_callbacks:
            from lazyops.libs.pooler import ThreadPooler
            for callback in self.client_callbacks:
                ThreadPooler.background(callback, response_object, **kwargs)
        
    def handle_response(
        self, 
        response: aiohttpx.Response,
        **kwargs
    ):
        """
        Handle the Response

        :param response: The Response
        """
        is_stream = not hasattr(response, "_content")
        if self.debug_enabled:
            if is_stream:
                logger.info(f'[{self.name} - Stream] [{response.status_code} - {response.request.url}] headers: {response.headers}')
            else:
                logger.info(f'[{self.name} - {response.status_code} - {response.request.url}] headers: {response.headers}, body: {response.text[:250]}')
        
        if response.status_code in self.success_codes:
            if is_stream: return response
            return response if response.content else None
        
        if self.ignore_errors: return None
        raise error_handler(
            response = response,
            data = response.content,
            **kwargs
        )
    
    def _send(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        stream: Optional[bool] = False,
        **kwargs
    ) -> aiohttpx.Response:
        """
        Handle Sending Requests
        """
        if timeout is None: timeout = self.timeout
        if self.debug_enabled: logger.info(f'[{self.name} - {method} - {url}] headers: {headers}, params: {params}, data: {data}')
        request = self.client.build_request(
            method = method,
            url = url,
            params = params,
            data = data,
            headers = headers,
            timeout = timeout,
            **kwargs
        )
        request_func = self.client.send
        if self.retry_function is not None:
            request_func = self.retry_function(request_func)
        elif not self.disable_retries:
            if max_retries is None: max_retries = get_max_retries()
            request_func = get_retry_wrapper(max_retries=max_retries)(request_func)
        return request_func(request, stream = stream)

    
    async def _async_send(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        stream: Optional[bool] = False,
        **kwargs
    ) -> aiohttpx.Response:
        """
        Handle Sending Requests
        """
        
        if timeout is None: timeout = self.timeout
        # if self.debug_enabled: logger.info(f'[{self.name} - {method} - {url}] headers: {headers}, params: {params}, data: {data}')
        request = await self.client.async_build_request(
            method = method,
            url = url,
            params = params,
            data = data,
            headers = headers,
            timeout = timeout,
            **kwargs
        )
        if self.debug_enabled: logger.info(f'[{self.name} - {method} - {url}] headers: {request.headers}, params: {params}, data: {data}')
        request_func = self.client.async_send
        if self.retry_function is not None:
            request_func = self.retry_function(request_func)
        elif not self.disable_retries:
            if max_retries is None: max_retries = get_max_retries()
            request_func = get_retry_wrapper(max_retries = max_retries)(request_func)
        return await request_func(request, stream = stream)
    
    def prepare_index_response(
        self, 
        data: Dict[str, Any],
        response_object: Optional[Type[BaseResource]] = None,
        **kwargs
    ):
        """
        Prepare the Response Object for Index Requests

        :param data: The Response Data
        :param response_object: The Response Object
        """
        collection = [
            self.prepare_response(el, response_object = response_object) for el in data[self.api_resource]
        ]
        return {
            self.api_resource: collection,
            'meta': data['meta']
        }
    

