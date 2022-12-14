import json
import aiohttpx
import backoff

from lazyops.types import BaseModel, Field, lazyproperty
from lazyops.utils import ObjectEncoder
from typing import Dict, Optional, Any, List, Type, Callable, Union

from async_openai.utils.logs import logger
from async_openai.utils.config import settings
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

class BaseRoute(BaseModel):
    client: aiohttpx.Client
    headers: Dict[str, str] = Field(default_factory = settings.get_headers)
    success_codes: Optional[List[int]] = RESPONSE_SUCCESS_CODES
    
    input_model: Optional[Type[BaseResource]] = None
    response_model: Optional[Type[BaseResponse]] = None

    # Options
    timeout: Optional[int] = None
    debug_enabled: Optional[bool] = False
    on_error: Optional[Callable] = None
    ignore_errors: Optional[bool] = False
    max_retries: Optional[int] = None


    @lazyproperty
    def api_resource(self):
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

    def create(
        self, 
        input_object: Optional[Type[BaseResource]] = None,
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
        
        data = json.dumps(
            input_object.dict(
                exclude_none = self.exclude_null
            ), 
            cls = ObjectEncoder
        )
        api_response = self._send(
            method = 'POST',
            url = self.api_resource,
            data = data,
            headers = self.headers,
            timeout = self.timeout,
            **kwargs
        )
        data = self.handle_response(api_response)
        return self.prepare_response(data, input_object = input_object)
    
    async def async_create(
        self, 
        input_object: Optional[Type[BaseResource]] = None,
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

        data = json.dumps(
            input_object.dict(
                exclude_none = self.exclude_null
            ), 
            cls = ObjectEncoder
        )
        api_response = await self._async_send(
            method = 'POST',
            url = self.api_resource,
            data = data,
            headers = self.headers,
            timeout = self.timeout,
            **kwargs
        )
        data = self.handle_response(api_response)
        return self.prepare_response(data, input_object = input_object)
    
    def batch_create(
        self, 
        input_object: Optional[Type[BaseResource]] = None,
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
            input_object.dict(
                exclude_none = self.exclude_null
            ), 
            cls = ObjectEncoder
        )
        api_response = self._send(
            method = 'POST',
            url = api_resource,
            data = data,
            headers = self.headers,
            timeout = self.timeout,
            **kwargs
        )
        resp = self.handle_response(api_response)
        return self.prepare_response(resp, input_object = input_object)

    async def async_batch_create(
        self, 
        input_object: Optional[Type[BaseResource]] = None,
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
            input_object.dict(
                exclude_none = self.exclude_null
            ), 
            cls = ObjectEncoder
        )
        api_response = await self._async_send(  
            method = 'POST',
            url = api_resource,
            data = data,
            headers = self.headers,
            timeout = self.timeout,
            **kwargs
        )
        resp = self.handle_response(api_response)
        return self.prepare_response(resp, input_object = input_object)

    
    def retrieve(
        self, 
        resource_id: str, 
        params: Optional[Dict[str, Any]] = None,
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
            headers = self.headers,
            **kwargs
        )
        data = self.handle_response(api_response)
        return self.prepare_response(data)
    
    async def async_retrieve(
        self,
        resource_id: str,
        params: Optional[Dict[str, Any]] = None,
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
            headers = self.headers,
            timeout = self.timeout,
            **kwargs
        )
        data = self.handle_response(api_response)
        return self.prepare_response(data)

    def get(
        self, 
        resource_id: str, 
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Type[BaseResource]:
        """
        GET a Single Resource

        :param resource_id: The ID of the Resource to GET
        :param params: Optional Query Parameters
        """
        return self.retrieve(resource_id = resource_id, params = params, **kwargs)
    
    async def async_get(
        self,
        resource_id: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Type[BaseResource]:
        """
        GET a Single Resource

        :param resource_id: The ID of the Resource to GET
        :param params: Optional Query Parameters
        """
        return await self.async_retrieve(resource_id = resource_id, params = params, **kwargs)

    
    def list(
        self, 
        params: Optional[Dict[str, Any]] = None,
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
            url = self.api_resource,
            params = params,
            headers = self.headers,
            timeout = self.timeout,
            **kwargs
        )
        data = self.handle_response(api_response)
        return self.prepare_response(data)
        #return self.prepare_index_response(data)

    async def async_list(
        self, 
        params: Optional[Dict[str, Any]] = None,
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
            url = self.api_resource,
            params = params,
            headers = self.headers,
            timeout = self.timeout,
            **kwargs
        )
        data = self.handle_response(api_response)
        return self.prepare_response(data)
        #return self.prepare_index_response(data)

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

    def delete(
        self, 
        resource_id: str,
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
            headers = self.headers,
            timeout = self.timeout,
            **kwargs
        )
        data = self.handle_response(api_response)
        return self.prepare_response(data)
    
    async def async_delete(
        self, 
        resource_id: str,
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
            headers = self.headers,
            timeout = self.timeout,
            **kwargs
        )
        data = self.handle_response(api_response)
        return self.prepare_response(data)


    def update(
        self, 
        input_object: Optional[Type[BaseResource]] = None,
        resource_id: str = None,
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
            input_object.dict(
                exclude_none = self.exclude_null
            ), 
            cls = ObjectEncoder
        )
        api_response = self._send(
            method = 'PUT',
            url = api_resource,
            data = data,
            headers = self.headers,
            timeout = self.timeout,
            **kwargs
        )
        data = self.handle_response(api_response)
        return self.prepare_response(data, input_object = input_object)

    async def async_update(
        self, 
        input_object: Optional[Type[BaseResource]] = None,
        resource_id: str = None,
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
            input_object.dict(
                exclude_none = self.exclude_null
            ), 
            cls = ObjectEncoder
        )
        api_response = await self._async_send(
            method = 'PUT',
            url = api_resource,
            data = data,
            headers = self.headers,
            timeout = self.timeout,
            **kwargs
        )
        data = self.handle_response(api_response)
        return self.prepare_response(data, input_object = input_object)




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



    def upload(
        self, 
        input_object: Optional[Type[FileResource]] = None,
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
        
        headers = self.headers.copy()
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
        
        headers = self.headers.copy()
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

    def download(
        self, 
        resource_id: str,
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
            headers = self.headers,
            timeout = self.timeout,
            **kwargs
        )
        data = self.handle_response(api_response)
        return self.prepare_response(data)

    async def async_download(
        self, 
        resource_id: str,
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
            headers = self.headers,
            timeout = self.timeout,
            **kwargs
        )
        data = self.handle_response(api_response)
        return self.prepare_response(data)

    def prepare_response(
        self, 
        data: aiohttpx.Response,
        input_object: Optional[Type[BaseResource]] = None,
        response_object: Optional[Type[BaseResource]] = None,
        **kwargs
    ):
        """
        Prepare the Response Object
        
        :param data: The Response Data
        :param response_object: The Response Object
        """
        response_object = response_object or self.response_model
        if response_object:
            return response_object.prepare_response(data, input_object = input_object)
        raise NotImplementedError('Response model not defined for this resource.')

    def handle_response(
        self, 
        response: aiohttpx.Response,
        **kwargs
    ):
        """
        Handle the Response

        :param response: The Response
        """
        if self.debug_enabled:
            logger.info(f'[{response.status_code} - {response.request.url}] headers: {response.headers}, body: {response.text}')
        
        if response.status_code in self.success_codes:
            return response if response.content else None
        
        if self.ignore_errors:
            return None
        
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
        # ignore_errors: Optional[bool] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        **kwargs
    ) -> aiohttpx.Response:
        # if ignore_errors is None: ignore_errors = self.ignore_errors
        if max_retries is None: max_retries = settings.max_retries
        if timeout is None: timeout = self.timeout
        if self.debug_enabled:
            logger.info(f'[{method} - {url}] headers: {headers}, params: {params}, data: {data}')

        @backoff.on_exception(
            backoff.expo, Exception, max_tries = max_retries + 1, giveup = fatal_exception
        )
        def _retryable_send():
            return self.client.request(
                method = method,
                url = url,
                params = params,
                data = data,
                headers = headers,
                timeout = timeout,
                **kwargs
            )
        return _retryable_send()
    
    async def _async_send(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
        # ignore_errors: Optional[bool] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        **kwargs
    ) -> aiohttpx.Response:
        # if ignore_errors is None: ignore_errors = self.ignore_errors
        if max_retries is None: max_retries = settings.max_retries
        if timeout is None: timeout = self.timeout
        if self.debug_enabled:
            logger.info(f'[{method} - {url}] headers: {headers}, params: {params}, data: {data}')


        @backoff.on_exception(
            backoff.expo, Exception, max_tries = max_retries + 1, giveup = fatal_exception
        )
        async def _retryable_async_send():
            return await self.client.async_request(
                method = method,
                url = url,
                params = params,
                data = data,
                headers = headers,
                timeout = timeout,
                **kwargs
            )
        return await _retryable_async_send()
    
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
    
    # def current_usage(
    #     self, 
    #     resource_id: str, 
    #     external_subscription_id: str,
    #     usage_object: Type[BaseResource] = None,
    #     **kwargs
    # ):
    #     """
    #     Get Current Usage for a Resource

    #     :param resource_id: The ID of the Resource to Get Usage For
    #     """
    #     if not self.usage_enabled:
    #         raise NotImplementedError('Usage is not enabled for this resource')
    #     api_resource = f'{self.api_resource}/{resource_id}/current_usage'
    #     #api_response = self.client.get(
    #     api_response = self._send(
    #         method = 'GET',
    #         url = api_resource, 
    #         params = {'external_subscription_id': external_subscription_id},
    #         headers = self.headers,
    #         **kwargs
    #     )
    #     data = self.handle_response(api_response).json().get('customer_usage')
    #     usage_object = usage_object or self.usage_model
    #     return usage_object.parse_obj(data)
    
    # async def async_current_usage(
    #     self, 
    #     resource_id: str, 
    #     external_subscription_id: str,
    #     usage_object: Type[BaseResource] = None,
    #     **kwargs
    # ):
    #     """
    #     Get Current Usage for a Resource

    #     :param resource_id: The ID of the Resource to Get Usage For
    #     """
    #     if not self.usage_enabled:
    #         raise NotImplementedError('Usage is not enabled for this resource')
    #     api_resource = f'{self.api_resource}/{resource_id}/current_usage'
    #     api_response = await self._async_send(
    #         method = 'GET',
    #         url = api_resource, 
    #         params = {'external_subscription_id': external_subscription_id},
    #         headers = self.headers,
    #         timeout = self.timeout,
    #         **kwargs
    #     )
    #     data = self.handle_response(api_response).json().get('customer_usage')
    #     usage_object = usage_object or self.usage_model
    #     return usage_object.parse_obj(data)






