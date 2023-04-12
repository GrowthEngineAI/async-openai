import aiohttpx
import pathlib
from typing import Optional, Callable, Dict, Union, List

from lazyops.types import lazyproperty
from async_openai.schemas import *
from async_openai.types.options import ApiType
from async_openai.utils.logs import logger
from async_openai.utils.config import settings
from async_openai.routes import ApiRoutes
from async_openai.meta import OpenAIMetaClass


class OpenAIClient:
    """
    Main Client for all the routes in the API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        scheme: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        api_path: Optional[str] = None,
        api_type: Optional[ApiType] = None,
        api_version: Optional[str] = None,
        organization: Optional[str] = None,
        proxies: Optional[Union[str, Dict]] = None,
        app_info: Optional[Dict[str, str]] = None,
        
        headers: Optional[Dict] = None,
        debug_enabled: Optional[bool] = None,
        on_error: Optional[Callable] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        ignore_errors: Optional[bool] = None,

        **kwargs
    ):
        self.api_key = api_key if api_key is not None else settings.api_key
        self.api_type = api_type if api_type is not None else settings.api_type
        self.api_version = api_version if api_version is not None else settings.api_version
        self.organization = organization if organization is not None else settings.organization
        self.proxies = proxies if proxies is not None else settings.proxies
        self.app_info = app_info if app_info is not None else settings.app_info

        self.api_url = settings.get_api_url(host = host, port = port, scheme = scheme, url = url)
        self.base_url = settings.get_base_api_url(host = host, port = port, scheme = scheme, url = url, api_path = api_path)
        self.debug_enabled = debug_enabled if debug_enabled is not None else settings.debug_enabled
        
        self.timeout = timeout if timeout is not None else settings.timeout
        self.headers = headers if headers is not None else settings.get_headers(api_key = self.api_key, api_version = self.api_version, api_type = self.api_type, organization = self.organization, app_info = self.app_info)

        self.on_error = on_error
        self.ignore_errors = ignore_errors if ignore_errors is not None else settings.ignore_errors
        self.max_retries = max_retries if max_retries is not None else settings.max_retries

        self._kwargs = kwargs
        self.log_method = logger.info if self.debug_enabled else logger.debug
        self.client = aiohttpx.Client(
            base_url = self.base_url,
            timeout = self.timeout,
        )

        self.routes = ApiRoutes(
            client = self.client,
            headers = self.headers,
            debug_enabled = self.debug_enabled,
            on_error = self.on_error,
            ignore_errors = self.ignore_errors,
            timeout = self.timeout,
            max_retries = self.max_retries,
            **self._kwargs
        )
        if self.debug_enabled:
            logger.info(f"OpenAI Client initialized: {self.client.base_url}")
            logger.debug(f"Debug Enabled: {self.debug_enabled}")


    @lazyproperty
    def completions(self) -> CompletionRoute:
        """
        Returns the `CompletionRoute` class for interacting with `Completions`.
        
        Doc: `https://beta.openai.com/docs/api-reference/completions`
        """
        return self.routes.completions
    
    @lazyproperty
    def chat(self) -> ChatRoute:
        """
        Returns the `ChatRoute` class for interacting with `Chat` components

        Doc: `https://platform.openai.com/docs/api-reference/chat`
        """
        return self.routes.chat

    @lazyproperty
    def edits(self) -> EditRoute:
        """
        Returns the `EditRoute` class for interacting with `Edits`.
        
        Doc: `https://beta.openai.com/docs/api-reference/edits`
        """
        return self.routes.edits
    
    @lazyproperty
    def embeddings(self) -> EmbeddingRoute:
        """
        Returns the `EmbeddingRoute` class for interacting with `Embeddings`.
        
        Doc: `https://beta.openai.com/docs/api-reference/embeddings`
        """
        return self.routes.embeddings
    
    @lazyproperty
    def images(self) -> ImageRoute:
        """
        Returns the `ImageRoute` class for interacting with `Images`.
        
        Doc: `https://beta.openai.com/docs/api-reference/images`
        """
        return self.routes.images
    
    @lazyproperty
    def models(self) -> ModelRoute:
        """
        Returns the `ModelRoute` class for interacting with `models`.
        
        Doc: `https://beta.openai.com/docs/api-reference/models`
        """
        return self.routes.models

    """
    Context Managers
    """

    async def async_close(self):
        await self.client.aclose()
    
    def close(self):
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.async_close()


class OpenAI(metaclass = OpenAIMetaClass):
    """
    Interface for OpenAI
    """
    pass



