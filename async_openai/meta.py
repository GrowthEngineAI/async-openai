"""
Client Metaclass
"""

from __future__ import annotations

import pathlib

from typing import Optional, Callable, Dict, Union, TYPE_CHECKING

from async_openai.schemas import *
from async_openai.types.options import ApiType
from async_openai.utils.config import settings

if TYPE_CHECKING:
    from async_openai.client import OpenAIClient


class OpenAIMetaClass(type):
    api_key: Optional[str] = None
    url: Optional[str] = None
    scheme: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None

    api_key: Optional[str] = None
    api_path: Optional[str] = None
    api_type: Optional[ApiType] = None
    api_version: Optional[str] = None
    api_key_path: Optional[pathlib.Path] = None
    
    organization: Optional[str] = None
    proxies: Optional[Union[str, Dict]] = None
    app_info: Optional[Dict[str, str]] = None
    
    debug_enabled: Optional[bool] = None
    ignore_errors: Optional[bool] = None
    max_retries: Optional[int] = None
    on_error: Optional[Callable] = None
    timeout: Optional[int] = None

    headers: Optional[Dict] = None
    _api: Optional['OpenAIClient'] = None

    """
    The Global Meta Class for OpenAI API.
    """


    def configure(
        cls, 
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        scheme: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        
        api_path: Optional[str] = None,
        api_type: Optional[ApiType] = None,
        api_version: Optional[str] = None,
        api_key_path: Optional[pathlib.Path] = None,

        organization: Optional[str] = None,
        proxies: Optional[Union[str, Dict]] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        app_info: Optional[Dict[str, str]] = None,
        debug_enabled: Optional[bool] = None,
        ignore_errors: Optional[bool] = None,
        on_error: Optional[Callable] = None,

        reset: Optional[bool] = None,
        **kwargs
    ):
        """
        Configure the global OpenAI client.

        :param url: The OpenAI API URL              | Env: [`OPENAI_API_URL`]
        :param scheme: The OpenAI API Scheme        | Env: [`OPENAI_API_SCHEME`]
        :param host: The OpenAI API Host            | Env: [`OPENAI_API_HOST`]
        :param port: The OpenAI API Port            | Env: [`OPENAI_API_PORT`]
        :param api_key: The OpenAI API Key          | Env: [`OPENAI_API_KEY`]
        :param api_path: The OpenAI API Path        | Env: [`OPENAI_API_PATH`]
        :param api_type: The OpenAI API Type        | Env: [`OPENAI_API_TYPE`]
        :param api_version: The OpenAI API Version  | Env: [`OPENAI_API_VERSION`]
        :param api_key_path: The API Key Path       | Env: [`OPENAI_API_KEY_PATH`]
        :param organization: Organization           | Env: [`OPENAI_ORGANIZATION`]
        :param proxies: The OpenAI Proxies          | Env: [`OPENAI_PROXIES`]
        :param timeout: Timeout in Seconds          | Env: [`OPENAI_TIMEOUT`]
        :param max_retries: The OpenAI Max Retries  | Env: [`OPENAI_MAX_RETRIES`]
        :param kwargs: Additional Keyword Arguments
        """
        if url is not None: cls.url = url
        if scheme is not None: cls.scheme = scheme
        if host is not None: cls.host = host
        if port is not None: cls.port = port
        if api_key is not None: cls.api_key = api_key
        if api_path is not None: cls.api_path = api_path
        if api_key_path is not None: 
            cls.api_key_path = api_key_path if isinstance(api_key_path, pathlib.Path) else pathlib.Path(api_key_path)
            cls.api_key = cls.api_key_path.read_text().strip()
        
        if api_type is not None: 
            cls.api_type = ApiType(api_type) if isinstance(api_type, str) else api_type
        
        if api_version is not None: 
            cls.api_version = cls.api_type.get_version(api_version)
        if organization is not None: cls.organization = organization
        if proxies is not None: cls.proxies = proxies
        if timeout is not None: cls.timeout = timeout
        if max_retries is not None: cls.max_retries = max_retries
        if app_info is not None: cls.app_info = app_info
        if debug_enabled is not None: cls.debug_enabled = debug_enabled
        if ignore_errors is not None: cls.ignore_errors = ignore_errors
        if on_error is not None: cls.on_error = on_error

        if reset: cls._api = None
        # if cls._api is None:
        #     cls.get_api(**kwargs)


    def get_api(cls, reset: Optional[bool] = None, **kwargs) -> 'OpenAIClient':
        if cls._api is None or reset:
            from async_openai.client import OpenAIClient
            cls.headers = settings.get_headers(
                api_key = cls.api_key,
                organization = cls.organization,
                api_type = cls.api_type,
                api_version = cls.api_version,
            )
            cls._api = OpenAIClient(
                url = cls.url,
                scheme = cls.scheme,
                host = cls.host,
                port = cls.port,
                api_path = cls.api_path,
                api_type = cls.api_type,
                api_version = cls.api_version,
                organization = cls.organization,
                proxies = cls.proxies,
                timeout = cls.timeout,
                max_retries = cls.max_retries,
                headers = cls.headers,
                app_info = cls.app_info,
                debug_enabled = cls.debug_enabled,
                ignore_errors = cls.ignore_errors,
                on_error = cls.on_error,
                **kwargs
            )
        return cls._api
    

    @property
    def api(cls) -> 'OpenAIClient':
        """
        Returns the inherited OpenAI client.
        """
        if cls._api is None:
            cls.get_api()
        return cls._api
    

    """
    API Routes
    """

    @property
    def completions(cls) -> CompletionRoute:
        """
        Returns the `CompletionRoute` class for interacting with `Completions`.
        
        Doc: `https://beta.openai.com/docs/api-reference/completions`
        """
        return cls.api.completions
    
    @property
    def Completions(cls) -> CompletionRoute:
        """
        Returns the `CompletionRoute` class for interacting with `Completions`.
        
        Doc: `https://beta.openai.com/docs/api-reference/completions`
        """
        return cls.api.completions
    

    @property
    def chat(cls) -> ChatRoute:
        """
        Returns the `ChatRoute` class for interacting with `Chat`.
        
        Doc: `https://beta.openai.com/docs/api-reference/chat`
        """
        return cls.api.chat
    
    @property
    def Chat(cls) -> ChatRoute:
        """
        Returns the `ChatRoute` class for interacting with `Chat`.
        
        Doc: `https://beta.openai.com/docs/api-reference/chat`
        """
        return cls.api.chat
    
    @property
    def edits(cls) -> EditRoute:
        """
        Returns the `EditRoute` class for interacting with `Edits`.
        
        Doc: `https://beta.openai.com/docs/api-reference/edits`
        """
        return cls.api.edits
    
    @property
    def embeddings(cls) -> EmbeddingRoute:
        """
        Returns the `EmbeddingRoute` class for interacting with `Embeddings`.
        
        Doc: `https://beta.openai.com/docs/api-reference/embeddings`
        """
        return cls.api.embeddings
    
    @property
    def images(cls) -> ImageRoute:
        """
        Returns the `ImageRoute` class for interacting with `Images`.
        
        Doc: `https://beta.openai.com/docs/api-reference/images`
        """
        return cls.api.images
    
    @property
    def models(cls) -> ModelRoute:
        """
        Returns the `ModelRoute` class for interacting with `models`.
        
        Doc: `https://beta.openai.com/docs/api-reference/models`
        """
        return cls.api.models

    """
    Context Managers
    """

    async def async_close(cls):
        if cls._api is not None:
            await cls._api.async_close()
    
    def close(cls):
        if cls._api is not None:
            cls._api.close()
    
    def __enter__(cls):
        return cls
    
    def __exit__(cls, exc_type, exc_value, traceback):
        cls.close()
    
    async def __aenter__(cls):
        return cls
    
    async def __aexit__(cls, exc_type, exc_value, traceback):
        await cls.async_close()