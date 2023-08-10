"""
Client Metaclass
"""

from __future__ import annotations

import pathlib

from typing import Optional, List, Callable, Dict, Union, overload, TYPE_CHECKING

from async_openai.schemas import *
from async_openai.types.options import ApiType
from async_openai.utils.config import get_settings, OpenAISettings

if TYPE_CHECKING:
    from async_openai.client import OpenAIClient


class OpenAIMetaClass(type):
    # api_key: Optional[str] = None
    # url: Optional[str] = None
    # scheme: Optional[str] = None
    # host: Optional[str] = None
    # port: Optional[int] = None

    # api_key: Optional[str] = None
    # api_path: Optional[str] = None
    # api_type: Optional[ApiType] = None
    # api_version: Optional[str] = None
    # api_key_path: Optional[pathlib.Path] = None
    
    # organization: Optional[str] = None
    # proxies: Optional[Union[str, Dict]] = None
    # app_info: Optional[Dict[str, str]] = None
    
    # debug_enabled: Optional[bool] = None
    # ignore_errors: Optional[bool] = None
    # max_retries: Optional[int] = None
    # timeout: Optional[int] = None

    # headers: Optional[Dict] = None

    on_error: Optional[Callable] = None

    _api: Optional['OpenAIClient'] = None
    _clients: Optional[Dict[str, 'OpenAIClient']] = {}
    _settings: Optional[OpenAISettings] = None

    """
    The Global Meta Class for OpenAI API.
    """

    @property
    def settings(cls) -> OpenAISettings:
        """
        Returns the global settings for the OpenAI API.
        """
        if cls._settings is None:
            cls._settings = get_settings()
        return cls._settings
    
    # Changing the behavior to become proxied through settings

    @property
    def api_key(cls) -> Optional[str]:
        """
        Returns the global API Key.
        """
        return cls.settings.api_key
    
    @property
    def url(cls) -> Optional[str]:
        """
        Returns the global URL.
        """
        return cls.settings.url
    
    @property
    def scheme(cls) -> Optional[str]:
        """
        Returns the global Scheme.
        """
        return cls.settings.scheme
    
    @property
    def host(cls) -> Optional[str]:
        """
        Returns the global Host.
        """
        return cls.settings.host
    
    @property
    def port(cls) -> Optional[int]:
        """
        Returns the global Port.
        """
        return cls.settings.port
    
    @property
    def api_base(cls) -> Optional[str]:
        """
        Returns the global API Base.
        """
        return cls.settings.api_base

    @property
    def api_path(cls) -> Optional[str]:
        """
        Returns the global API Path.
        """
        return cls.settings.api_path
    
    @property
    def api_type(cls) -> Optional[ApiType]:
        """
        Returns the global API Type.
        """
        return cls.settings.api_type
    
    @property
    def api_version(cls) -> Optional[str]:
        """
        Returns the global API Version.
        """
        return cls.settings.api_version
    
    @property
    def api_key_path(cls) -> Optional[pathlib.Path]:
        """
        Returns the global API Key Path.
        """
        return cls.settings.api_key_path
    
    @property
    def organization(cls) -> Optional[str]:
        """
        Returns the global Organization.
        """
        return cls.settings.organization
    
    @property
    def proxies(cls) -> Optional[Union[str, Dict]]:
        """
        Returns the global Proxies.
        """
        return cls.settings.proxies
    
    @property
    def timeout(cls) -> Optional[int]:
        """
        Returns the global Timeout.
        """
        return cls.settings.timeout
    
    @property
    def max_retries(cls) -> Optional[int]:
        """
        Returns the global Max Retries.
        """
        return cls.settings.max_retries
    
    @property
    def app_info(cls) -> Optional[Dict[str, str]]:
        """
        Returns the global App Info.
        """
        return cls.settings.app_info
    
    @property
    def debug_enabled(cls) -> Optional[bool]:
        """
        Returns the global Debug Enabled.
        """
        return cls.settings.debug_enabled
    
    @property
    def ignore_errors(cls) -> Optional[bool]:
        """
        Returns the global Ignore Errors.
        """
        return cls.settings.ignore_errors
    
    @property
    def timeout(cls) -> Optional[int]:
        """
        Returns the global Timeout.
        """
        return cls.settings.timeout


    @overload
    def configure(
        cls, 
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        scheme: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        
        api_base: Optional[str] = None,
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
        :param api_base: The OpenAI API Base        | Env: [`OPENAI_API_BASE`]
        :param api_key: The OpenAI API Key          | Env: [`OPENAI_API_KEY`]
        :param api_path: The OpenAI API Path        | Env: [`OPENAI_API_PATH`]
        :param api_type: The OpenAI API Type        | Env: [`OPENAI_API_TYPE`]
        :param api_version: The OpenAI API Version  | Env: [`OPENAI_API_VERSION`]
        :param api_key_path: The API Key Path       | Env: [`OPENAI_API_KEY_PATH`]
        :param organization: Organization           | Env: [`OPENAI_ORGANIZATION`]
        :param proxies: The OpenAI Proxies          | Env: [`OPENAI_PROXIES`]
        :param timeout: Timeout in Seconds          | Env: [`OPENAI_TIMEOUT`]
        :param max_retries: The OpenAI Max Retries  | Env: [`OPENAI_MAX_RETRIES`]
        :param ignore_errors: Ignore Errors         | Env: [`OPENAI_IGNORE_ERRORS`]
        :param on_error: On Error Callback          

        :param kwargs: Additional Keyword Arguments
        """
        ...
    
    @overload
    def configure(
        azure_api_key: Optional[str] = None,
        azure_url: Optional[str] = None,
        azure_scheme: Optional[str] = None,
        azure_host: Optional[str] = None,
        azure_port: Optional[int] = None,
        
        azure_api_base: Optional[str] = None,
        azure_api_path: Optional[str] = None,
        azure_api_type: Optional[ApiType] = None,
        azure_api_version: Optional[str] = None,
        azure_api_key_path: Optional[pathlib.Path] = None,

        azure_organization: Optional[str] = None,
        azure_proxies: Optional[Union[str, Dict]] = None,
        azure_timeout: Optional[int] = None,
        azure_max_retries: Optional[int] = None,
        azure_app_info: Optional[Dict[str, str]] = None,
        azure_debug_enabled: Optional[bool] = None,
        azure_ignore_errors: Optional[bool] = None,

        on_error: Optional[Callable] = None,
        reset: Optional[bool] = None,
        **kwargs
    ):
        """
        Configure the global OpenAI client for Azure

        :param azure_url: The OpenAI API URL              | Env: [`AZURE_OPENAI_API_URL`]
        :param azure_scheme: The OpenAI API Scheme        | Env: [`AZURE_OPENAI_API_SCHEME`]
        :param azure_host: The OpenAI API Host            | Env: [`AZURE_OPENAI_API_HOST`]
        :param azure_port: The OpenAI API Port            | Env: [`AZURE_OPENAI_API_PORT`]
        :param azure_api_key: The OpenAI API Key          | Env: [`AZURE_OPENAI_API_KEY`]
        :param azure_api_base: The OpenAI API Base        | Env: [`AZURE_OPENAI_API_BASE`]
        :param azure_api_path: The OpenAI API Path        | Env: [`AZURE_OPENAI_API_PATH`]
        :param azure_api_type: The OpenAI API Type        | Env: [`AZURE_OPENAI_API_TYPE`]
        :param azure_api_version: The OpenAI API Version  | Env: [`AZURE_OPENAI_API_VERSION`]
        :param azure_api_key_path: The API Key Path       | Env: [`AZURE_OPENAI_API_KEY_PATH`]
        :param azure_organization: Organization           | Env: [`AZURE_OPENAI_ORGANIZATION`]
        :param azure_proxies: The OpenAI Proxies          | Env: [`AZURE_OPENAI_PROXIES`]
        :param azure_timeout: Timeout in Seconds          | Env: [`AZURE_OPENAI_TIMEOUT`]
        :param azure_max_retries: The OpenAI Max Retries  | Env: [`AZURE_OPENAI_MAX_RETRIES`]
        :param kwargs: Additional Keyword Arguments
        """
        ...


    def configure(
        cls, 
        on_error: Optional[Callable] = None,
        # reset: Optional[bool] = None,
        **kwargs
    ):
        """
        Configure the global OpenAI client.
        """
        if on_error is not None: cls.on_error = on_error
        cls.settings.configure(**kwargs)
    
    def configure_client(
        cls,
        client_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Configure a specific client.
        """
        client_name = client_name or 'default'
        if client_name not in cls._clients:
            raise ValueError(f'Client `{client_name}` does not exist.')
        cls._clients[client_name].reset(**kwargs)

    def get_api_client(
        cls,
        client_name: Optional[str] = None,
        **kwargs,
    ) -> 'OpenAIClient':
        """
        Initializes a new OpenAI client or Returns an existing one.
        """
        client_name = client_name or 'default'
        if client_name not in cls._clients:
            cls._clients[client_name] = cls.init_api_client(client_name = client_name, **kwargs)
        return cls._clients[client_name]
        
    def init_api_client(
        cls, 
        client_name: Optional[str] = None,
        set_as_default: Optional[bool] = False,
        is_azure: Optional[bool] = None,
        **kwargs
    ) -> 'OpenAIClient':
        """
        Creates a new OpenAI client.
        """
        client_name = client_name or 'default'
        if client_name in cls._clients:
            return cls._clients[client_name]
        
        from async_openai.client import OpenAIClient
        if is_azure is None and \
            (
                # (client_name == 'default' or 'az' in client_name) and 
                'az' in client_name and cls.settings.has_valid_azure
            ):
            is_azure = True
        client = OpenAIClient(
            name = client_name,
            settings = cls.settings,
            is_azure = is_azure,
            **kwargs
        )
        cls._clients[client_name] = client
        if set_as_default or not cls._api:
            cls._api = client
        return client
    
    @property
    def api(cls) -> 'OpenAIClient':
        """
        Returns the inherited OpenAI client.
        """
        if cls._api is None:
            cls.init_api_client()
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
        """
        Closes the OpenAI API Client.
        """
        for client in cls._clients.values():
            await client.async_close()

    
    def close(cls):
        """
        Closes the OpenAI API Client.
        """
        for client in cls._clients.values():
            client.close()
    
    def __enter__(cls):
        return cls
    
    def __exit__(cls, exc_type, exc_value, traceback):
        cls.close()
    
    async def __aenter__(cls):
        return cls
    
    async def __aexit__(cls, exc_type, exc_value, traceback):
        await cls.async_close()

    def __getitem__(cls, key: str) -> 'OpenAIClient':
        """
        Returns the OpenAI API Client.
        """
        return cls._clients[key]
    
    @property
    def client_names(cls) -> List[str]:
        """
        Returns the list of client names.
        """
        return list(cls._clients.keys())
                    