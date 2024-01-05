"""
Client Metaclass
"""

from __future__ import annotations

import pathlib

from typing import Optional, List, Callable, Dict, Union, overload, TYPE_CHECKING

from async_openai.schemas import *
from async_openai.types.options import ApiType
from async_openai.types.context import ModelContextHandler
from async_openai.utils.config import get_settings, OpenAISettings
from async_openai.utils.logs import logger

if TYPE_CHECKING:
    from async_openai.client import OpenAIClient


class RotatingClients:
    """
    Manages a set of clients that can be rotated.    
    """
    def __init__(self, prioritize: Optional[str] = None, settings: Optional[OpenAISettings] = None, azure_model_mapping: Optional[Dict[str, str]] = None):
        self.settings = settings or get_settings()
        self.clients: Dict[str, 'OpenAIClient'] = {}
        self.rotate_index: int = 0
        self.rotate_client_names: List[str] = []
        self.azure_model_mapping: Dict[str, str] = azure_model_mapping

        assert prioritize in [None, 'azure', 'openai'], f'Invalid `prioritize` value: {prioritize}'
        self.prioritize: Optional[str] = prioritize
    
    @property
    def client_names(self) -> List[str]:
        """
        Returns the list of client names.
        """
        return list(self.clients.keys())
    
    def run_client_init(self):
        """
        Initializes the Client. 

        Can be subclassed to provide custom initialization.
        """
        self.init_api_client()
        if self.settings.has_valid_azure:
            self.init_api_client(client_name = 'az', is_azure = True, set_as_default = self.prioritize == 'azure', set_as_current = self.prioritize == 'azure')

    @property
    def api(self) -> 'OpenAIClient':
        """
        Returns the inherited OpenAI client.
        """
        if not self.clients: 
            self.run_client_init()
        if not self.rotate_client_names:
            return self.clients[self.client_names[self.rotate_index]]
        return self.clients[self.rotate_client_names[self.rotate_index]]
    
    def increase_rotate_index(self):
        """
        Increases the rotate index
        """
        if self.rotate_index >= len(self.clients) - 1:
            self.rotate_index = 0
        else:
            self.rotate_index += 1

    
    def rotate_client(self, index: Optional[int] = None, require_azure: Optional[bool] = None, verbose: Optional[bool] = False):
        """
        Rotates the clients
        """
        if index is not None:
            self.rotate_index = index
            return
        self.increase_rotate_index()
        if require_azure:
            while not self.api.is_azure:
                self.increase_rotate_index()
        if verbose:
            logger.info(f'Rotated Client: {self.api.name} (Azure: {self.api.is_azure} - {self.api.api_version}) [{self.rotate_index+1}/{len(self.clients)}]')
    
    def set_client(self, client_name: Optional[str] = None, verbose: Optional[bool] = False):
        """
        Sets the client
        """
        if client_name is None:
            raise ValueError('`client_name` is required.')
        if client_name not in self.clients:
            raise ValueError(f'Client `{client_name}` does not exist.')
        self.rotate_index = self.client_names.index(client_name)
        if verbose:
            logger.info(f'Set Client: {self.api.name} (Azure: {self.api.is_azure} - {self.api.api_version})) [{self.rotate_index+1}/{len(self.clients)}]')

    def current_client_info(self, verbose: Optional[bool] = False) -> Dict[str, Union[str, int]]:
        """
        Returns the current client info
        """
        data = {
            'name': self.api.name,
            'is_azure': self.api.is_azure,
            'api_version': self.api.api_version,
            'index': self.rotate_index,
            'total': len(self.clients),
        }
        if verbose:
            logger.info(f'Current Client: {self.api.name} (Azure: {self.api.is_azure} - {self.api.api_version}) [{self.rotate_index+1}/{len(self.clients)}]')
        return data


    def configure_client(self, client_name: Optional[str] = None, priority: Optional[int] = None, **kwargs):
        """
        Configure a new client
        """
        client_name = client_name or 'default'
        if client_name not in self.clients:
            raise ValueError(f'Client `{client_name}` does not exist.')
        self.clients[client_name].reset(**kwargs)
        if priority is not None:
            if client_name in self.rotate_client_names:
                self.rotate_client_names.remove(client_name)
            self.rotate_client_names.insert(priority, client_name)

    def init_api_client(
        self, 
        client_name: Optional[str] = None, 
        set_as_default: Optional[bool] = False, 
        is_azure: Optional[bool] = None,
        priority: Optional[int] = None,
        set_as_current: Optional[bool] = False,
        **kwargs
    ) -> 'OpenAIClient':
        """
        Creates a new OpenAI client.
        """
        client_name = client_name or 'default'
        if client_name in self.clients:
            return self.clients[client_name]

        from async_openai.client import OpenAIClient
        if is_azure is None and \
                (
                'az' in client_name and self.settings.has_valid_azure
            ):
            is_azure = True
        client = OpenAIClient(
            name = client_name,
            settings = self.settings,
            is_azure = is_azure,
            azure_model_mapping = self.azure_model_mapping,
            **kwargs
        )
        self.clients[client_name] = client
        if set_as_default:
            self.rotate_client_names.insert(0, client_name)
        elif priority is not None:
            if client_name in self.rotate_client_names:
                self.rotate_client_names.remove(client_name)
            self.rotate_client_names.insert(priority, client_name)
        elif self.prioritize:
            if (
                self.prioritize == 'azure'
                and is_azure
                or self.prioritize != 'azure'
                and self.prioritize == 'openai'
                and not is_azure
            ):
                self.rotate_client_names.insert(0, client_name)
            elif self.prioritize in ['azure', 'openai']:
                self.rotate_client_names.append(client_name)
        if set_as_current:
            self.rotate_index = self.rotate_client_names.index(client_name)
        return client
    
    def get_api_client(self, client_name: Optional[str] = None, require_azure: Optional[bool] = None, **kwargs) -> 'OpenAIClient':
        """
        Initializes a new OpenAI client or Returns an existing one.
        """
        if not client_name and not self.clients:
            client_name = 'default'
        if client_name and client_name not in self.clients:
            self.clients[client_name] = self.init_api_client(client_name = client_name, **kwargs)

        if not client_name and require_azure:
            while not self.api.is_azure:
                self.increase_rotate_index()
            return self.api
        return self.clients[client_name] if client_name else self.api

    def __getitem__(self, key: Union[str, int]) -> 'OpenAIClient':
        """
        Returns a client by name.
        """
        if isinstance(key, int):
            key = self.rotate_client_names[key] if self.rotate_client_names else self.client_names[key]
        return self.clients[key]

# Model Mapping for Azure
DefaultModelMapping = {
    'gpt-3.5-turbo': 'gpt-35-turbo',
    'gpt-3.5-turbo-16k': 'gpt-35-turbo-16k',
    'gpt-3.5-turbo-instruct': 'gpt-35-turbo-instruct',
    'gpt-3.5-turbo-0301': 'gpt-35-turbo-0301',
    'gpt-3.5-turbo-0613': 'gpt-35-turbo-0613',
    'gpt-3.5-turbo-1106': 'gpt-35-turbo-1106',
}
    
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
    prioritize: Optional[str] = None
    enable_rotating_clients: Optional[bool] = False
    azure_model_mapping: Optional[Dict[str, str]] = DefaultModelMapping

    _api: Optional['OpenAIClient'] = None
    _apis: Optional['RotatingClients'] = None
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
        disable_retries: Optional[bool] = None,
        max_connections: Optional[int] = None,
        max_keepalive_connections: Optional[int] = None,
        keepalive_expiry: Optional[int] = None,
        custom_headers: Optional[Dict[str, str]] = None,

        on_error: Optional[Callable] = None,
        reset: Optional[bool] = None,
        prioritize: Optional[str] = None,
        enable_rotating_clients: Optional[bool] = None,
        azure_model_mapping: Optional[Dict[str, str]] = None,
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
        :param disable_retries: Disable Retries     | Env: [`OPENAI_DISABLE_RETRIES`]
        :param max_connections: Max Connections     | Env: [`OPENAI_MAX_CONNECTIONS`]
        :param max_keepalive_connections: Max Keepalive Connections | Env: [`OPENAI_MAX_KEEPALIVE_CONNECTIONS`]
        :param keepalive_expiry: Keepalive Expiry   | Env: [`OPENAI_KEEPALIVE_EXPIRY`]
        :param custom_headers: Custom Headers       | Env: [`OPENAI_CUSTOM_HEADERS`]
        
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
        azure_max_connections: Optional[int] = None,
        azure_max_keepalive_connections: Optional[int] = None,
        azure_keepalive_expiry: Optional[int] = None,
        azure_custom_headers: Optional[Dict[str, str]] = None,

        on_error: Optional[Callable] = None,
        reset: Optional[bool] = None,
        prioritize: Optional[str] = None,
        enable_rotating_clients: Optional[bool] = None,
        azure_model_mapping: Optional[Dict[str, str]] = None,
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

    # @overload
    def configure(
        cls, 
        on_error: Optional[Callable] = None,
        prioritize: Optional[str] = None,
        enable_rotating_clients: Optional[bool] = None,
        azure_model_mapping: Optional[Dict[str, str]] = None,
        # reset: Optional[bool] = None,
        **kwargs
    ):
        """
        Configure the global OpenAI client.
        """
        if on_error is not None: cls.on_error = on_error
        if prioritize is not None: cls.prioritize = prioritize
        if enable_rotating_clients is not None: cls.enable_rotating_clients = enable_rotating_clients
        if azure_model_mapping is not None: 
            cls.azure_model_mapping = azure_model_mapping
            for key, val in azure_model_mapping.items():
                ModelContextHandler.add_model(key, val)
        cls.settings.configure(**kwargs)
    
    def configure_client(
        cls,
        client_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Configure a specific client.
        """
        if cls.enable_rotating_clients:
            return cls.apis.configure_client(client_name = client_name, **kwargs)
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
        if cls.enable_rotating_clients:
            return cls.apis.get_api_client(client_name = client_name, **kwargs)
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
        if cls.enable_rotating_clients:
            return cls.apis.init_api_client(client_name = client_name, set_as_default = set_as_default, is_azure = is_azure, **kwargs)
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
            azure_model_mapping = cls.azure_model_mapping,
            **kwargs
        )
        cls._clients[client_name] = client
        if set_as_default or not cls._api:
            cls._api = client
        return client
    
    def rotate_client(cls, index: Optional[int] = None, verbose: Optional[bool] = False, **kwargs):
        """
        Rotates the clients
        """
        if not cls.enable_rotating_clients:
            raise ValueError('Rotating Clients is not enabled.')
        cls.apis.rotate_client(index = index, verbose = verbose, **kwargs)
    
    def set_client(cls, client_name: Optional[str] = None, verbose: Optional[bool] = False):
        """
        Sets the client
        """
        if cls.enable_rotating_clients:
            cls.apis.set_client(client_name = client_name, verbose = verbose)
        else:
            cls._api = cls._clients[client_name]
            if verbose:
                logger.info(f'Set Client: {cls.api.name} ({cls.api.is_azure})')
    
    def get_current_client_info(cls, verbose: Optional[bool] = False) -> Dict[str, Union[str, int]]:
        """
        Returns the current client info
        """
        if cls.enable_rotating_clients:
            return cls.apis.current_client_info(verbose = verbose)
        data = {
            'name': cls.api.name,
            'is_azure': cls.api.is_azure,
            'api_version': cls.api.api_version,
        }
        if verbose:
            logger.info(f'Current Client: {cls.api.name} (Azure: {cls.api.is_azure} - {cls.api.api_version})')
        return data

    
    @property
    def apis(cls) -> RotatingClients:
        """
        Returns the global Rotating Clients.
        """
        if cls._apis is None:
            cls._apis = RotatingClients(prioritize=cls.prioritize, settings=cls.settings, azure_model_mapping=cls.azure_model_mapping)
        return cls._apis
    
    @property
    def api(cls) -> 'OpenAIClient':
        """
        Returns the inherited OpenAI client.
        """
        if cls.enable_rotating_clients: return cls.apis.api
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

    def __getitem__(cls, key: Union[int, str]) -> 'OpenAIClient':
        """
        Returns the OpenAI API Client.
        """
        if cls.enable_rotating_clients:
            return cls.apis[key]
        if isinstance(key, int):
            key = cls.client_names[key]
        return cls._clients[key]
    
    @property
    def client_names(cls) -> List[str]:
        """
        Returns the list of client names.
        """
        return list(cls._clients.keys())


    """
    Auto Rotating Functions
    """

    def chat_create(
        cls, 
        input_object: Optional[ChatObject] = None,
        parse_stream: Optional[bool] = True,
        auto_retry: Optional[bool] = False,
        auto_retry_limit: Optional[int] = None,
        verbose: Optional[bool] = False,
        **kwargs
    ) -> ChatResponse:
        """
        Creates a chat response for the provided prompt and parameters

        Usage:

        ```python
        >>> result = OpenAI.chat_create(
        >>>    messages = [{'content': 'say this is a test'}],
        >>>    max_tokens = 4,
        >>>    stream = True
        >>> )
        ```

        **Parameters:**

        :model (required): ID of the model to use. You can use the List models API 
        to see all of your available models,  or see our Model overview for descriptions of them.
        Default: `gpt-3.5-turbo`
        
        :messages: The messages to generate chat completions for, in the chat format.

        :max_tokens (optional): The maximum number of tokens to generate in the completion.
        The token count of your prompt plus `max_tokens` cannot exceed the model's context length. 
        Most models have a context length of 2048 tokens (except for the newest models, which 
        support 4096 / 8182 / 32,768). If max_tokens is not provided, the model will use the maximum number of tokens
        Default: None

        :temperature (optional): What sampling temperature to use. Higher values means 
        the model will take more risks. Try 0.9 for more creative applications, and 0 (argmax sampling) 
        for ones with a well-defined answer. We generally recommend altering this or `top_p` but not both.
        Default: `1.0`

        :top_p (optional): An alternative to sampling with `temperature`, called nucleus 
        sampling, where the model considers the results of the tokens with `top_p` probability mass. 
        So `0.1` means only  the tokens comprising the top 10% probability mass are considered.
        We generally recommend altering this or `temperature` but not both
        Default: `1.0`

        :n (optional): How many completions to generate for each prompt.
        Note: Because this parameter generates many completions, it can quickly 
        consume your token quota. Use carefully and ensure that you have reasonable 
        settings for `max_tokens` and stop.
        Default: `1`

        :stream (optional): CURRENTLY NOT SUPPORTED
        Whether to stream back partial progress. 
        If set, tokens will be sent as data-only server-sent events as they become 
        available, with the stream terminated by a `data: [DONE]` message. This is 
        handled automatically by the Client and enables faster response processing.
        Default: `False`

        :logprobs (optional): Include the log probabilities on the `logprobs` 
        most likely tokens, as well the chosen tokens. For example, if `logprobs` is 5, 
        the API will return a list of the 5 most likely tokens. The API will always 
        return the logprob of the sampled token, so there may be up to `logprobs+1` 
        elements in the response. The maximum value for `logprobs` is 5.
        Default: `None`

        :stop (optional): Up to 4 sequences where the API will stop generating 
        further tokens. The returned text will not contain the stop sequence.
        Default: `None`

        :presence_penalty (optional): Number between `-2.0` and `2.0`. Positive values 
        penalize new tokens based on whether they appear in the text so far, increasing the 
        model's likelihood to talk about new topics
        Default: `0.0`

        :frequency_penalty (optional): Number between `-2.0` and `2.0`. Positive 
        values penalize new tokens based on their existing frequency in the text so 
        far, decreasing the model's likelihood to repeat the same line verbatim.
        Default: `0.0`

        :logit_bias (optional): Modify the likelihood of specified tokens appearing in the completion.
        Accepts a json object that maps tokens (specified by their token ID in the GPT tokenizer) to an associated 
        bias value from -100 to 100. You can use this tokenizer tool (which works for both GPT-2 and GPT-3) to 
        convert text to token IDs. Mathematically, the bias is added to the logits generated by the model prior 
        to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase 
        likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the 
        relevant token.
        As an example, you can pass `{"50256": -100}` to prevent the `<|endoftext|>` token from being generated.
        Default: `None`

        :user (optional): A unique identifier representing your end-user, which can help OpenAI to 
        monitor and detect abuse.
        Default: `None`

        :functions (optional): A list of dictionaries representing the functions to call
        
        :function_call (optional): The name of the function to call. Default: `auto` if functions are provided

        :auto_retry (optional): Whether to automatically retry the request if it fails due to a rate limit error.

        :auto_retry_limit (optional): The maximum number of times to retry the request if it fails due to a rate limit error.

        Returns: `ChatResponse`
        """
    
        try:
            return cls.api.chat.create(input_object = input_object, parse_stream = parse_stream, auto_retry = auto_retry, auto_retry_limit = auto_retry_limit, **kwargs)

        except Exception as e:
            if not cls.enable_rotating_clients: raise e
            cls.rotate_client(verbose=verbose)
            return cls.chat_create(input_object = input_object, parse_stream = parse_stream, auto_retry = auto_retry, auto_retry_limit = auto_retry_limit, **kwargs)
    

    async def async_chat_create(
        cls, 
        input_object: Optional[ChatObject] = None,
        parse_stream: Optional[bool] = True,
        auto_retry: Optional[bool] = False,
        auto_retry_limit: Optional[int] = None,
        verbose: Optional[bool] = False,
        **kwargs
    ) -> ChatResponse:
        """
        Creates a chat response for the provided prompt and parameters

        Usage:

        ```python
        >>> result = await OpenAI.async_chat_create(
        >>>    messages = [{'content': 'say this is a test'}],
        >>>    max_tokens = 4,
        >>>    stream = True
        >>> )
        ```

        **Parameters:**

        :model (required): ID of the model to use. You can use the List models API 
        to see all of your available models,  or see our Model overview for descriptions of them.
        Default: `gpt-3.5-turbo`
        
        :messages: The messages to generate chat completions for, in the chat format.

        :max_tokens (optional): The maximum number of tokens to generate in the completion.
        The token count of your prompt plus `max_tokens` cannot exceed the model's context length. 
        Most models have a context length of 2048 tokens (except for the newest models, which 
        support 4096).
        Default: `16`

        :temperature (optional): What sampling temperature to use. Higher values means 
        the model will take more risks. Try 0.9 for more creative applications, and 0 (argmax sampling) 
        for ones with a well-defined answer. We generally recommend altering this or `top_p` but not both.
        Default: `1.0`

        :top_p (optional): An alternative to sampling with `temperature`, called nucleus 
        sampling, where the model considers the results of the tokens with `top_p` probability mass. 
        So `0.1` means only  the tokens comprising the top 10% probability mass are considered.
        We generally recommend altering this or `temperature` but not both
        Default: `1.0`

        :n (optional): How many completions to generate for each prompt.
        Note: Because this parameter generates many completions, it can quickly 
        consume your token quota. Use carefully and ensure that you have reasonable 
        settings for `max_tokens` and stop.
        Default: `1`

        :stream (optional): CURRENTLY NOT SUPPORTED
        Whether to stream back partial progress. 
        If set, tokens will be sent as data-only server-sent events as they become 
        available, with the stream terminated by a `data: [DONE]` message. This is 
        handled automatically by the Client and enables faster response processing.
        Default: `False`

        :logprobs (optional): Include the log probabilities on the `logprobs` 
        most likely tokens, as well the chosen tokens. For example, if `logprobs` is 5, 
        the API will return a list of the 5 most likely tokens. The API will always 
        return the logprob of the sampled token, so there may be up to `logprobs+1` 
        elements in the response. The maximum value for `logprobs` is 5.
        Default: `None`

        :stop (optional): Up to 4 sequences where the API will stop generating 
        further tokens. The returned text will not contain the stop sequence.
        Default: `None`

        :presence_penalty (optional): Number between `-2.0` and `2.0`. Positive values 
        penalize new tokens based on whether they appear in the text so far, increasing the 
        model's likelihood to talk about new topics
        Default: `0.0`

        :frequency_penalty (optional): Number between `-2.0` and `2.0`. Positive 
        values penalize new tokens based on their existing frequency in the text so 
        far, decreasing the model's likelihood to repeat the same line verbatim.
        Default: `0.0`

        :logit_bias (optional): Modify the likelihood of specified tokens appearing in the completion.
        Accepts a json object that maps tokens (specified by their token ID in the GPT tokenizer) to an associated 
        bias value from -100 to 100. You can use this tokenizer tool (which works for both GPT-2 and GPT-3) to 
        convert text to token IDs. Mathematically, the bias is added to the logits generated by the model prior 
        to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase 
        likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the 
        relevant token.
        As an example, you can pass `{"50256": -100}` to prevent the `<|endoftext|>` token from being generated.
        Default: `None`

        :user (optional): A unique identifier representing your end-user, which can help OpenAI to 
        monitor and detect abuse.

        :functions (optional): A list of dictionaries representing the functions to call
        
        :function_call (optional): The name of the function to call. Default: `auto` if functions are provided

        :auto_retry (optional): Whether to automatically retry the request if it fails due to a rate limit error.

        :auto_retry_limit (optional): The maximum number of times to retry the request if it fails due to a rate limit error.

        Default: `None`

        Returns: `ChatResponse`
        """

        try:
            return await cls.api.chat.async_create(input_object = input_object, parse_stream = parse_stream, auto_retry = auto_retry, auto_retry_limit = auto_retry_limit, **kwargs)

        except Exception as e:
            if not cls.enable_rotating_clients: raise e
            cls.rotate_client(verbose=verbose)
            return await cls.async_chat_create(input_object = input_object, parse_stream = parse_stream, auto_retry = auto_retry, auto_retry_limit = auto_retry_limit, **kwargs)
    

    