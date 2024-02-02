from __future__ import annotations

"""
OpenAI Session Manager
"""

import abc
import copy
import pathlib
import random
from typing import Optional, List, Callable, Dict, Union, Any, overload, TYPE_CHECKING

from async_openai.schemas import *
from async_openai.types.options import ApiType
from async_openai.types.context import ModelContextHandler
from async_openai.utils.config import get_settings, OpenAISettings
from async_openai.types.functions import FunctionManager, OpenAIFunctions
from async_openai.utils.logs import logger

from .loadbalancer import ClientLoadBalancer

if TYPE_CHECKING:
    from async_openai.client import OpenAIClient
    from lazyops.libs.pooler import ThreadPool



# Model Mapping for Azure
DefaultModelMapping = {
    'gpt-3.5-turbo': 'gpt-35-turbo',
    'gpt-3.5-turbo-16k': 'gpt-35-turbo-16k',
    'gpt-3.5-turbo-instruct': 'gpt-35-turbo-instruct',
    'gpt-3.5-turbo-0301': 'gpt-35-turbo-0301',
    'gpt-3.5-turbo-0613': 'gpt-35-turbo-0613',
    'gpt-3.5-turbo-1106': 'gpt-35-turbo-1106',
}
    
class OpenAIManager(abc.ABC):
    name: Optional[str] = "openai"
    on_error: Optional[Callable] = None
    prioritize: Optional[str] = None
    auto_healthcheck: Optional[bool] = None
    auto_loadbalance_clients: Optional[bool] = None
    azure_model_mapping: Optional[Dict[str, str]] = DefaultModelMapping

    _api: Optional['OpenAIClient'] = None
    _apis: Optional['ClientLoadBalancer'] = None
    _clients: Optional[Dict[str, 'OpenAIClient']] = {}
    _settings: Optional[OpenAISettings] = None

    _pooler: Optional['ThreadPool'] = None

    """
    The Global Session Manager for OpenAI API.
    """

    def __init__(self, **kwargs):
        """
        Initializes the OpenAI API Client
        """
        self.client_model_exclusions: Optional[Dict[str, Dict[str, Union[bool, List[str]]]]] = {}
        self.no_proxy_client_names: Optional[List[str]] = []
        self.client_callbacks: Optional[List[Callable]] = []
        self.functions: FunctionManager = OpenAIFunctions
        self.ctx = ModelContextHandler
        if self.auto_loadbalance_clients is None: self.auto_loadbalance_clients = self.settings.auto_loadbalance_clients
        if self.auto_healthcheck is None: self.auto_healthcheck = self.settings.auto_healthcheck

    def add_callback(self, callback: Callable):
        """
        Adds a callback to the client
        """
        self.client_callbacks.append(callback)

    @property
    def settings(self) -> OpenAISettings:
        """
        Returns the global settings for the OpenAI API.
        """
        if self._settings is None:
            self._settings = get_settings()
        return self._settings
    
    # Changing the behavior to become proxied through settings

    @property
    def api_key(self) -> Optional[str]:
        """
        Returns the global API Key.
        """
        return self.settings.api_key
    
    @property
    def url(self) -> Optional[str]:
        """
        Returns the global URL.
        """
        return self.settings.url
    
    @property
    def scheme(self) -> Optional[str]:
        """
        Returns the global Scheme.
        """
        return self.settings.scheme
    
    @property
    def host(self) -> Optional[str]:
        """
        Returns the global Host.
        """
        return self.settings.host
    
    @property
    def port(self) -> Optional[int]:
        """
        Returns the global Port.
        """
        return self.settings.port
    
    @property
    def api_base(self) -> Optional[str]:
        """
        Returns the global API Base.
        """
        return self.settings.api_base

    @property
    def api_path(self) -> Optional[str]:
        """
        Returns the global API Path.
        """
        return self.settings.api_path
    
    @property
    def api_type(self) -> Optional[ApiType]:
        """
        Returns the global API Type.
        """
        return self.settings.api_type
    
    @property
    def api_version(self) -> Optional[str]:
        """
        Returns the global API Version.
        """
        return self.settings.api_version
    
    @property
    def api_key_path(self) -> Optional[pathlib.Path]:
        """
        Returns the global API Key Path.
        """
        return self.settings.api_key_path
    
    @property
    def organization(self) -> Optional[str]:
        """
        Returns the global Organization.
        """
        return self.settings.organization
    
    @property
    def proxies(self) -> Optional[Union[str, Dict]]:
        """
        Returns the global Proxies.
        """
        return self.settings.proxies
    
    @property
    def timeout(self) -> Optional[int]:
        """
        Returns the global Timeout.
        """
        return self.settings.timeout
    
    @property
    def max_retries(self) -> Optional[int]:
        """
        Returns the global Max Retries.
        """
        return self.settings.max_retries
    
    @property
    def app_info(self) -> Optional[Dict[str, str]]:
        """
        Returns the global App Info.
        """
        return self.settings.app_info
    
    @property
    def debug_enabled(self) -> Optional[bool]:
        """
        Returns the global Debug Enabled.
        """
        return self.settings.debug_enabled
    
    @property
    def ignore_errors(self) -> Optional[bool]:
        """
        Returns the global Ignore Errors.
        """
        return self.settings.ignore_errors
    
    @property
    def timeout(self) -> Optional[int]:
        """
        Returns the global Timeout.
        """
        return self.settings.timeout
    
    @property
    def pooler(self) -> Optional['ThreadPool']:
        """
        Returns the global ThreadPool.
        """
        if self._pooler is None:
            from lazyops.libs.pooler import ThreadPooler
            self._pooler = ThreadPooler
        return self._pooler

    
    def configure_client(
        self,
        client_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Configure a specific client.
        """
        if self.auto_loadbalance_clients:
            return self.apis.configure_client(client_name = client_name, **kwargs)
        client_name = client_name or 'default'
        if client_name not in self._clients:
            raise ValueError(f'Client `{client_name}` does not exist.')
        self._clients[client_name].reset(**kwargs)

    def get_api_client(
        self,
        client_name: Optional[str] = None,
        **kwargs,
    ) -> 'OpenAIClient':
        """
        Initializes a new OpenAI client or Returns an existing one.
        """
        if self.auto_loadbalance_clients:
            return self.apis.get_api_client(client_name = client_name, **kwargs)
        client_name = client_name or 'default'
        if client_name not in self._clients:
            self._clients[client_name] = self.init_api_client(client_name = client_name, **kwargs)
        return self._clients[client_name]
    

    def get_api_client_from_list(
        self,
        client_names: Optional[List[str]] = None,
        **kwargs,
    ) -> 'OpenAIClient':
        """
        Initializes a new OpenAI client or Returns an existing one.
        """
        if self.auto_loadbalance_clients:
            if not client_names: return self.apis.get_api_client(**kwargs)
            return self.apis.get_api_client_from_list(client_names = client_names, **kwargs)
        if not client_names: return self.get_api_client(**kwargs)
        if not self.auto_healthcheck:
            name = random.choice(client_names)
            return self.get_api_client(client_name = name, **kwargs)
        
        for client_name in client_names:
            if client_name not in self._clients:
                self._clients[client_name] = self.init_api_client(client_name = client_name, **kwargs)
            if not self._clients[client_name].ping():
                continue
            return self._clients[client_name]
        raise ValueError(f'No healthy client found from: {client_names}')
    
    async def aget_api_client_from_list(
        self,
        client_names: Optional[List[str]] = None,
        **kwargs,
    ) -> 'OpenAIClient':
        """
        Initializes a new OpenAI client or Returns an existing one.
        """
        if self.auto_loadbalance_clients:
            if not client_names: return self.apis.get_api_client(**kwargs)
            return await self.apis.aget_api_client_from_list(client_name = client_name, **kwargs)
        if not client_names: return self.get_api_client(**kwargs)
        if not self.auto_healthcheck:
            name = random.choice(client_names)
            return self.get_api_client(client_name = name, **kwargs)
        
        for client_name in client_names:
            if client_name not in self._clients:
                self._clients[client_name] = self.init_api_client(client_name = client_name, **kwargs)
            if not await self._clients[client_name].aping():
                continue
            return self._clients[client_name]
        raise ValueError(f'No healthy client found from: {client_names}')
    
        
    def init_api_client(
        self, 
        client_name: Optional[str] = None,
        set_as_default: Optional[bool] = False,
        is_azure: Optional[bool] = None,
        **kwargs
    ) -> 'OpenAIClient':
        """
        Creates a new OpenAI client.
        """
        if self.auto_loadbalance_clients:
            return self.apis.init_api_client(client_name = client_name, set_as_default = set_as_default, is_azure = is_azure, **kwargs)
        client_name = client_name or 'default'
        if client_name in self._clients:
            return self._clients[client_name]
        
        from async_openai.client import OpenAIClient
        if is_azure is None and \
            (
                # (client_name == 'default' or 'az' in client_name) and 
                'az' in client_name and self.settings.has_valid_azure
            ):
            is_azure = True
        if 'client_callbacks' not in kwargs and self.client_callbacks:
            kwargs['client_callbacks'] = self.client_callbacks
        client = OpenAIClient(
            name = client_name,
            settings = self.settings,
            is_azure = is_azure,
            azure_model_mapping = self.azure_model_mapping,
            **kwargs
        )
        self._clients[client_name] = client
        if set_as_default or not self._api:
            self._api = client
        return client
    
    def rotate_client(self, index: Optional[int] = None, verbose: Optional[bool] = False, **kwargs):
        """
        Rotates the clients
        """
        if not self.auto_loadbalance_clients:
            raise ValueError('Rotating Clients is not enabled.')
        self.apis.rotate_client(index = index, verbose = verbose, **kwargs)
    
    def set_client(self, client_name: Optional[str] = None, verbose: Optional[bool] = False):
        """
        Sets the client
        """
        if self.auto_loadbalance_clients:
            self.apis.set_client(client_name = client_name, verbose = verbose)
        else:
            self._api = self._clients[client_name]
            if verbose:
                logger.info(f'Set Client: {self.api.name} ({self.api.is_azure})')
    
    def get_current_client_info(self, verbose: Optional[bool] = False) -> Dict[str, Union[str, int]]:
        """
        Returns the current client info
        """
        if self.auto_loadbalance_clients:
            return self.apis.current_client_info(verbose = verbose)
        data = {
            'name': self.api.name,
            'is_azure': self.api.is_azure,
            'api_version': self.api.api_version,
        }
        if verbose:
            logger.info(f'Current Client: {self.api.name} (Azure: {self.api.is_azure} - {self.api.api_version})')
        return data

    
    @property
    def apis(self) -> ClientLoadBalancer:
        """
        Returns the global Rotating Clients.
        """
        if self._apis is None:
            self._apis = ClientLoadBalancer(
                prioritize=self.prioritize, 
                settings=self.settings, 
                azure_model_mapping=self.azure_model_mapping, 
                healthcheck=self.auto_healthcheck,
                manager = self,
            )
            if self.settings.client_configurations:
                self.register_client_endpoints()
            else:
                self.register_default_endpoints()
        return self._apis
    
    @property
    def api(self) -> 'OpenAIClient':
        """
        Returns the inherited OpenAI client.
        """
        if self.auto_loadbalance_clients: return self.apis.api
        if self._api is None:
            self.init_api_client()
        return self._api
    
    def configure_internal_apis(self):
        """
        Helper method to ensure that the APIs are initialized
        """
        if self._api is not None: return
        # Invoke it to ensure that it is initialized
        if self.auto_loadbalance_clients:
            self.apis
        else:
            self.init_api_client()

    def _ensure_api(self):
        """
        Ensures that the API is initialized
        """
        if self._api is None: self.configure_internal_apis()


    
    
    """
    API Routes
    """

    @property
    def completions(self) -> CompletionRoute:
        """
        Returns the `CompletionRoute` class for interacting with `Completions`.
        
        Doc: `https://beta.openai.com/docs/api-reference/completions`
        """
        return self.api.completions
    
    @property
    def Completions(self) -> CompletionRoute:
        """
        Returns the `CompletionRoute` class for interacting with `Completions`.
        
        Doc: `https://beta.openai.com/docs/api-reference/completions`
        """
        return self.api.completions
    

    @property
    def chat(self) -> ChatRoute:
        """
        Returns the `ChatRoute` class for interacting with `Chat`.
        
        Doc: `https://beta.openai.com/docs/api-reference/chat`
        """
        return self.api.chat
    
    @property
    def Chat(self) -> ChatRoute:
        """
        Returns the `ChatRoute` class for interacting with `Chat`.
        
        Doc: `https://beta.openai.com/docs/api-reference/chat`
        """
        return self.api.chat
    
    @property
    def edits(self) -> EditRoute:
        """
        Returns the `EditRoute` class for interacting with `Edits`.
        
        Doc: `https://beta.openai.com/docs/api-reference/edits`
        """
        return self.api.edits
    
    @property
    def embeddings(self) -> EmbeddingRoute:
        """
        Returns the `EmbeddingRoute` class for interacting with `Embeddings`.
        
        Doc: `https://beta.openai.com/docs/api-reference/embeddings`
        """
        return self.api.embeddings
    
    @property
    def images(self) -> ImageRoute:
        """
        Returns the `ImageRoute` class for interacting with `Images`.
        
        Doc: `https://beta.openai.com/docs/api-reference/images`
        """
        return self.api.images
    
    @property
    def models(self) -> ModelRoute:
        """
        Returns the `ModelRoute` class for interacting with `models`.
        
        Doc: `https://beta.openai.com/docs/api-reference/models`
        """
        return self.api.models


    """
    V2 Endpoint Registration with Proxy Support
    """

    def register_default_endpoints(self):
        """
        Register the default clients
        """
        if self.settings.proxy.enabled:
            api_base = self.settings.proxy.endpoint
            az_custom_headers = {
                "Helicone-OpenAI-Api-Base": self.settings.azure.api_base
            }
            self.configure(
                api_base = api_base,
                azure_api_base = api_base,
                azure_custom_headers = az_custom_headers,
                enable_rotating_clients = True,
                prioritize = "azure",
            )

        self.init_api_client('openai', is_azure = False)
        if self.settings.has_valid_azure:
            self.init_api_client('azure', is_azure = True)


    def register_client_endpoints(self):
        """
        Register the Client Endpoints
        """
        client_configs = copy.deepcopy(self.settings.client_configurations)
        for name, config in client_configs.items():
            is_enabled = config.pop('enabled', False)
            if not is_enabled: continue
            is_azure = 'azure' in name or 'az' in name or config.get('is_azure', False)
            is_default = config.pop('default', False)
            proxy_disabled = config.pop('proxy_disabled', False)
            source_endpoint = config.get('api_base')
            if self.debug_enabled is not None: config['debug_enabled'] = self.debug_enabled
            if excluded_models := config.pop('excluded_models', None):
                self.client_model_exclusions[name] = {
                    'models': excluded_models, 'is_azure': is_azure,
                }
            else:
                self.client_model_exclusions[name] = {
                    'models': None, 'is_azure': is_azure,
                }
            
            if (self.settings.proxy.enabled and not proxy_disabled) and config.get('api_base'):
                # Initialize a non-proxy version of the client
                config['api_base'] = source_endpoint
                non_proxy_name = f'{name}_noproxy'
                self.client_model_exclusions[non_proxy_name] = self.client_model_exclusions[name].copy()
                self.no_proxy_client_names.append(non_proxy_name)
                self.init_api_client(non_proxy_name, is_azure = is_azure, set_as_default = False, **config)
                config['headers'] = self.settings.proxy.create_proxy_headers(
                    name = name,
                    config = config,
                )
                config['api_base'] = self.settings.proxy.endpoint
            c = self.init_api_client(name, is_azure = is_azure, set_as_default = is_default, **config)
            logger.info(f'Registered: `|g|{c.name}|e|` @ `{source_endpoint or c.base_url}` (Azure: {c.is_azure})', colored = True)
            

    def select_client_names(
        self, 
        client_name: Optional[str] = None, 
        azure_required: Optional[bool] = None, 
        openai_required: Optional[bool] = None,
        model: Optional[str] = None,
        noproxy_required: Optional[bool] = None,
        excluded_clients: Optional[List[str]] = None,
    ) -> Optional[List[str]]:
        """
        Select Client based on the client name, azure_required, and model
        """
        if client_name is not None: return client_name
        oai_name = 'openai_noproxy' if noproxy_required else 'openai'
        if openai_required: return [oai_name]
        if model is not None:
            available_clients = []
            for name, values in self.client_model_exclusions.items():
                if (noproxy_required and 'noproxy' not in name) or (not noproxy_required and 'noproxy' in name): continue
                if excluded_clients and name in excluded_clients: continue

                # Prioritize Azure Clients
                if (
                    azure_required and not values['is_azure']
                ) or (
                    not azure_required and not values['is_azure']
                ):
                    continue
                if not values['models'] or model not in values['models']:
                    available_clients.append(name)
                    # return name
            
            if not available_clients: available_clients.append(oai_name)
            return available_clients
        if azure_required:
            return [
                k for k, v in self.client_model_exclusions.items() if v['is_azure'] and \
                ('noproxy' in k if noproxy_required else 'noproxy' not in k)
            ]
            # return [k for k, v in self.client_model_exclusions.items() if v['is_azure']]
        return None
    
    def get_client(
        self, 
        client_name: Optional[str] = None, 
        azure_required: Optional[bool] = None, 
        openai_required: Optional[bool] = None,
        model: Optional[str] = None,
        noproxy_required: Optional[bool] = None,
        excluded_clients: Optional[List[str]] = None,
        **kwargs,
    ) -> "OpenAIClient":
        """
        Gets the OpenAI client

        Args:
            client_name (str, optional): The name of the client to use. If not provided, it will be selected based on the other parameters.
            azure_required (bool, optional): Whether the client must be an Azure client.
            openai_required (bool, optional): Whether the client must be an OpenAI client.
            model (str, optional): The model to use. If provided, the client will be selected based on the model.
            noproxy_required (bool, optional): Whether the client must be a non-proxy client.
            excluded_clients (List[str], optional): A list of client names to exclude from selection.
        """
        self._ensure_api()
        client_names = self.select_client_names(
            client_name = client_name, 
            azure_required = azure_required, 
            openai_required = openai_required, 
            model = model, 
            noproxy_required = noproxy_required, 
            excluded_clients = excluded_clients
        )
        client = self.get_api_client_from_list(
            client_names = client_names, 
            azure_required = azure_required,
            **kwargs
        )
        if self.debug_enabled:
            logger.info(f'Available Clients: {client_names} - Selected: {client.name}')
        if not client_name and self.auto_loadbalance_clients:
            self.apis.increase_rotate_index()
        return client
    

    def get_chat_client(
        self, 
        client_name: Optional[str] = None, 
        azure_required: Optional[bool] = None, 
        openai_required: Optional[bool] = None,
        model: Optional[str] = None,
        noproxy_required: Optional[bool] = None,
        excluded_clients: Optional[List[str]] = None,
        **kwargs,
    ) -> ChatRoute:
        """
        Gets the chat client
        """
        return self.get_client(client_name = client_name, azure_required = azure_required, openai_required = openai_required, model = model, noproxy_required = noproxy_required, excluded_clients = excluded_clients, **kwargs).chat

    def get_completion_client(
        self, 
        client_name: Optional[str] = None, 
        azure_required: Optional[bool] = None, 
        openai_required: Optional[bool] = None,
        model: Optional[str] = None,
        noproxy_required: Optional[bool] = None,
        excluded_clients: Optional[List[str]] = None,
        **kwargs
    ) -> CompletionRoute:
        """
        Gets the chat client
        """
        return self.get_client(client_name = client_name, azure_required = azure_required, openai_required = openai_required, model = model, noproxy_required = noproxy_required, excluded_clients = excluded_clients, **kwargs).completions
    
    def get_embedding_client(
        self, 
        client_name: Optional[str] = None, 
        azure_required: Optional[bool] = None, 
        openai_required: Optional[bool] = None,
        model: Optional[str] = None,
        noproxy_required: Optional[bool] = None,
        excluded_clients: Optional[List[str]] = None,
        **kwargs
    ) -> EmbeddingRoute:
        """
        Gets the chat client
        """
        return self.get_client(client_name = client_name, azure_required = azure_required, openai_required = openai_required, model = model, noproxy_required = noproxy_required, excluded_clients = excluded_clients, **kwargs).embeddings


    """
    V2 Utilities
    """
    def truncate_to_max_length(
        self,
        text: str,
        model: str,
        max_length: Optional[int] = None,
        buffer_length: Optional[int] = None,
    ) -> str:
        """
        Truncates the text to the max length
        """
        if max_length is None:
            model_ctx = ModelContextHandler.get(model)
            max_length = model_ctx.context_length
        if buffer_length is not None: max_length -= buffer_length
        
        encoder = ModelContextHandler.get_tokenizer(model)
        tokens = encoder.encode(text)
        if len(tokens) > max_length:
            tokens = tokens[-max_length:]
            decoded = encoder.decode(tokens)
            text = text[-len(decoded):]
        return text
    
    def truncate_batch_to_max_length(
        self,
        texts: List[str],
        model: str,
        max_length: Optional[int] = None,
        buffer_length: Optional[int] = None,
    ) -> List[str]:
        """
        Truncates the text to the max length
        """
        if max_length is None:
            model_ctx = ModelContextHandler.get(model)
            max_length = model_ctx.context_length
        if buffer_length is not None: max_length -= buffer_length
        encoder = ModelContextHandler.get_tokenizer(model)
        truncated_texts = []
        for text in texts:
            tokens = encoder.encode(text)
            if len(tokens) > max_length:
                tokens = tokens[-max_length:]
                decoded = encoder.decode(tokens)
                text = text[-len(decoded):]
            truncated_texts.append(text)
        return truncated_texts

    async def atruncate_to_max_length(
        self,
        text: str,
        model: str,
        max_length: Optional[int] = None,
        buffer_length: Optional[int] = None,
    ) -> str:
        """
        Truncates the text to the max length
        """
        return await self.pooler.arun(
            self.truncate_to_max_length,
            text = text,
            model = model,
            max_length = max_length,
            buffer_length = buffer_length,
        )
    
    async def atruncate_batch_to_max_length(
        self,
        texts: List[str],
        model: str,
        max_length: Optional[int] = None,
        buffer_length: Optional[int] = None,
    ) -> List[str]:
        """
        Truncates the text to the max length
        """
        return await self.pooler.arun(
            self.truncate_batch_to_max_length,
            texts = texts,
            model = model,
            max_length = max_length,
            buffer_length = buffer_length,
        )
            


    """
    Context Managers
    """

    async def async_close(self):
        """
        Closes the OpenAI API Client.
        """
        for client in self._clients.values():
            await client.async_close()

    
    def close(self):
        """
        Closes the OpenAI API Client.
        """
        for client in self._clients.values():
            client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.async_close()

    def __getitem__(self, key: Union[int, str]) -> 'OpenAIClient':
        """
        Returns the OpenAI API Client.
        """
        if self.auto_loadbalance_clients:
            return self.apis[key]
        if isinstance(key, int):
            key = self.client_names[key]
        return self._clients[key]
    
    @property
    def client_names(self) -> List[str]:
        """
        Returns the list of client names.
        """
        return list(self._clients.keys())


    """
    Auto Rotating Functions
    """

    def chat_create(
        self, 
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
            return self.api.chat.create(input_object = input_object, parse_stream = parse_stream, auto_retry = auto_retry, auto_retry_limit = auto_retry_limit, **kwargs)

        except Exception as e:
            if not self.auto_loadbalance_clients: raise e
            self.rotate_client(verbose=verbose)
            return self.chat_create(input_object = input_object, parse_stream = parse_stream, auto_retry = auto_retry, auto_retry_limit = auto_retry_limit, **kwargs)
    

    async def async_chat_create(
        self, 
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
            return await self.api.chat.async_create(input_object = input_object, parse_stream = parse_stream, auto_retry = auto_retry, auto_retry_limit = auto_retry_limit, **kwargs)

        except Exception as e:
            if not self.auto_loadbalance_clients: raise e
            self.rotate_client(verbose=verbose)
            return await self.async_chat_create(input_object = input_object, parse_stream = parse_stream, auto_retry = auto_retry, auto_retry_limit = auto_retry_limit, **kwargs)
    
    achat_create = async_chat_create

    def create_embeddings(
        self,
        inputs: Union[str, List[str]],
        model: Optional[str] = None,
        auto_retry: Optional[bool] = True,
        strip_newlines: Optional[bool] = False,
        **kwargs,
    ) -> List[List[float]]:
        """
        Creates the embeddings

        Args:
            inputs (Union[str, List[str]]): The input text or list of input texts.
            model (str, optional): The model to use. Defaults to None.
            auto_retry (bool, optional): Whether to automatically retry the request. Defaults to True.
            strip_newlines (bool, optional): Whether to strip newlines from the input. Defaults to False.
        """
        from lazyops.utils.helpers import split_into_batches
        model = model or 'text-embedding-ada-002'
        inputs = [inputs] if isinstance(inputs, str) else inputs
        inputs = self.truncate_batch_to_max_length(
            inputs, 
            model = model, 
            **kwargs
        )
        if strip_newlines: inputs = [i.replace('\n', ' ').strip() for i in inputs]
        client = self.get_client(model = model, **kwargs)
        if not client.is_azure:
            response = client.embeddings.create(input = inputs, model = model, auto_retry = auto_retry, **kwargs)
            return response.embeddings

        embeddings = []
        # We need to split into batches of 5 for Azure
        # Azure has a limit of 5 inputs per request
        batches = split_into_batches(inputs, 5)
        for batch in batches:
            response = client.embeddings.create(input = batch, model = model, auto_retry = auto_retry, **kwargs)
            embeddings.extend(response.embeddings)
            # Shuffle the clients to load balance
            client = self.get_client(model = model, azure_required = True, **kwargs)
        return embeddings


    async def async_create_embeddings(
        self,
        inputs: Union[str, List[str]],
        model: Optional[str] = None,
        auto_retry: Optional[bool] = True,
        strip_newlines: Optional[bool] = False,
        **kwargs,
    ) -> List[List[float]]:
        """
        Creates the embeddings

        Args:
            inputs (Union[str, List[str]]): The input text or list of input texts.
            model (str, optional): The model to use. Defaults to None.
            auto_retry (bool, optional): Whether to automatically retry the request. Defaults to True.
            strip_newlines (bool, optional): Whether to strip newlines from the input. Defaults to False.
        """
        from lazyops.utils.helpers import split_into_batches
        model = model or 'text-embedding-ada-002'
        inputs = [inputs] if isinstance(inputs, str) else inputs
        inputs = await self.atruncate_batch_to_max_length(
            inputs, 
            model = model, 
            **kwargs
        )
        if strip_newlines: inputs = [i.replace('\n', ' ').strip() for i in inputs]
        client = self.get_client(model = model, **kwargs)
        if not client.is_azure:
            response = await client.embeddings.async_create(input = inputs, model = model, auto_retry = auto_retry, **kwargs)
            return response.embeddings

        embeddings = []
        # We need to split into batches of 5 for Azure
        # Azure has a limit of 5 inputs per request
        batches = split_into_batches(inputs, 5)
        for batch in batches:
            response = await client.embeddings.async_create(input = batch, model = model, auto_retry = auto_retry, **kwargs)
            embeddings.extend(response.embeddings)
            # Shuffle the clients to load balance
            client = self.get_client(model = model, azure_required = True, **kwargs)
        return embeddings

    acreate_embeddings = async_create_embeddings

    
    @overload
    def configure(
        self, 
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
        # enable_rotating_clients: Optional[bool] = None,
        azure_model_mapping: Optional[Dict[str, str]] = None,

        auto_healthcheck: Optional[bool] = None,
        auto_loadbalance_clients: Optional[bool] = None,
        proxy_config: Optional[Union[Dict[str, Any], pathlib.Path]] = None,
        client_configurations: Optional[Union[Dict[str, Dict[str, Any]], pathlib.Path]] = None,
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
        # enable_rotating_clients: Optional[bool] = None,
        azure_model_mapping: Optional[Dict[str, str]] = None,
        debug_enabled: Optional[bool] = None,

        auto_healthcheck: Optional[bool] = None,
        auto_loadbalance_clients: Optional[bool] = None,
        proxy_config: Optional[Union[Dict[str, Any], pathlib.Path]] = None,
        client_configurations: Optional[Union[Dict[str, Dict[str, Any]], pathlib.Path]] = None,
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
        self, 
        on_error: Optional[Callable] = None,
        prioritize: Optional[str] = None,
        # enable_rotating_clients: Optional[bool] = None,
        azure_model_mapping: Optional[Dict[str, str]] = None,
        debug_enabled: Optional[bool] = None,

        auto_healthcheck: Optional[bool] = None,
        auto_loadbalance_clients: Optional[bool] = None,
        **kwargs
    ):
        """
        Configure the global OpenAI client.
        """
        if on_error is not None: self.on_error = on_error
        if prioritize is not None: self.prioritize = prioritize
        if debug_enabled is not None: self.settings.debug_enabled = debug_enabled
        # if enable_rotating_clients is not None: self.enable_rotating_clients = enable_rotating_clients
        if auto_loadbalance_clients is not None: self.auto_loadbalance_clients = auto_loadbalance_clients
        if auto_healthcheck is not None: self.auto_healthcheck = auto_healthcheck
        if azure_model_mapping is not None: 
            self.azure_model_mapping = azure_model_mapping
            for key, val in azure_model_mapping.items():
                ModelContextHandler.add_model(key, val)
        self.settings.configure(auto_loadbalance_clients = auto_loadbalance_clients, auto_healthcheck = auto_healthcheck, **kwargs)
