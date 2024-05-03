from __future__ import annotations

"""
OpenAI Client that supports external providers and configurations
that have OpenAI-compatible endpoints.
"""

import abc
import aiohttpx
import contextlib
from typing import Optional, Callable, Dict, Union, List

from async_openai.schemas import *
from async_openai.utils.config import get_settings, OpenAISettings
from async_openai.utils.logs import logger
from async_openai.utils.config import ProxyObject
from async_openai.utils.external_config import ExternalProviderSettings, ExternalProviderAuth
from async_openai.routes import ApiRoutes


class ExternalOpenAIClient(abc.ABC):
    """
    External Client for all the routes in the API.
    """

    is_azure: bool = False

    _client: Optional[aiohttpx.Client] = None
    _routes: Optional[ApiRoutes] = None
    _kwargs: Optional[Dict] = None

    def __init__(
        self,
        name: str,
        provider: ExternalProviderSettings,
        is_proxied: Optional[bool] = None,
        **kwargs
    ):  
        """
        Lazily Instantiates the OpenAI Client
        """
        self.name = name
        self.provider = provider
        self.debug_enabled: Optional[bool] = None
        self.on_error: Optional[Callable] = None
        self.timeout: Optional[int] = None
        self.max_retries: Optional[int] = None
        self.ignore_errors: Optional[bool] = None
        self.disable_retries: Optional[bool] = None
        self.retry_function: Optional[Callable] = None

        self.is_proxied = is_proxied if is_proxied is not None else \
           (self.provider.config.has_proxy and  '_noproxy' not in self.name)
        # logger.info(f"External Provider Configured: {self.name} [Proxied: {self.is_proxied}]")
        
        self.settings: Optional[OpenAISettings] = kwargs.pop('settings', get_settings())
        self.client_callbacks: List[Callable] = []
        self.auth: Optional[ExternalProviderAuth] = None
        self.configure_params(**kwargs)


    @property
    def client(self) -> aiohttpx.Client:
        """
        Returns the aiohttpx client
        """
        if self._client is None:
            self.configure_client()
        return self._client
    
    @property
    def routes(self) -> ApiRoutes:
        """
        Returns the routes class
        """
        if self._routes is None:
            self.configure_routes()
        return self._routes

    def configure_params(
        self, 
        debug_enabled: Optional[bool] = None,
        on_error: Optional[Callable] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        ignore_errors: Optional[bool] = None,
        disable_retries: Optional[bool] = None,
        retry_function: Optional[Callable] = None,
        auth: Optional[ExternalProviderAuth] = None,
        client_callbacks: Optional[List[Callable]] = None,
        **kwargs
    ):  # sourcery skip: low-code-quality
        """
        Helper to configure the client
        """
        
        if debug_enabled is not None:
            self.debug_enabled = debug_enabled
        elif self.debug_enabled is None:
            self.debug_enabled = self.settings.debug_enabled
        
        if timeout is not None:
            self.timeout = timeout
        elif self.timeout is None:
            self.timeout = self.settings.timeout

        if on_error is not None:
            self.on_error = on_error
        if ignore_errors is not None:
            self.ignore_errors = ignore_errors
        elif self.ignore_errors is None:
            self.ignore_errors = self.settings.ignore_errors
        if max_retries is not None:
            self.max_retries = max_retries
        elif self.max_retries is None:
            if self.provider.config.max_retries is not None:
                self.max_retries = self.provider.config.max_retries
            else:
                self.max_retries = self.settings.max_retries
        if disable_retries is not None:
            self.disable_retries = disable_retries
        elif self.disable_retries is None:
            self.disable_retries = self.settings.disable_retries
        if retry_function is not None:
            self.retry_function = retry_function
        
        if auth is not None:
            self.auth = auth
        elif self.auth is None:
            self.auth = ExternalProviderAuth(config = self.provider.config, is_proxied = self.is_proxied)
        
        if kwargs: self._kwargs = kwargs
        self.log_method = logger.info if self.debug_enabled else logger.debug
        if not self.debug_enabled:
            self.settings.disable_httpx_logger()
        
        if client_callbacks is not None:
            self.client_callbacks = client_callbacks

    def configure_client(self, **kwargs):
        """
        Helper to configure the client
        """
        if self._client is not None: return
        # logger.info(f"OpenAI Client Configured: {self.base_url} [{self.name}]")
        extra_kwargs = {}
        
        self._client = aiohttpx.Client(
            base_url = self.provider.config.proxy_url if self.is_proxied else self.provider.config.api_url,
            timeout = self.timeout,
            limits = self.settings.api_client_limits,
            auth = self.auth,
            headers = {
                'content-type': 'application/json',
            },
            **extra_kwargs,
        )
        # logger.info(f"External Configured: {self._client.base_url} [{self.name}]")

    def configure_routes(self, **kwargs):
        """
        Helper to configure the client routes
        """
        if self._routes is not None: return
        kwargs = kwargs or {}
        if self._kwargs: kwargs.update(self._kwargs)
        self._routes = ApiRoutes(
            client = self.client,
            name = self.provider.name,
            # headers = self.headers,
            debug_enabled = self.debug_enabled,
            on_error = self.on_error,
            ignore_errors = self.ignore_errors,
            timeout = self.timeout,
            max_retries = self.max_retries,
            settings = self.settings,
            disable_retries = self.disable_retries,
            retry_function = self.retry_function,
            client_callbacks = self.client_callbacks,
            route_classes = self.provider.routes.api_route_classes,
            is_azure = False,
            **kwargs
        )
        if self.debug_enabled:
            logger.info(f"[{self.name}] External Provider Configured: {self.client.base_url}")
            logger.debug(f"Debug Enabled: {self.debug_enabled}")


    def reset(
        self,
        **kwargs
    ):
        """
        Resets the client to the default settings
        """
        self._client = None
        self._routes = None
        self.configure_params(**kwargs)


    @property
    def completions(self) -> CompletionRoute:
        """
        Returns the `CompletionRoute` class for interacting with `Completions`.
        
        Doc: `https://beta.openai.com/docs/api-reference/completions`
        """
        return self.routes.completions
    
    @property
    def chat(self) -> ChatRoute:
        """
        Returns the `ChatRoute` class for interacting with `Chat` components

        Doc: `https://platform.openai.com/docs/api-reference/chat`
        """
        return self.routes.chat

    @property
    def edits(self) -> EditRoute:
        """
        Returns the `EditRoute` class for interacting with `Edits`.
        
        Doc: `https://beta.openai.com/docs/api-reference/edits`
        """
        return self.routes.edits
    
    @property
    def embeddings(self) -> EmbeddingRoute:
        """
        Returns the `EmbeddingRoute` class for interacting with `Embeddings`.
        
        Doc: `https://beta.openai.com/docs/api-reference/embeddings`
        """
        return self.routes.embeddings
    
    @property
    def images(self) -> ImageRoute:
        """
        Returns the `ImageRoute` class for interacting with `Images`.
        
        Doc: `https://beta.openai.com/docs/api-reference/images`
        """
        return self.routes.images
    
    @property
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


    def ping(self, timeout: Optional[float] = 1.0) -> bool:
        """
        Pings the API Endpoint to check if it's alive.
        """
        try:
        # with contextlib.suppress(Exception):
            response = self.client.get('/', timeout = timeout)
            data = response.json()
            # we should expect a 404 with a json response
            # if self.debug_enabled: logger.info(f"API Ping: {data}\n{response.headers}")
            if data.get('error'): return True
        except Exception as e:
            logger.error(f"API Ping Failed: {e}")
        return False
    
    async def aping(self, timeout: Optional[float] = 1.0) -> bool:
        """
        Pings the API Endpoint to check if it's alive.
        """
        try:
            response = await self.client.async_get('/', timeout = timeout)
            data = response.json()
            # we should expect a 404 with a json response
            if data.get('error'): return True
        except Exception as e:
            logger.error(f"[{self.name}] API Ping Failed: {e}")
        return False


