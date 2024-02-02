import aiohttpx
import contextlib
from typing import Optional, Callable, Dict, Union, List

from async_openai.schemas import *
from async_openai.types.options import ApiType
from async_openai.utils.logs import logger
from async_openai.utils.config import get_settings, OpenAISettings, AzureOpenAISettings, OpenAIAuth, ProxyObject
from async_openai.routes import ApiRoutes
from async_openai.meta import OpenAIMetaClass
from async_openai.manager import OpenAIManager as OpenAISessionManager

_update_params = [
    'url',
    'scheme',
    'host',
    'port',
    'api_path',
    'api_base',
    'api_key',
    'api_type',
    'api_version',
    'organization',
    'proxies',
    'app_info',

]

class OpenAIClient:
    """
    Main Client for all the routes in the API.
    """

    api_key: Optional[str] = None
    url: Optional[str] = None
    scheme: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    api_base: Optional[str] = None
    api_path: Optional[str] = None
    api_type: Optional[ApiType] = None
    api_version: Optional[str] = None
    organization: Optional[str] = None
    proxies: Optional[Union[str, Dict]] = None
    app_info: Optional[Dict[str, str]] = None
    
    headers: Optional[Dict] = None
    debug_enabled: Optional[bool] = None
    on_error: Optional[Callable] = None
    timeout: Optional[int] = None
    max_retries: Optional[int] = None
    ignore_errors: Optional[bool] = None
    disable_retries: Optional[bool] = None
    retry_function: Optional[Callable] = None

    api_url: Optional[str] = None
    base_url: Optional[str] = None

    settings: Optional[OpenAISettings] = None
    name: Optional[str] = None
    is_azure: Optional[bool] = None
    azure_model_mapping: Optional[Dict[str, str]] = None

    auth: Optional[OpenAIAuth] = None
    _client: Optional[aiohttpx.Client] = None
    _routes: Optional[ApiRoutes] = None
    _kwargs: Optional[Dict] = None

    def __init__(
        self,
        **kwargs
    ):  
        """
        Lazily Instantiates the OpenAI Client
        """
        self.model_rate_limits: Dict[str, Dict[str, int]] = {}
        self.client_callbacks: List[Callable] = []
        self.configure_params(**kwargs)

    def response_event_hook(self, response: aiohttpx.Response):
        """
        Monitor the rate limits
        """
        url = response.url
        headers = response.headers
        with contextlib.suppress(Exception):
            if self.is_azure:
                model_name = str(url).split('deployments/', 1)[-1].split('/', 1)[0].strip()
            else:
                model_name = headers.get('openai-model')
            model_name = model_name.lstrip("https:").strip()
            if not model_name: return
            if model_name not in self.model_rate_limits:
                self.model_rate_limits[model_name] = {}
            for key, value in {
                'x-ratelimit-remaining-requests': 'remaining',
                'x-ratelimit-remaining-tokens': 'remaining_tokens',
                'x-ratelimit-limit-tokens': 'limit_tokens',
                'x-ratelimit-limit-requests': 'limit_requests',
            }.items():
                if key in headers:
                    self.model_rate_limits[model_name][value] = int(headers[key])
            if self.debug_enabled:
                logger.info(f"Rate Limits: {self.model_rate_limits}")
    
    async def aresponse_event_hook(self, response: aiohttpx.Response):
        """
        Monitor the rate limits
        """
        return self.response_event_hook(response)

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
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        scheme: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        api_base: Optional[str] = None,
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
        disable_retries: Optional[bool] = None,
        retry_function: Optional[Callable] = None,

        settings: Optional[OpenAISettings] = None,
        name: Optional[str] = None,
        is_azure: Optional[bool] = None,
        azure_model_mapping: Optional[Dict[str, str]] = None,
        auth: Optional[OpenAIAuth] = None,
        client_callbacks: Optional[List[Callable]] = None,
        **kwargs
    ):  # sourcery skip: low-code-quality
        """
        Helper to configure the client
        """
        if self.settings is None and settings is None:
            settings = get_settings()
        if settings is not None:
            self.settings = settings.azure if is_azure else settings
        if api_key is not None:
            self.api_key = api_key
        elif self.api_key is None:
            self.api_key = self.settings.api_key
        if api_type is not None:
            self.api_type = api_type
        elif self.api_type is None:
            self.api_type = self.settings.api_type
        if organization is not None:
            self.organization = organization
        elif self.organization is None:
            self.organization = self.settings.organization
        if proxies is not None:
            self.proxies = proxies
        elif self.proxies is None:
            self.proxies = self.settings.proxies
        if app_info is not None:
            self.app_info = app_info
        elif self.app_info is None:
            self.app_info = self.settings.app_info
        if any(
            [
                url is not None,
                scheme is not None,
                host is not None,
                port is not None,
                api_base is not None,
                self.api_url is None,
            ]
        ):
            self.api_url = self.settings.get_api_url(host = host, port = port, scheme = scheme, url = url, api_base = api_base)
        if any(
            [
                url is not None,
                scheme is not None,
                host is not None,
                port is not None,
                api_path is not None,
                api_base is not None,
                self.base_url is None,
            ]
        ):
            self.base_url = self.settings.get_base_api_url(host = host, port = port, scheme = scheme, url = url, api_path = api_path, api_base = api_base)
        
        if debug_enabled is not None:
            self.debug_enabled = debug_enabled
        elif self.debug_enabled is None:
            self.debug_enabled = self.settings.debug_enabled
        
        if timeout is not None:
            self.timeout = timeout
        elif self.timeout is None:
            self.timeout = self.settings.timeout

        if headers is not None:
            self.headers = headers
        else:
            self.headers = self.settings.get_headers(api_version = self.api_version, api_type = self.api_type, organization = self.organization, app_info = self.app_info)
            # self.headers = self.settings.get_headers(api_key = self.api_key, api_version = self.api_version, api_type = self.api_type, organization = self.organization, app_info = self.app_info)
        
        if on_error is not None:
            self.on_error = on_error
        if ignore_errors is not None:
            self.ignore_errors = ignore_errors
        elif self.ignore_errors is None:
            self.ignore_errors = self.settings.ignore_errors
        if max_retries is not None:
            self.max_retries = max_retries
        elif self.max_retries is None:
            self.max_retries = self.settings.max_retries
        if disable_retries is not None:
            self.disable_retries = disable_retries
        elif self.disable_retries is None:
            self.disable_retries = self.settings.disable_retries
        
        if retry_function is not None:
            self.retry_function = retry_function
        
        if is_azure is not None:
            self.is_azure = is_azure
        elif self.is_azure is None:
            self.is_azure = isinstance(self.settings, AzureOpenAISettings)
        if azure_model_mapping is not None:
            self.azure_model_mapping = azure_model_mapping
        if name is not None:
            self.name = name
        elif self.name is None:
            self.name = 'default'
        if api_version is not None:
            self.api_version = api_version
        elif self.api_version is None:
            self.api_version = self.settings.api_version
        
        
        if auth is not None:
            self.auth = auth
        elif self.auth is None:
            self.auth = self.settings.get_api_client_auth(api_key = self.api_key, api_type = self.api_type)

        if kwargs: self._kwargs = kwargs
        self.log_method = logger.info if self.debug_enabled else logger.debug
        if not self.debug_enabled:
            self.settings.disable_httpx_logger()
        
        if client_callbacks is not None:
            self.client_callbacks = client_callbacks
        # if self.debug_enabled:
        #     logger.info(f"OpenAI Client Configured: {self.client.base_url}")
        #     logger.debug(f"Debug Enabled: {self.debug_enabled}")

    def configure_client(self, **kwargs):
        """
        Helper to configure the client
        """
        if self._client is not None: return
        # logger.info(f"OpenAI Client Configured: {self.base_url} [{self.name}]")
        extra_kwargs = {}
        if self.settings.limit_monitor_enabled:
            extra_kwargs['event_hooks'] = {'response': [self.response_event_hook]}
            extra_kwargs['async_event_hooks'] = {'response': [self.aresponse_event_hook]}

        self._client = aiohttpx.Client(
            base_url = self.base_url,
            timeout = self.timeout,
            limits = self.settings.api_client_limits,
            auth = self.auth,
            headers = self.headers,
            **extra_kwargs,
        )

    def configure_routes(self, **kwargs):
        """
        Helper to configure the client routes
        """
        if self._routes is not None: return
        kwargs = kwargs or {}
        if self._kwargs: kwargs.update(self._kwargs)
        self._routes = ApiRoutes(
            client = self.client,
            name = self.name,
            # headers = self.headers,
            debug_enabled = self.debug_enabled,
            on_error = self.on_error,
            ignore_errors = self.ignore_errors,
            timeout = self.timeout,
            max_retries = self.max_retries,
            settings = self.settings,
            is_azure = self.is_azure,
            azure_model_mapping = self.azure_model_mapping,
            disable_retries = self.disable_retries,
            retry_function = self.retry_function,
            client_callbacks = self.client_callbacks,
            **kwargs
        )
        if self.debug_enabled:
            logger.info(f"[{self.name}] OpenAI Client Configured: {self.client.base_url} [Azure: {self.is_azure}]")
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
        with contextlib.suppress(Exception):
            response = await self.client.async_get('/', timeout = timeout)
            data = response.json()
            # we should expect a 404 with a json response
            if data.get('error'): return True
        return False


class OpenAI(metaclass = OpenAIMetaClass):
    """
    [V1] Interface for OpenAI

    Deprecating this class in future versions
    """
    pass

OpenAIManager: OpenAISessionManager = ProxyObject(OpenAISessionManager)



