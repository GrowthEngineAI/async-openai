import aiohttpx
import pathlib
from typing import Optional, Callable, Dict, Union, List

from lazyops.types import lazyproperty
from async_openai.schemas import *
from async_openai.types.options import ApiType
from async_openai.utils.logs import logger
from async_openai.utils.config import settings
from async_openai.routes import ApiRoutes


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
        logger.info(f"OpenAI Client initialized: {self.client.base_url}")
        if self.debug_enabled:
            logger.debug(f"Debug Enabled: {self.debug_enabled}")


    @lazyproperty
    def completions(self) -> CompletionRoute:
        """
        Returns the `CompletionRoute` class for interacting with `Completions`.
        
        Doc: `https://beta.openai.com/docs/api-reference/completions`
        """
        return self.routes.completions
    
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



class OpenAIAPI:

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
    _api: Optional[OpenAIClient] = None

    """
    The Global Class for OpenAI API.
    """

    def configure(
        self, 
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
        if url is not None: self.url = url
        if scheme is not None: self.scheme = scheme
        if host is not None: self.host = host
        if port is not None: self.port = port
        if api_key is not None: self.api_key = api_key
        if api_path is not None: self.api_path = api_path
        if api_key_path is not None: 
            self.api_key_path = api_key_path if isinstance(api_key_path, pathlib.Path) else pathlib.Path(api_key_path)
            self.api_key = self.api_key_path.read_text().strip()
        
        if api_type is not None: 
            self.api_type = ApiType(api_type) if isinstance(api_type, str) else api_type
        
        if api_version is not None: 
            self.api_version = self.api_type.get_version(api_version)
        if organization is not None: self.organization = organization
        if proxies is not None: self.proxies = proxies
        if timeout is not None: self.timeout = timeout
        if max_retries is not None: self.max_retries = max_retries
        if app_info is not None: self.app_info = app_info
        if debug_enabled is not None: self.debug_enabled = debug_enabled
        if ignore_errors is not None: self.ignore_errors = ignore_errors
        if on_error is not None: self.on_error = on_error

        if reset: self._api = None
        if self._api is None:
            self.get_api(**kwargs)
    
    def get_api(self, **kwargs) -> OpenAIClient:
        if self._api is None:
            self.headers = settings.get_headers(
                api_key = self.api_key,
                organization = self.organization,
                api_type = self.api_type,
                api_version = self.api_version,
            )
            self._api = OpenAIClient(
                url = self.url,
                scheme = self.scheme,
                host = self.host,
                port = self.port,
                api_path = self.api_path,
                api_type = self.api_type,
                api_version = self.api_version,
                organization = self.organization,
                proxies = self.proxies,
                timeout = self.timeout,
                max_retries = self.max_retries,
                headers = self.headers,
                app_info = self.app_info,
                debug_enabled = self.debug_enabled,
                ignore_errors = self.ignore_errors,
                on_error = self.on_error,
                **kwargs
            )
        return self._api
    
    @property
    def api(self) -> OpenAIClient:
        """
        Returns the inherited OpenAI client.
        """
        if self._api is None:
            self.configure()
        return self._api
    

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
    
    @lazyproperty
    def edits(self) -> EditRoute:
        """
        Returns the `EditRoute` class for interacting with `Edits`.
        
        Doc: `https://beta.openai.com/docs/api-reference/edits`
        """
        return self.api.edits
    
    @lazyproperty
    def embeddings(self) -> EmbeddingRoute:
        """
        Returns the `EmbeddingRoute` class for interacting with `Embeddings`.
        
        Doc: `https://beta.openai.com/docs/api-reference/embeddings`
        """
        return self.api.embeddings
    
    @lazyproperty
    def images(self) -> ImageRoute:
        """
        Returns the `ImageRoute` class for interacting with `Images`.
        
        Doc: `https://beta.openai.com/docs/api-reference/images`
        """
        return self.api.images
    
    @lazyproperty
    def models(self) -> ModelRoute:
        """
        Returns the `ModelRoute` class for interacting with `models`.
        
        Doc: `https://beta.openai.com/docs/api-reference/models`
        """
        return self.api.models

    """
    Context Managers
    """

    async def async_close(self):
        if self._api is not None:
            await self._api.async_close()
    
    def close(self):
        if self._api is not None:
            self._api.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.async_close()


    
OpenAI: OpenAIAPI = OpenAIAPI()







