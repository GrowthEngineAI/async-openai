import json
import logging
import pathlib
import aiohttpx
import contextlib
from typing import Optional, Dict, Union, Any
from lazyops.types import BaseSettings, validator, BaseModel, lazyproperty, Field
from lazyops.libs.proxyobj import ProxyObject
from lazyops.libs.abcs.configs.types import AppEnv
from async_openai.version import VERSION
from async_openai.types.options import ApiType

_should_reset_api: bool = False

def configure_httpx_logger(level: int = logging.ERROR):
    """
    Helper Method to Disable HTTPX Logger
    """
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(level)


class OpenAIContext(BaseModel):
    """
    A context object for OpenAI
    """
    custom_headers: Optional[Dict[str, str]] = Field(default_factory=dict)
    data: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def update_headers(self, headers: Dict[str, Any]):
        """
        Updates the custom headers
        """
        if self.custom_headers is None: self.custom_headers = {}
        for k,v in headers.items():
            if isinstance(v, str): continue
            if isinstance(v, bool):
                headers[k] = "true" if v else "false"
            if isinstance(v, (int, float, type(None))):
                headers[k] = str(v)
        self.custom_headers.update(headers)


class OpenAIAuth(aiohttpx.Auth):
    """
    Custom Authentication Wrapper for OpenAI Client
    """
    def __init__(
        self, 
        settings: Union['OpenAISettings', 'AzureOpenAISettings'],
        context: 'OpenAIContext',
        auth_key: Optional[str] = None,
        auth_value: Optional[str] = None,
        **kwargs,
    ):
        """
        Initializes the OpenAI Auth
        """
        self.settings = settings
        self.context = context
        self.auth_key = auth_key
        self.auth_value = auth_value
    
    def auth_flow(self, request):
        """
        Injects the API Key into the Request
        """
        # request.headers.update(self.settings.get_api_key_headers())
        if self.auth_key not in request.headers:
            request.headers[self.auth_key] = self.auth_value
        if custom_headers := self.context.custom_headers:
            request.headers.update(custom_headers)
        
        # if self.settings.custom_headers:
        #     request.headers.update(self.settings.custom_headers)
        yield request

    async def async_auth_flow(self, request):
        """
        Injects the API Key into the Request
        """
        if self.auth_key not in request.headers:
            request.headers[self.auth_key] = self.auth_value
        if custom_headers := self.context.custom_headers:
            request.headers.update(custom_headers)
        # request.headers.update(self.settings.get_api_key_headers())
        # request.headers[self.auth_key] = self.auth_value
        # if self.settings.custom_headers:
        #     request.headers.update(self.settings.custom_headers)
        yield request



class BaseOpenAISettings(BaseSettings):
    url: Optional[str] = None
    scheme: Optional[str] = 'http://'
    host: Optional[str] = None
    port: Optional[int] = 8080
    
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    api_path: Optional[str] = '/v1'
    api_type: Optional[ApiType] = ApiType.open_ai
    api_version: Optional[str] = None
    api_key_path: Optional[pathlib.Path] = None
    
    organization: Optional[str] = None
    proxies: Optional[Union[Dict, str]] = None
    app_info: Optional[Dict[str, str]] = None
    debug_enabled: Optional[bool] = False
    ignore_errors: Optional[bool] = False
    disable_retries: Optional[bool] = False # Allows users to customize the retry behavior

    timeout: Optional[int] = 600
    max_retries: Optional[int] = 3

    # Additional Configuration
    ## Request Pool Configuration
    max_connections: Optional[int] = 250
    max_keepalive_connections: Optional[int] = 150
    keepalive_expiry: Optional[int] = 60

    custom_headers: Optional[Dict[str, str]] = None
    limit_monitor_enabled: Optional[bool] = True

    @validator("api_type")
    def validate_api_type(cls, v):
        """
        Validates the API Type
        """
        if v is None: return ApiType.open_ai
        return ApiType(v) if isinstance(v, str) else v
    
    @validator("api_key")
    def validate_api_key(cls, v, values: Dict[str, Union[str, int, bool, pathlib.Path, Any]]):
        """
        Validates the API Key
        """
        if v is None and values.get('api_key_path') is not None:
            return values['api_key_path'].read_text()
        return v
    
    @lazyproperty
    def ctx(self) -> OpenAIContext:
        """
        Returns the context
        """
        return OpenAIContext(
            custom_headers = self.custom_headers or {},
        )

    @lazyproperty
    def api_url(self) -> str:
        """
        Returns the API URL
        """
        if self.api_base: return self.api_base
        if self.url: return self.url
        if self.host:
            url = f"{self.scheme}{self.host}"
            if self.port: url += f":{self.port}"
            return url
        
        # Return the official Open API URL
        return "https://api.openai.com"
    
    @property
    def api_client_limits(self) -> aiohttpx.Limits:
        """
        Returns the API Client Limits
        """
        return aiohttpx.Limits(
            max_connections = self.max_connections,
            max_keepalive_connections = self.max_keepalive_connections,
            keepalive_expiry = self.keepalive_expiry,
        )
    

    @lazyproperty
    def base_url(self) -> str:
        """
        Returns the Base URL
        """
        if self.api_path:
            from urllib.parse import urljoin
            return urljoin(self.api_url, self.api_path)
        return self.api_url
    
    @property
    def base_headers(self) -> Dict[str, str]:
        """
        Returns the Base Headers
        """
        if 'app_headers' not in self.ctx.data:
            import platform
            ua = f"OpenAI/v1 async_openai/{VERSION}"
            if self.app_info:
                t = ""
                if "name" in self.app_info:
                    t += self.app_info["name"]
                if "version" in self.app_info:
                    t += f"/{self.app_info['version']}"
                if "url" in self.app_info:
                    t += f" ({self.app_info['url']})"
                ua += f" {t}"
            uname_without_node = " ".join(
                v for k, v in platform.uname()._asdict().items() if k != "node"
            )
            data = {
                "bindings_version": VERSION,
                "httplib": "httpx",
                "lang": "python",
                "lang_version": platform.python_version(),
                "platform": platform.platform(),
                "publisher": "growthengineai",
                "uname": uname_without_node,
            }
            if self.app_info: data["application"] = self.app_info
            self.ctx.data['app_headers'] = {"X-OpenAI-Client-User-Agent": json.dumps(data), "User-Agent": ua}
        return self.ctx.data['app_headers']
    
    # Deprecated
    def get_api_key_headers(
        self,
        api_key: Optional[str] = None, 
        api_type: Optional[Union[ApiType, str]] = None
    ) -> Dict[str, str]:
        """
        Returns the API Key Headers
        """
        if api_key is None: api_key = self.api_key
        if api_type is None: api_type = self.api_type
        api_type = api_type.value if isinstance(api_type, ApiType) else api_type
        if api_type in {"openai", "azure_ad"}:
            return {"Authorization": f"Bearer {api_key}"}
        return {"api-key": api_key}

    def get_api_client_auth(
        self,
        api_key: Optional[str] = None, 
        api_type: Optional[Union[ApiType, str]] = None,
        **kwargs
    ) -> OpenAIAuth:
        """
        Returns the API Client Auth
        """
        if api_key is None: api_key = self.api_key
        if api_type is None: api_type = self.api_type
        api_type = api_type.value if isinstance(api_type, ApiType) else api_type
        if api_type in {"openai", "azure_ad"}:
            return OpenAIAuth(
                settings = self, context = self.ctx, auth_key = "Authorization", auth_value= f"Bearer {api_key}",
            )
        return OpenAIAuth(
            settings = self, context = self.ctx, auth_key = "api-key", auth_value = api_key,
        )

    # @property
    # def headers(self):
    #     """
    #     Returns the Headers
    #     """
    #     return self.base_headers.copy()
        # _headers = self.base_headers.copy()
        # if self.api_key: 
        #     _headers.update(self.get_api_key_headers())
        # return _headers

    def get_headers(
        self, 
        # api_key: Optional[str] = None, 
        api_version: Optional[str] = None,
        api_type: Optional[Union[ApiType, str]] = None,
        organization: Optional[str] = None,
        app_info: Optional[str] = None,
        **kwargs
    ) -> Dict[str, str]:
        # print(api_key, api_version, api_type, organization, kwargs)

        # headers = self.headers.copy()
        headers = self.base_headers.copy()
        if kwargs: headers.update(**kwargs)
        if app_info is not None: headers['application'] = app_info
        # headers.update(self.get_api_key_headers(api_key = api_key, api_type = api_type))
        if organization is None: organization = self.organization
        if api_version is None: api_version = self.api_version
        if organization:
            headers["OpenAI-Organization"] = organization
        if api_version is not None and api_type.value == 'open_ai':
            headers["OpenAI-Version"] = api_version
        if self.debug_enabled:
            headers["OpenAI-Debug"] = "true"
        headers['Content-Type'] = 'application/json'
        return headers

    def get_api_url(
        self, 
        host: Optional[str] = None, 
        port: Optional[int] = None, 
        scheme: Optional[str] = None, 
        url: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Returns the API URL
        """
        if api_base: return api_base
        if url: return url
        if host:
            url = f"{scheme or self.scheme}{host}"
            if port: url += f":{port}"
            return url
        return self.api_url

    def get_base_api_url(
        self, 
        host: Optional[str] = None, 
        port: Optional[int] = None, 
        scheme: Optional[str] = None, 
        url: Optional[str] = None,
        api_path: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Returns the Base API URL
        """
        api_url = self.get_api_url(
            host=host,
            port=port,
            scheme=scheme,
            url=url,
            api_base=api_base,
        )
        api_path = api_path or self.api_path
        if api_path:
            from urllib.parse import urljoin
            return urljoin(api_url, api_path)
        return api_url


    def configure(
        self, 
        url: Optional[str] = None,
        scheme: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
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

        **kwargs
    ):  # sourcery skip: low-code-quality
        """
        Allows Post-Init Configuration of the OpenAI Settings
        """
        if url is not None: self.url = url
        if api_base is not None: self.api_base = api_base
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
        
        if api_version is None: 
            self.api_version = self.api_type.get_version(api_version)
        else:
            self.api_version = api_version
        if organization is not None: self.organization = organization
        if proxies is not None: self.proxies = proxies
        if timeout is not None: self.timeout = timeout
        if max_retries is not None: self.max_retries = max_retries
        if app_info is not None: self.app_info = app_info
        if debug_enabled is not None: self.debug_enabled = debug_enabled
        if ignore_errors is not None: self.ignore_errors = ignore_errors
        if disable_retries is not None: self.disable_retries = disable_retries

        if max_connections is not None: self.max_connections = max_connections
        if max_keepalive_connections is not None: self.max_keepalive_connections = max_keepalive_connections
        if keepalive_expiry is not None: self.keepalive_expiry = keepalive_expiry
        if custom_headers is not None: self.ctx.update_headers(custom_headers)

    def disable_httpx_logger(self):
        """
        Disables the HTTPX Logger
        """
        configure_httpx_logger(logging.ERROR)
    
    def enable_httpx_logger(self):
        """
        Enables the HTTPX Logger
        """
        configure_httpx_logger(logging.INFO)


class AzureOpenAISettings(BaseOpenAISettings):
    """
    The Azure OpenAI Settings
    """

    api_type: Optional[ApiType] = ApiType.azure
    api_version: Optional[str] = "2023-12-01-preview"
    api_path: Optional[str] = None

    class Config:
        env_prefix = 'AZURE_OPENAI_'
        case_sensitive = False

    @property
    def is_valid(self) -> bool:
        """
        Returns whether the Azure Settings are Valid
        """
        return self.api_key is not None and (
            self.url is not None or self.api_base is not None
        )



class OpenAIProxySettings(BaseSettings):

    proxy_enabled: Optional[bool] = None
    proxy_endpoint: Optional[str] = None

    proxy_name: Optional[str] = None
    proxy_kind: Optional[str] = 'helicone'
    proxy_env_name: Optional[str] = None
    proxy_app_name: Optional[str] = None
    proxy_endpoints: Optional[Dict[str, str]] = Field(default_factory = dict)
    proxy_apikeys: Optional[Dict[str, str]] = Field(default_factory = dict)

    @property
    def endpoint(self) -> Optional[str]:
        """
        Returns the Proxy Endpoint
        """
        return self.proxy_endpoint
    
    @property
    def enabled(self) -> Optional[bool]:
        """
        Returns whether the proxy is enabled
        """
        return self.proxy_enabled

    def get_proxy_endpoint(self) -> Optional[str]:
        """
        Returns the proxy endpoint
        """
        if self.proxy_name and self.proxy_endpoints.get(self.proxy_name):
            return self.proxy_endpoints[self.proxy_name]
        for name, endpoint in self.proxy_endpoints.items():
            with contextlib.suppress(Exception):
                resp = aiohttpx.get(endpoint, timeout = 2.0)
                # data = resp.json()
                # if data.get('error'):
                self.proxy_name = name
                return endpoint
        return None

    def init(self, config_path: Optional[pathlib.Path] = None):
        """
        Initializes the core settings
        """
        if config_path: self.load_proxy_config(config_path)
        if self.proxy_endpoint is None: 
            self.proxy_endpoint = self.get_proxy_endpoint()
            self.proxy_enabled = self.proxy_endpoint is not None

    def get_apikey(
        self, source: Optional[str] = None, 
    ) -> str:
        """
        Gets the appropriate API Key for the proxy
        """
        if source:
            source = source.lower()
            for k, v in self.proxy_apikeys.items():
                if k in source: return v
        return self.proxy_apikeys.get('default', None)
    
    def load_proxy_config(
        self,
        path: pathlib.Path,
    ):
        """
        Loads the Proxy Configuration from a File
        """
        if not path.exists(): return
        data: Dict[str, Union[Dict[str, str], str]] = json.loads(path.read_text())
        for k, v in data.items():
            if v is None: continue
            if k in {'endpoint', 'enabled'}: k = f'proxy_{k}'
            if hasattr(self, k): setattr(self, k, v)
            elif hasattr(self, f'proxy_{k}'): setattr(self, f'proxy_{k}', v)
        self.proxy_endpoint = None
        self.proxy_enabled = None

    def update(
        self,
        **kwargs
    ) -> None:
        """
        Updates the Proxy Settings
        """
        for k, v in kwargs.items():
            if v is None: continue
            if k in {'endpoint', 'enabled'}: k = f'proxy_{k}'
            if hasattr(self, k): setattr(self, k, v)
            elif hasattr(self, f'proxy_{k}'): setattr(self, f'proxy_{k}', v)
        self.proxy_endpoint = None
        self.proxy_enabled = None


    def create_proxy_headers_for_helicone(
        self,
        name: str,
        config: Dict[str, Any],
        **properties: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Creates the Proxy Headers for Helicone
        """
        headers = {
            'Helicone-OpenAI-Api-Base': config.get('api_base', ''),
            'Helicone-Auth': f"Bearer {self.get_apikey(self.proxy_app_name)}",
            "Helicone-Property-ClientName": name,
            'Content-Type': 'application/json',
        }
        user_id = ''
        if self.proxy_app_name:
            headers['Helicone-Property-AppName'] = self.proxy_app_name
            user_id += self.proxy_app_name
        if self.proxy_env_name:
            headers['Helicone-Property-AppEnvironment'] = self.proxy_env_name
            if user_id: user_id += f'-{self.proxy_env_name}'
        if user_id: headers['Helicone-User-Id'] = user_id
        if 'properties' in config: properties = config.pop('properties')
        if properties:
            for k, v in properties.items():
                if 'Helicone-Property-' not in k: k = f'Helicone-Property-{k}'
                headers[k] = str(v)
        return headers

    def create_proxy_headers(
        self,
        name: str,
        config: Dict[str, Any],
        kind: Optional[str] = None,
        **properties: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Creates the Proxy Headers
        """
        if kind is None: kind = self.proxy_kind
        if kind == 'helicone':
            return self.create_proxy_headers_for_helicone(name, config, **properties)
        raise ValueError(f"Unsupported Proxy Kind: {kind}")
        
    class Config:
        # We use a different prefix here to avoid conflicts
        env_prefix = "OAI_"
        case_sensitive = False


class OpenAISettings(BaseOpenAISettings):
    """
    The OpenAI Settings
    """

    app_env: Optional[AppEnv] = None
    client_configurations: Optional[Dict[str, Dict[str, Any]]] = Field(default_factory = dict)
    auto_loadbalance_clients: Optional[bool] = True
    auto_healthcheck: Optional[bool] = True

    function_cache_enabled: Optional[bool] = True
    
    class Config:
        env_prefix = 'OPENAI_'
        case_sensitive = False


    @lazyproperty
    def azure(self) -> AzureOpenAISettings:
        """
        Returns the Azure Settings
        """
        return AzureOpenAISettings()
    
    @lazyproperty
    def proxy(self) -> OpenAIProxySettings:
        """
        Return the Proxy Settings
        """
        return OpenAIProxySettings()
    
    @property
    def has_valid_azure(self) -> bool:
        """
        Returns whether the Azure Settings are Valid
        """
        return self.azure.is_valid

    def load_client_configurations(
        self,
        path: pathlib.Path,
    ):
        """
        Loads the Client Configurations
        """
        if not path.exists(): return
        data: Dict[str, Dict[str, Any]] = json.loads(path.read_text())
        self.client_configurations.update(data)

    def configure(
        self, 
        auto_healthcheck: Optional[bool] = None,
        auto_loadbalance_clients: Optional[bool] = None,
        proxy_app_name: Optional[str] = None,
        proxy_env_name: Optional[str] = None,
        proxy_config: Optional[Union[Dict[str, Any], pathlib.Path]] = None,
        client_configurations: Optional[Union[Dict[str, Dict[str, Any]], pathlib.Path]] = None,
        **kwargs
    ):
        """
        Allows Post-Init Configuration of the OpenAI Settings

        Usage:

        ```python
        >>> settings.configure(
        >>>    api_key = 'sk-...',
        >>>    organization = 'org-...',
        >>>    max_retries = 4,
        >>>    timeout = 60,
        >>> )
        ```

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
        if auto_healthcheck is not None: self.auto_healthcheck = auto_healthcheck
        if auto_loadbalance_clients is not None: self.auto_loadbalance_clients = auto_loadbalance_clients
        if proxy_config:
            if isinstance(proxy_config, pathlib.Path):
                self.proxy.load_proxy_config(proxy_config)
            else: self.proxy.update(**proxy_config)
            self.proxy.init()
        if proxy_app_name: self.proxy.proxy_app_name = proxy_app_name
        if proxy_env_name: self.proxy.proxy_name = proxy_env_name
        if client_configurations:
            if isinstance(client_configurations, pathlib.Path):
                self.load_client_configurations(client_configurations)
            else: self.client_configurations.update(client_configurations)

        # Parse apart the azure setting configurations
        az_kwargs, rm_keys = {}, []
        for k, v in kwargs.items():
            if k.startswith('azure_'):
                az_kwargs[k[6:]] = v
                rm_keys.append(k)
            
            elif k.startswith('az_'):
                az_kwargs[k[3:]] = v
                rm_keys.append(k)
        
        # Configure the Azure Settings
        if az_kwargs: self.azure.configure(**az_kwargs)
        for k in rm_keys: kwargs.pop(k, None)
        super().configure(**kwargs)

    
    @validator('app_env', pre=True)
    def validate_app_env(cls, value: Optional[Any]) -> Any:
        """
        Validates the app environment
        """
        if value is None:
            from lazyops.libs.abcs.configs.base import get_app_env
            return get_app_env(cls.__module__)
        return AppEnv.from_env(value) if isinstance(value, str) else value

    @property
    def in_k8s(self) -> bool:
        """
        Returns whether the app is running in kubernetes
        """
        from lazyops.utils.system import is_in_kubernetes
        return is_in_kubernetes()
    
    @property
    def is_local_env(self) -> bool:
        """
        Returns whether the environment is development
        """
        return self.app_env in [AppEnv.DEVELOPMENT, AppEnv.LOCAL] and not self.in_k8s
    
    @property
    def is_production_env(self) -> bool:
        """
        Returns whether the environment is production
        """
        return self.app_env == AppEnv.PRODUCTION and self.in_k8s

    @property
    def is_development_env(self) -> bool:
        """
        Returns whether the environment is development
        """
        return self.app_env in [AppEnv.DEVELOPMENT, AppEnv.LOCAL, AppEnv.CICD]


settings: OpenAISettings = ProxyObject(OpenAISettings)

def get_settings(**kwargs) -> OpenAISettings:
    """
    Returns the OpenAI Settings
    """
    if kwargs: settings.configure(**kwargs)
    return settings

def get_default_headers() -> Dict[str, Any]:
    """
    Returns the Default Headers
    """
    return settings.get_headers()

def get_max_retries() -> int:
    """
    Returns the Max Retries
    """
    return settings.max_retries
