import json
import pathlib
from typing import Optional, Dict, Union
from lazyops.types import BaseSettings, validator, lazyproperty
from async_openai.version import VERSION
from async_openai.types.options import ApiType

_should_reset_api: bool = False

class OpenAISettings(BaseSettings):
    url: Optional[str] = None
    scheme: Optional[str] = 'http://'
    host: Optional[str] = None
    port: Optional[int] = 8080
    
    api_key: Optional[str] = None
    api_path: Optional[str] = '/v1'
    api_type: Optional[ApiType] = ApiType.open_ai
    api_version: Optional[str] = None
    api_key_path: Optional[pathlib.Path] = None
    
    organization: Optional[str] = None
    proxies: Optional[Union[Dict, str]] = None
    app_info: Optional[Dict[str, str]] = None
    debug_enabled: Optional[bool] = True
    ignore_errors: Optional[bool] = False

    timeout: Optional[int] = 600
    max_retries: Optional[int] = 3

    @validator("api_type")
    def validate_api_type(cls, v):
        if v is None: return ApiType.open_ai
        return ApiType(v) if isinstance(v, str) else v
    
    @validator("api_key")
    def validate_api_key(cls, v, values):
        if v is None and values.get('api_key_path') is not None:
            return values['api_key_path'].read_text()
        return v
    

    class Config:
        env_prefix = 'OPENAI_'
        case_sensitive = False

    @lazyproperty
    def api_url(self) -> str:
        if self.url:
            return self.url
        if self.host:
            url = f"{self.scheme}{self.host}"
            if self.port: url += f":{self.port}"
            return url
        
        # Return the official Open API URL
        return "https://api.openai.com"
    

    @lazyproperty
    def base_url(self) -> str:
        if self.api_path:
            from urllib.parse import urljoin
            return urljoin(self.api_url, self.api_path)
        return self.api_url
    
    @lazyproperty
    def base_headers(self) -> Dict[str, str]:
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
        if self.app_info:
            data["application"] = self.app_info
        return {"X-OpenAI-Client-User-Agent": json.dumps(data), "User-Agent": ua}
    
    def get_api_key_headers(
        self,
        api_key: Optional[str] = None, 
        api_type: Optional[Union[ApiType, str]] = None
    ) -> Dict[str, str]:
        if api_key is None: api_key = self.api_key
        if api_type is None: api_type = self.api_type
        api_type = api_type.value if isinstance(api_type, ApiType) else api_type
        if api_type in {"openai", "azure_ad"}:
            return {"Authorization": f"Bearer {api_key}"}
        return {"api-key": api_key}

    @lazyproperty
    def headers(self):
        _headers = self.base_headers.copy()
        if self.api_key: 
            _headers.update(self.get_api_key_headers())
        return _headers

    def get_headers(
        self, 
        api_key: Optional[str] = None, 
        api_version: Optional[str] = None,
        api_type: Optional[Union[ApiType, str]] = None,
        organization: Optional[str] = None,
        **kwargs
    ) -> Dict[str, str]:

        headers = self.headers.copy()
        if kwargs: headers.update(**kwargs)
        headers.update(self.get_api_key_headers(api_key = api_key, api_type = api_type))
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
        **kwargs
    ) -> str:
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
        **kwargs
    ) -> str:
        api_url = self.get_api_url(
            host=host,
            port=port,
            scheme=scheme,
            url=url,
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



settings = OpenAISettings()