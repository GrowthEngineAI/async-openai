import json
import pathlib
from enum import Enum
from pydantic import BaseSettings, validator

from typing import Optional, Dict, Any, Union

from async_openai.version import VERSION
from async_openai.types.classprops import lazyproperty


class ApiType(str, Enum):
    azure = "azure"
    openai = "openai"
    open_ai = "openai"
    azure_ad = "azure_ad"
    azuread = "azure_ad"

    def get_version(self, version: Optional[str] = None):
        if self.value in {"azure", "azure_ad", "azuread"}:
            return "2022-03-01-preview"
        return version
        

_should_reset_api: bool = False

class OpenAISettings(BaseSettings):
    api_key: str = None
    api_base: str = "https://api.openai.com/v1"
    organization: Optional[str] = None
    api_type: Optional[ApiType] = ApiType.open_ai
    api_version: Optional[str] = None
    api_key_path: Optional[pathlib.Path] = None

    verify_ssl_certs: Optional[bool] = True  # No effect. Certificates are always verified.
    proxy: Optional[str] = None
    app_info: Optional[Dict[str, str]] = None
    enable_telemetry: Optional[bool] = False  # Ignored; the telemetry feature was removed.
    ca_bundle_path: Optional[pathlib.Path] = None  # No longer used, feature was removed
    debug: Optional[bool] = False
    log_level: Optional[str] = None  # Set to either 'debug' or 'info', controls console logging

    timeout_secs: Optional[int] = 600
    max_retries: Optional[int] = 3


    @validator("api_type")
    def validate_api_type(cls, v):
        if v is None: return ApiType.open_ai
        return ApiType(v) if isinstance(v, str) else v


    @validator("api_version")
    def validate_api_version(cls, v, values):
        if 'api_type' in values and values['api_type'] in {"azure", "azure_ad", "azuread"}:
            return "2022-03-01-preview"
        return v
    
    
    @validator("api_key")
    def validate_api_key(cls, v, values):
        if v is None and values.get('api_key_path') is not None:
            return values['api_key_path'].read_text()
        return v

    def should_reset_api(
        self,
    ):
        global _should_reset_api
        if _should_reset_api:
            _should_reset_api = False
            return True
        return False

    def configure(
        self, 
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_type: Optional[Union[str, ApiType]] = None,
        api_version: Optional[str] = None,
        organization: Optional[str] = None,
        proxies: Optional[Union[str, Dict]] = None,
        timeout_secs: Optional[int] = None,
        max_retries: Optional[int] = None,
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
        >>>    timeout_secs = 60,
        >>> )
        ```
        **Parameters:**

        * `api_key` - Your OpenAI API key.  Env: [`OPENAI_API_KEY`]
        * `api_base` - The base URL of the OpenAI API. Env: [`OPENAI_API_BASE`]
        * `api_type` - The OpenAI API type.  Env: [`OPENAI_API_TYPE`]
        * `api_version` - The OpenAI API version.  Env: [`OPENAI_API_VERSION`]
        * `organization` - The OpenAI organization. Env: [`OPENAI_ORGANIZATION`]
        * `proxies` - A dictionary of proxies to be used. Env: [`OPENAI_PROXIES`]
        * `timeout_secs` - The timeout in seconds to be used. Env: [`OPENAI_TIMEOUT_SECS`]
        * `max_retries` - The number of retries to be used. Env: [`OPENAI_MAX_RETRIES`]
        """
        if api_key is not None: self.api_key = api_key
        if api_base is not None: self.api_base = api_base
        if api_type is not None: 
            self.api_type = api_type if isinstance(api_type, ApiType) else ApiType(api_type)
        if api_version is not None: self.api_version = self.api_type.get_version(api_version)
        if organization is not None: self.organization = organization
        if proxies is not None: self.proxies = proxies
        if timeout_secs is not None: self.timeout_secs = timeout_secs
        if max_retries is not None: self.max_retries = max_retries
        
        for k, v in kwargs.items():
            if not hasattr(self, k):  continue
            if isinstance(getattr(self, k), pathlib.Path):
                setattr(self, k, pathlib.Path(v))
            else:
                setattr(self, k, v)

    
    @lazyproperty
    def user_agent(self) -> Dict[str, str]:
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

    def api_key_headers(self, api_key: Optional[str] = None, api_type: Optional[Union[ApiType, str]] = None) -> Dict[str, str]:
        if api_key is None: api_key = self.api_key
        if api_type is None: api_type = self.api_type

        api_type = api_type.value if isinstance(api_type, ApiType) else api_type
        if api_type in {"openai", "azure_ad"}:
            return {"Authorization": f"Bearer {api_key}"}
        return {"api-key": api_key}
        
    @lazyproperty
    def base_headers(self) -> Dict[str, Any]:
        """
        Returns the Base Headers for the Client
        """
        headers = self.user_agent
        headers.update(self.api_key_headers())
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        if self.api_version is not None and self.api_type == 'open_ai':
            headers["OpenAI-Version"] = self.api_version
        if self.debug:
            headers["OpenAI-Debug"] = "true"
        return headers

    def contruct_headers(
        self, 
        api_key: Optional[str] = None, 
        api_version: Optional[str] = None,
        api_type: Optional[Union[ApiType, str]] = None,
        organization: Optional[str] = None,
    ) -> Dict[str, str]:
        headers = self.user_agent
        headers.update(self.api_key_headers(api_key=api_key, api_type=api_type))
        if organization is None: organization = self.organization
        if api_version is None: api_version = self.api_version
        if api_type is None: api_type = self.api_type
        if organization:
            headers["OpenAI-Organization"] = organization
        if api_version is not None and api_type.value == 'open_ai':
            headers["OpenAI-Version"] = api_version
        if self.debug:
            headers["OpenAI-Debug"] = "true"
        return headers


    class Config:
        env_prefix = "OPENAI_"


settings = OpenAISettings()


