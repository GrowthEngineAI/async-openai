import httpx
import asyncio
from typing import Optional, Union, List, Dict

from async_openai.utils.configs import settings, ApiType
from async_openai.schemas.utils import build_proxies


class APIRequestor:
    def __init__(
        self, 
        api_key: Optional[str] =  settings.api_key,
        api_base: Optional[str] = settings.api_base,
        api_type: Optional[Union[str, ApiType]] = settings.api_type,
        api_version: Optional[str] = None,
        organization: Optional[str] = settings.organization,
        proxies: Optional[Union[str, Dict]] = settings.proxy,
        timeout_secs: Optional[int] = settings.timeout_secs,
        **kwargs
    ):
        self.api_key = api_key or settings.api_key
        self.api_base = api_base or settings.api_base
        self.api_type = api_type if isinstance(api_type, ApiType) else ApiType(api_type)
        self.api_version = self.api_type.get_version(api_version) if api_version else settings.api_version
        self.organization = organization or settings.organization
        self.proxies = build_proxies(proxies or settings.proxy)
        self.timeout_secs = timeout_secs or settings.timeout_secs

        self.base_headers = settings.contruct_headers(
            api_key = self.api_key,
            api_type = self.api_type,
            api_version = self.api_version,
            organization = self.organization,
        )
        # print(self.base_headers)
        self._aclient = httpx.AsyncClient(
            base_url = self.api_base,
            proxies = self.proxies,
            timeout = self.timeout_secs,
        )
        self._client = httpx.Client(
            base_url = self.api_base,
            proxies = self.proxies,
            timeout = self.timeout_secs,
        )
    
    def create_headers(
        self,
        supplied_headers: Optional[Dict[str, str]] = None,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, str]:
        """
        Create the headers for the request
        """
        headers = self.base_headers.copy()
        if supplied_headers: headers.update(supplied_headers)
        if request_id: headers["X-Request-Id"] = request_id
        return headers
    
    def close(
        self,
        **kwargs
    ):
        self._client.close()
        asyncio.create_task(
            self._aclient.aclose()
        )


class Client:
    api: APIRequestor = None
    
    @classmethod
    def configure(
        cls,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_type: Optional[Union[str, ApiType]] = None,
        api_version: Optional[str] = None,
        organization: Optional[str] = None,
        proxies: Optional[Union[str, Dict]] = None,
        timeout_secs: Optional[int] = None,
        **kwargs
    ):
        _reset = api_base and api_base != settings.api_base
        settings.configure(
            api_key = api_key,
            api_base = api_base,
            api_type = api_type,
            api_version = api_version,
            organization = organization,
            proxies = proxies,
            timeout_secs = timeout_secs,
            **kwargs
        )
        cls.init_api(_reset = _reset)

    @classmethod
    def init_api(
        cls,
        _reset: bool = False,
        **kwargs
    ):
        if (cls.api is None or _reset or settings.should_reset_api()): 
            # Ensure that existing calls are first closed.
            if cls.api: cls.api.close()
            cls.api = APIRequestor(**kwargs)
    
    @classmethod
    def get_headers(
        cls,
        **kwargs
    ) -> Dict:

        cls.init_api()
        return cls.api.create_headers(**kwargs)

    @classmethod
    def request(
        cls,
        **kwargs
    ) -> httpx.Response:
        cls.init_api()
        return cls.api._client.request(**kwargs)
    
    @classmethod
    async def async_request(
        cls,
        **kwargs
    ) -> httpx.Response:
        cls.init_api()
        return await cls.api._aclient.request(**kwargs)
