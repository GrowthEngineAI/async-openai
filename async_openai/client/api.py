import json
import httpx

from enum import Enum
from pydantic import validator
from typing import Optional, Union, List, Dict, Any, Iterator, Tuple, TypeVar, Callable, cast, overload

from async_openai.types import BaseModel
from async_openai.utils.configs import settings, ApiType, logger
from async_openai.client.utils import build_proxies, parse_stream
from async_openai.client.exceptions import *

class Method(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"


class OpenAIResponse(BaseModel):
    data: Any
    headers: Optional[Dict[str, Any]] = {}

    @property
    def request_id(self) -> Optional[str]:
        return self.headers.get("request-id")

    @property
    def organization(self) -> Optional[str]:
        return self.headers.get("OpenAI-Organization")

    @property
    def response_ms(self) -> Optional[int]:
        h = self.headers.get("Openai-Processing-Ms")
        return None if h is None else round(float(h))
    


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

    def create_request_params(
        self,
        method: Method,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        supplied_headers: Optional[Dict[str, str]] = None,
        files: Optional[Any] = None,
        request_id: Optional[str] = None,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Unified method for constructing the request parameters
        """
        method = Method(method) if isinstance(method, str) else method
        headers = self.create_headers(supplied_headers = supplied_headers, request_id = request_id)
        data = None

        if method in {method.GET, method.DELETE}:
            if params: params = {k: v for k, v in params.items() if v is not None}
        elif method in {method.POST, method.PUT}:
            if params and files: raise ValueError("At most one of params and files may be specified.")
            if params:
                data = json.dumps(params).encode()
                headers["Content-Type"] = "application/json"
                params = None
        else:
            raise APIConnectionError(
                "Unrecognized HTTP method %r. This may indicate a bug in the "
                "OpenAI bindings. Please contact support@openai.com for "
                "assistance." % (method.value,)
            )
        
        return {
            "method": method.value,
            "url": url,
            "params": params,
            "headers": headers,
            "data": data,
            "files": files,
            "timeout": request_timeout,
        }
    
    def _interpret_response_line(
        self, rbody, rcode, rheaders, stream: bool
    ) -> OpenAIResponse:
        # HTTP 204 response code does not have any content in the body.
        if rcode == 204:
            return OpenAIResponse(None, rheaders)

        if rcode == 503:
            raise ServiceUnavailableError(
                "The server is overloaded or not ready yet.",
                rbody,
                rcode,
                headers=rheaders,
            )
        try:
            if hasattr(rbody, "decode"):
                rbody = rbody.decode("utf-8")
            data = json.loads(rbody)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise APIError(f"HTTP code {rcode} from API ({rbody})", rbody, rcode, headers=rheaders) from e

        resp = OpenAIResponse(data, rheaders)
        # In the future, we might add a "status" parameter to errors
        # to better handle the "error while streaming" case.
        stream_error = stream and "error" in resp.data
        if stream_error or not 200 <= rcode < 300:
            raise handle_error_response(
                rbody, rcode, resp.data, rheaders, stream_error=stream_error
            )
        return resp

    def _interpret_response(
        self, result: httpx.Response, stream: bool
    ) -> Tuple[Union[OpenAIResponse, Iterator[OpenAIResponse]], bool]:
        """Returns the response(s) and a bool indicating whether it is a stream."""
        if stream and "text/event-stream" in result.headers.get("Content-Type", ""):
            return (
                self._interpret_response_line(
                    line, result.status_code, result.headers, stream=True
                )
                for line in parse_stream(result.iter_lines())
            ), True
        else:
            return (
                self._interpret_response_line(
                    result.content, result.status_code, result.headers, stream=False
                ),
                False,
            )


    def request_raw(
        self,
        method: Method,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        supplied_headers: Optional[Dict[str, str]] = None,
        files: Optional[Any] = None,
        stream: Optional[bool] = False,
        request_id: Optional[str] = None,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = None,
        **kwargs
    ) -> httpx.Response:
        """
        The sync raw request method.
        """
        request_params = self.create_request_params(
            method = method,
            url = url,
            params = params,
            supplied_headers = supplied_headers,
            files = files,
            request_id = request_id,
            request_timeout = request_timeout,
        )
        req_method = self._client.stream if stream else self._client.request
        try:
            return req_method(**request_params)
        except httpx.TimeoutException as e:
            raise APIConnectionError("Request timed out.") from e
        except httpx.HTTPError as e:
            raise APIConnectionError("Unexpected error communicating with OpenAI.") from e


    def request(
        self,
        method: Method,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        supplied_headers: Optional[Dict[str, str]] = None,
        files: Optional[Any] = None,
        stream: Optional[bool] = False,
        request_id: Optional[str] = None,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = None,
        **kwargs
    ) ->  Tuple[Union[OpenAIResponse, Iterator[OpenAIResponse]], bool, str]:
        """
        The sync request method.
        """
        result = self.request_raw(
            method = method,
            url = url,
            params = params,
            supplied_headers = supplied_headers,
            files = files,
            stream = stream,
            request_id = request_id,
            request_timeout = request_timeout,
        )
        resp, got_stream = self._interpret_response(result, stream)
        return resp, got_stream, self.api_key

    """
    Async Methods
    """

    async def async_request_raw(
        self,
        method: Method,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        supplied_headers: Optional[Dict[str, str]] = None,
        files: Optional[Any] = None,
        stream: Optional[bool] = False,
        request_id: Optional[str] = None,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = None,
        **kwargs
    ) -> httpx.Response:
        """
        The sync raw request method.
        """
        request_params = self.create_request_params(
            method = method,
            url = url,
            params = params,
            supplied_headers = supplied_headers,
            files = files,
            request_id = request_id,
            request_timeout = request_timeout,
        )
        req_method = self._aclient.stream if stream else self._aclient.request
        try:
            return req_method(**request_params)
        except httpx.TimeoutException as e:
            raise APIConnectionError("Request timed out.") from e
        except httpx.HTTPError as e:
            raise APIConnectionError("Unexpected error communicating with OpenAI.") from e

    async def async_request(
        self,
        method: Method,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        supplied_headers: Optional[Dict[str, str]] = None,
        files: Optional[Any] = None,
        stream: Optional[bool] = False,
        request_id: Optional[str] = None,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = None,
        **kwargs
    ) ->  Tuple[Union[OpenAIResponse, Iterator[OpenAIResponse]], bool, str]:
        """
        The sync request method.
        """
        result = await self.async_request_raw(
            method = method,
            url = url,
            params = params,
            supplied_headers = supplied_headers,
            files = files,
            stream = stream,
            request_id = request_id,
            request_timeout = request_timeout,
        )
        resp, got_stream = self._interpret_response(result, stream)
        return resp, got_stream, self.api_key


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
        if cls.api is None or _reset: cls.api = APIRequestor(**kwargs)
    
    @classmethod
    def request(
        cls,
        method: Method,
        url: str,
        *args,
        params: Optional[Dict[str, Any]] = None,
        supplied_headers: Optional[Dict[str, str]] = None,
        files: Optional[Any] = None,
        stream: Optional[bool] = False,
        request_id: Optional[str] = None,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = None,
        **kwargs
    ) ->  Tuple[Union[OpenAIResponse, Iterator[OpenAIResponse]], bool, str]:
        """
        The sync request method.
        """
        cls.init_api()
        return cls.api.request(
            *args,
            method = method,
            url = url,
            params = params,
            supplied_headers = supplied_headers,
            files = files,
            stream = stream,
            request_id = request_id,
            request_timeout = request_timeout,
            **kwargs
        )

    @classmethod
    async def async_request(
        cls,
        method: Method,
        url: str,
        *args,
        params: Optional[Dict[str, Any]] = None,
        supplied_headers: Optional[Dict[str, str]] = None,
        files: Optional[Any] = None,
        stream: Optional[bool] = False,
        request_id: Optional[str] = None,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = None,
        **kwargs
    ) ->  Tuple[Union[OpenAIResponse, Iterator[OpenAIResponse]], bool, str]:
        """
        The async request method.
        """
        cls.init_api()
        return await cls.api.async_request(
            *args,
            method = method,
            url = url,
            params = params,
            supplied_headers = supplied_headers,
            files = files,
            stream = stream,
            request_id = request_id,
            request_timeout = request_timeout,
            **kwargs
        )


