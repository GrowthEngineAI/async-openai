
import json
import aiohttpx
from typing import Any, Optional, Union, Dict
from lazyops.types import BaseModel, lazyproperty

__all__ = [
    "OpenAIError",
    "ExceptionModel",
    "fatal_exception",
    "APIError",
    "TryAgain",
    "APIConnectionError",
    "Timeout",
    "InvalidRequestError",
    "AuthenticationError",
    "PermissionError",
    "RateLimitError",
    "ServiceUnavailableError",
    "InvalidAPIType",
    "error_handler",
]

class ExceptionModel(BaseModel):
    response: aiohttpx.Response
    data: Optional[Union[Dict, Any]]
    message: Optional[str] = None
    should_retry: Optional[bool] = False

    @lazyproperty
    def headers(self):
        return self.response.headers
    
    @lazyproperty
    def stream(self) -> bool:
        return "text/event-stream" in self.headers.get("content-type", "")

    @lazyproperty
    def response_data(self):
        return self.data or self.response.json()
    
    @lazyproperty
    def http_body(self):
        body = self.data if self.stream else self.response.content
        try:
            if hasattr(body, "decode"):
                body = body.decode("utf-8")
            return body
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(
                f"HTTP code {self.status_code} from API ({body})"
            ) from e
    
    @lazyproperty
    def response_json(self):
        try:
            return json.loads(self.http_body)
        except json.JSONDecodeError:
            return {}

    @lazyproperty
    def response_text(self):
        return self.response.text
    
    @lazyproperty
    def status_code(self):
        return self.response.status_code
    
    @lazyproperty
    def error_data(self) -> Dict:
        return self.response_json.get("error", {})
    
    @lazyproperty
    def request_id(self) -> str:
        return self.headers.get("request-id", None)
    
    @lazyproperty
    def organization(self) -> str:
        return self.headers.get("openai-organization", None)
    
    @lazyproperty
    def error_message(self) -> str:
        msg = self.message or ("(Error occurred while streaming.)" if self.stream else "")
        msg += " " + self.error_data.get("message", "")
        if self.error_data.get("internal_message"):
            msg += "\n\n" + self.error_data["internal_message"]
        return msg


class OpenAIError(Exception):
    def __init__(
        self,
        response: aiohttpx.Response,
        data: Optional[Union[Dict, Any]],
        message: Optional[str] = None,
        should_retry: Optional[bool] = False,
        **kwargs
    ):
        self.status = response.status_code
        self.response = response
        self.message = message
        self.exc = ExceptionModel(
            response=response,
            message=message,
            data=data,
            should_retry=should_retry,
            **kwargs
        )
    
    def __str__(self):
        msg = self.exc.error_message or "<empty message>"
        if self.exc.request_id is not None:
            return f"Request {self.exc.request_id}: {msg}"
        else:
            return msg
        
    @property
    def user_message(self):
        return self.exc.error_message
    
    def __repr__(self):
        return f"[OpenAI] {self.__class__.__name__} \
            (message={self.exc.error_message}, \
                http_status={self.exc.status_code}, \
                    request_id={self.exc.request_id})"



class APIError(OpenAIError):
    pass


class TryAgain(OpenAIError):
    pass


class Timeout(OpenAIError):
    pass


class APIConnectionError(OpenAIError):
    pass


class InvalidRequestError(OpenAIError):
    pass


class AuthenticationError(OpenAIError):
    pass


class PermissionError(OpenAIError):
    pass


class RateLimitError(OpenAIError):
    pass


class ServiceUnavailableError(OpenAIError):
    pass


class InvalidAPIType(OpenAIError):
    pass


def fatal_exception(exc):
    if isinstance(exc, OpenAIError):
        # retry on server errors and client errors
        # with 429 status code (rate limited),
        # don't retry on other client errors
        return (400 <= exc.status < 500) and exc.status != 429
    else:
        # retry on all other errors (eg. network)
        return False


def error_handler(
    response: aiohttpx.Response,
    data: Optional[Any] = None, # Line specific for streaming responses
    should_retry: Optional[bool] = False,
    **kwargs
):

    if response.status_code == 503:
        raise ServiceUnavailableError(
            response = response,
            message = "The server is overloaded or not ready yet.",
            data = data,
            should_retry = should_retry,
            **kwargs
        )
    if response.status_code == 429:
        raise RateLimitError(
            response = response,
            data = data,
            should_retry = should_retry,
            **kwargs
        )
    if response.status_code in [400, 404, 415]:
        return InvalidRequestError(
            response = response,
            data = data,
            should_retry = should_retry,
            **kwargs
        )
    if response.status_code == 401:
        return AuthenticationError(
            response = response,
            data = data,
            should_retry = should_retry,
            **kwargs
        )
    if response.status_code == 403:
        return PermissionError(
            response = response,
            data = data,
            should_retry = should_retry,
            **kwargs
        )
    if response.status_code == 409:
        return TryAgain(
            response = response,
            data = data,
            should_retry = should_retry,
            **kwargs
        )
    
    raise APIError(
        response = response,
        data = data,
        should_retry = should_retry,
        **kwargs
    )



