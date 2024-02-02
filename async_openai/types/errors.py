
import json
import httpx
import aiohttpx
import contextlib
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
        """
        Returns the response headers.
        """
        return self.response.headers
    
    @lazyproperty
    def stream(self) -> bool:
        """
        Returns True if the response is a streaming response.
        """
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
        msg: str = self.message or ("(Error occurred while streaming.)" if self.stream else "")
        if self.error_data.get("message"):
            msg += " " + self.error_data.get("message")
        if self.error_data.get("internal_message"):
            msg += "\n\n" + self.error_data["internal_message"]
        return msg.strip() or self.response_text


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
        self.post_init(**kwargs)
    
    def post_init(self, **kwargs):
        pass
    
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



class MaxRetriesExhausted(Exception):
    """
    Max Retries Exhausted
    """

    def __init__(self, name: str, func_name: str, model: str, attempts: int, max_attempts: int):
        self.name = name
        self.func_name = func_name
        self.model = model
        self.attempts = attempts
        self.max_attempts = max_attempts
    
    def __str__(self):
        return f"[{self.name} - {self.model}] All retries exhausted for {self.func_name}. ({self.attempts}/{self.max_attempts})"
        
    def __repr__(self):
        """
        Returns the string representation of the error.
        """
        return f"[{self.name} - {self.model}] (func_name={self.func_name}, attempts={self.attempts}, max_attempts={self.max_attempts})"
        

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
    
    def post_init(self, **kwargs):
        """
        Gets the rate limit reset time
        """
        self.retry_after_seconds: Optional[float] = None
        with contextlib.suppress(Exception):
            self.retry_after_seconds = float(self.exc.error_message.split("Please retry after", 1)[1].split("second", 1)[0].strip())

class ServiceTimeoutError(OpenAIError):
    pass


class ServiceUnavailableError(OpenAIError):
    pass


class InvalidAPIType(OpenAIError):
    pass


class InvalidMaxTokens(InvalidRequestError):
    pass

    def post_init(self, **kwargs):
        """
        Gets the maximum context length and requested max tokens
        """
        self.maximum_context_length: Optional[int] = None
        self.requested_max_tokens: Optional[int] = None
        with contextlib.suppress(Exception):
            self.maximum_context_length = int(self.exc.error_message.split("maximum context length is", 1)[1].split(" ", 1)[0].strip())
            self.requested_max_tokens = int(self.exc.error_message.split("requested", 1)[1].split(" ", 1)[0].strip())


def fatal_exception(exc) -> bool:
    """
    Checks if the exception is fatal.
    """
    print(f"Checking if exception is fatal: {exc} ({type(exc)} = ({type(exc) == aiohttpx.ReadTimeout} is readtimeout)) ")

    if isinstance(exc, aiohttpx.ReadTimeout) or type(exc) == aiohttpx.ReadTimeout:
        return True
    
    if not isinstance(exc, OpenAIError):
        # retry on all other errors (eg. network)
        return False
    
    # retry on server errors and client errors
    # with 429 status code (rate limited),
    # with 400, 404, 415 status codes (invalid request),
    # 400 can include invalid parameters, such as invalid `max_tokens`
    # don't retry on other client errors
    if isinstance(exc, (InvalidMaxTokens, InvalidRequestError, MaxRetriesExhausted)):
        return True
    
    return (400 <= exc.status < 500) and exc.status not in [429, 400, 404, 415, 524] # [429, 400, 404, 415]


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
        if 'maximum context length' in response.text:
            return InvalidMaxTokens(
                response = response,
                data = data,
                should_retry = False,
                **kwargs
            )
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
    
    # Service is likely down.
    if response.status_code == 524:
        raise ServiceTimeoutError(
            response = response,
            data = data,
            should_retry = False,
            **kwargs
        )
    
    raise APIError(
        response = response,
        data = data,
        should_retry = should_retry,
        **kwargs
    )



class MaxRetriesExceeded(Exception):
    def __init__(
        self,
        attempts: int,
        base_exception: OpenAIError,
        name: Optional[str] = None,
    ):
        self.name = name
        self.attempts = attempts
        self.ex = base_exception
    
    def __str__(self):
        return f"[{self.name}] Max {self.attempts} retries exceeded: {str(self.ex)}"
        
        
    @property
    def user_message(self):
        """
        Returns the error message.
        """
        return f"[{self.name}] Max {self.attempts} retries exceeded: {self.ex.user_message}"
    
    def __repr__(self):
        """
        Returns the string representation of the error.
        """
        return f"[{self.name}] {repr(self.ex)} (attempts={self.attempts})"
        