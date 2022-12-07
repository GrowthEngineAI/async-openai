
# from .utils import merge_dicts
import httpx
import json
from typing import Any, Optional, Union, Dict
from async_openai.types import BaseModel, lazyproperty

# We'll use this to construct our exception
class ExceptionModel(BaseModel):
    response: httpx.Response
    message: Optional[str]
    data: Optional[Union[Dict, Any]]
    stream: Optional[bool] = False
    should_retry: Optional[bool] = False

    @lazyproperty
    def headers(self):
        return self.response.headers

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
        msg = self.message or ""
        msg += " " + self.error_data.get("message", "")
        if self.error_data.get("internal_message"):
            msg += "\n\n" + self.error_data["internal_message"]
        return msg



class OpenAIError(Exception):
    def __init__(
        self,
        exc: ExceptionModel,
        **kwargs
    ):
        super(OpenAIError, self).__init__(exc.error_message)
        self.exc = exc
    
    def __str__(self):
        msg = self.exc.error_message or "<empty message>"
        if self.exc.request_id is not None:
            return "Request {0}: {1}".format(self.exc.request_id, msg)
        else:
            return msg
    
    # Returns the underlying `Exception` (base class) message, which is usually
    # the raw message returned by OpenAI's API. This was previously available
    # in python2 via `error.message`. Unlike `str(error)`, it omits "Request
    # req_..." from the beginning of the string.
    @property
    def user_message(self):
        return self.exc.error_message
    
    def __repr__(self):
        return "%s(message=%r, http_status=%r, request_id=%r)" % (
            self.__class__.__name__,
            self.exc.error_message,
            self.exc.status_code,
            self.exc.request_id,
        )


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


def error_handler(
    response: httpx.Response,
    data: Optional[Any] = None, # Line specific for streaming responses
    stream: bool = False,
    should_retry: Optional[bool] = False,
    **kwargs
):
    exc = ExceptionModel(
        response = response,
        data = data,
        stream = stream,
        should_retry = should_retry,
        **kwargs
    )
    if response.status_code == 503:
        exc.message = "The server is overloaded or not ready yet."
        raise ServiceUnavailableError(
            exc = exc
        )
    if response.status_code == 429:
        raise RateLimitError(
            exc = exc
        )
    if response.status_code in [400, 404, 415]:
        return InvalidRequestError(
            exc = exc
        )
    if response.status_code == 401:
        return AuthenticationError(
            exc = exc
        )
    if response.status_code == 403:
        return PermissionError(
            exc = exc
        )
    if response.status_code == 409:
        return TryAgain(
            exc = exc
        )
    
    if stream: exc.message = "(Error occurred while streaming.)"
    raise APIError(exc = exc)



