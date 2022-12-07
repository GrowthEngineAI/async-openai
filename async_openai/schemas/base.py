
import backoff
from async_openai.types import BaseModel

from async_openai.schemas.client import Client
from async_openai.schemas.utils import fatal_exception
from async_openai.utils import settings
from async_openai.schemas.types.base import *
from async_openai.schemas.types.exceptions import RateLimitError

__all__  = [
    'BaseSchema',
]

class BaseSchema(BaseModel):
    """
    Base class for all schemas.
    """

    @backoff.on_exception(backoff.expo, Exception, max_tries = settings.max_retries + 1, giveup = fatal_exception)
    def send(
        self,
        **kwargs,
    ):
        """
        Send the request.
        """
        headers = Client.get_headers(**kwargs)
        return Client.request(
            headers = headers,
            **kwargs
        )

    @backoff.on_exception(backoff.expo, Exception, max_tries = settings.max_retries + 1, giveup = fatal_exception)
    async def async_send(
        self,
        **kwargs,
    ):
        """
        [Async] Send the request.
        """
        headers = Client.get_headers(**kwargs)
        return await Client.async_request(
            headers = headers,
            **kwargs
        )
        
