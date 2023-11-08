import time
import asyncio
from typing import Optional, Type, Any, Union, List, Dict
from lazyops.types import validator, lazyproperty

from async_openai.types.context import ModelContextHandler
from async_openai.types.resources import BaseResource
from async_openai.types.responses import BaseResponse
from async_openai.types.routes import BaseRoute
from async_openai.types.errors import RateLimitError, InvalidMaxTokens, InvalidRequestError, APIError, MaxRetriesExceeded
from async_openai.utils import logger

__all__ = [
    'EmbeddingData',
    'EmbeddingObject',
    'EmbeddingResponse',
    'EmbeddingRoute',
]



class EmbeddingData(BaseResource):
    object: Optional[str] = 'embedding'
    embedding: Optional[List[float]] = []
    index: Optional[int] = 0

class EmbeddingObject(BaseResource):
    model: Optional[str] = "text-embedding-ada-002"
    input: Optional[Union[List[Any], Any]] = None
    user: Optional[str] = None

    @validator('model', pre=True, always=True)
    def validate_model(cls, v, values: Dict[str, Any]) -> str:
        """
        Validate the model
        """
        if not v:
            if values.get('engine'):
                v = values.get('engine')
            elif values.get('deployment'):
                v = values.get('deployment')
        v = ModelContextHandler.resolve_model_name(v)
        # if values.get('validate_model_aliases', False):
        #     v = ModelContextHandler[v].name
        return v
    

    def dict(self, *args, exclude: Any = None, **kwargs):
        """
        Returns the dict representation of the response
        """
        return super().dict(*args, exclude = exclude, **kwargs)
    


class EmbeddingResponse(BaseResponse):
    data: Optional[List[EmbeddingData]] = None
    data_model: Optional[Type[BaseResource]] = EmbeddingData
    input_object: Optional[EmbeddingObject] = None

    @lazyproperty
    def embeddings(self) -> List[List[float]]:
        """
        Returns the text for the response
        object
        """
        if self.data:
            return [data.embedding for data in self.data] if self.data else []
        return None

    @lazyproperty
    def openai_model(self):
        """
        Returns the model for the completions
        """
        return self.headers.get('openai-model', self.input_object.model)

    @lazyproperty
    def consumption(self) -> int:
        """
        Returns the consumption for the completions
        """ 
        return ModelContextHandler.get_consumption_cost(
            model_name = self.openai_model,
            usage = self.usage,
        )



class EmbeddingRoute(BaseRoute):
    input_model: Optional[Type[BaseResource]] = EmbeddingObject
    response_model: Optional[Type[BaseResource]] = EmbeddingResponse

    @lazyproperty
    def api_resource(self):
        return 'embeddings'
    
    @lazyproperty
    def root_name(self):
        return 'embedding'


    def create(
        self, 
        input_object: Optional[EmbeddingObject] = None,
        auto_retry: Optional[bool] = False,
        auto_retry_limit: Optional[int] = None,
        **kwargs
    ) -> EmbeddingResponse:
        """
        Creates a embedding response for the provided prompt and parameters

        Usage:

        ```python
        >>> result = OpenAI.embedding.create(
        >>>    input = 'say this is a test',
        >>> )
        ```

        **Parameters:**

        :input (string, array, required): Input text to embed, encoded as a string or array of tokens. To embed multiple inputs in a single request, pass an array of strings or array of token arrays. Each input must not exceed the max input tokens for the model (8191 tokens for text-embedding-ada-002). Example Python code for counting tokens.

        :model (string, required): ID of the model to use. You can use the List models API to see all of your available models, or see our Model overview for descriptions of them.
        Default: `text-embedding-ada-002`

        :user (optional): A unique identifier representing your end-user, which can help OpenAI to 
        monitor and detect abuse.
        Default: `None`

        Returns: `EmbeddingResponse`
        """
        if self.is_azure and self.azure_model_mapping and kwargs.get('model') and kwargs['model'] in self.azure_model_mapping:
            kwargs['model'] = self.azure_model_mapping[kwargs['model']]
            kwargs['validate_model_aliases'] = False

        current_attempt = kwargs.pop('_current_attempt', 0)
        if not auto_retry:
            return super().create(input_object=input_object, **kwargs)
        
        # Handle Auto Retry Logic
        if not auto_retry_limit: auto_retry_limit = self.settings.max_retries
        try:
            return super().create(input_object = input_object, **kwargs)
        except RateLimitError as e:
            if current_attempt >= auto_retry_limit:
                raise MaxRetriesExceeded(name = self.name, attempts = current_attempt, base_exception = e) from e
            sleep_interval = e.retry_after_seconds * 1.5 if e.retry_after_seconds else 15.0
            logger.warning(f'[{self.name}: {current_attempt}/{auto_retry_limit}] Rate Limit Error. Sleeping for {sleep_interval} seconds')
            time.sleep(sleep_interval)
            current_attempt += 1
            return self.create(
                input_object = input_object,
                auto_retry = auto_retry,
                auto_retry_limit = auto_retry_limit,
                _current_attempt = current_attempt,
                **kwargs
            )

        
        except APIError as e:
            if current_attempt >= auto_retry_limit:
                raise MaxRetriesExceeded(name = self.name, attempts=current_attempt, base_exception = e) from e
            logger.warning(f'[{self.name}: {current_attempt}/{auto_retry_limit}] API Error: {e}. Sleeping for 10 seconds')
            time.sleep(10.0)
            current_attempt += 1
            return self.create(
                input_object = input_object,
                auto_retry = auto_retry,
                auto_retry_limit = auto_retry_limit,
                _current_attempt = current_attempt,
                **kwargs
            )
        
        except (InvalidMaxTokens, InvalidRequestError) as e:
            raise e
        
        except Exception as e:
            if current_attempt >= auto_retry_limit:
                raise MaxRetriesExceeded(name = self.name, attempts = current_attempt, base_exception = e) from e
            logger.warning(f'[{self.name}: {current_attempt}/{auto_retry_limit}] Unknown Error: {e}. Sleeping for 10 seconds')
            time.sleep(10.0)
            current_attempt += 1
            return self.create(
                input_object = input_object,
                auto_retry = auto_retry,
                auto_retry_limit = auto_retry_limit,
                _current_attempt = current_attempt,
                **kwargs
            )
    

    async def async_create(
        self, 
        input_object: Optional[EmbeddingObject] = None,
        auto_retry: Optional[bool] = False,
        auto_retry_limit: Optional[int] = None,
        **kwargs
    ) -> EmbeddingResponse:
        """
        Usage:

        ```python
        >>> result = OpenAI.embedding.create(
        >>>    input = 'say this is a test',
        >>> )
        ```

        **Parameters:**

        :input (string, array, required): Input text to embed, encoded as a string or array of tokens. To embed multiple inputs in a single request, pass an array of strings or array of token arrays. Each input must not exceed the max input tokens for the model (8191 tokens for text-embedding-ada-002). Example Python code for counting tokens.

        :model (string): ID of the model to use. You can use the List models API to see all of your available models, or see our Model overview for descriptions of them.
        Default: `text-embedding-ada-002`

        :user (optional): A unique identifier representing your end-user, which can help OpenAI to 
        monitor and detect abuse.
        Default: `None`

        Returns: `EmbeddingResponse`
        """
        if self.is_azure and self.azure_model_mapping and kwargs.get('model') and kwargs['model'] in self.azure_model_mapping:
            kwargs['model'] = self.azure_model_mapping[kwargs['model']]
            kwargs['validate_model_aliases'] = False

        current_attempt = kwargs.pop('_current_attempt', 0)
        if not auto_retry:
            return await super().async_create(input_object = input_object, **kwargs)

        # Handle Auto Retry Logic
        if not auto_retry_limit: auto_retry_limit = self.settings.max_retries
        try:
            return await super().async_create(input_object = input_object, **kwargs)
        except RateLimitError as e:
            if current_attempt >= auto_retry_limit:
                raise MaxRetriesExceeded(name = self.name, attempts = current_attempt, base_exception = e) from e
            sleep_interval = e.retry_after_seconds * 1.5 if e.retry_after_seconds else 15.0
            logger.warning(f'[{self.name}: {current_attempt}/{auto_retry_limit}] Rate Limit Error. Sleeping for {sleep_interval} seconds')
            await asyncio.sleep(sleep_interval)
            current_attempt += 1
            return await self.async_create(
                input_object = input_object,
                auto_retry = auto_retry,
                auto_retry_limit = auto_retry_limit,
                _current_attempt = current_attempt,
                **kwargs
            )
        except APIError as e:
            if current_attempt >= auto_retry_limit:
                raise MaxRetriesExceeded(name = self.name, attempts = current_attempt, base_exception = e) from e
            logger.warning(f'[{self.name}: {current_attempt}/{auto_retry_limit}] API Error: {e}. Sleeping for 10 seconds')
            await asyncio.sleep(10.0)
            current_attempt += 1
            return await self.async_create(
                input_object = input_object,
                auto_retry = auto_retry,
                auto_retry_limit = auto_retry_limit,
                _current_attempt = current_attempt,
                **kwargs
            )

        except (InvalidMaxTokens, InvalidRequestError) as e:
            raise e
        
        except Exception as e:
            if current_attempt >= auto_retry_limit:
                raise MaxRetriesExceeded(name = self.name, attempts = current_attempt, base_exception = e) from e
            logger.warning(f'[{self.name}: {current_attempt}/{auto_retry_limit}] Unknown Error: {e}. Sleeping for 10 seconds')
            await asyncio.sleep(10.0)
            current_attempt += 1
            return await self.async_create(
                input_object = input_object,
                auto_retry = auto_retry,
                auto_retry_limit = auto_retry_limit,
                _current_attempt = current_attempt,
                **kwargs
            )
