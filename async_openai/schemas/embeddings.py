from typing import Optional, Union, List, Any
from async_openai.schemas.base import BaseSchema
from async_openai.schemas.types.embeddings import *


class EmbeddingSchema(BaseSchema):

    def create(
        self,
        input: Optional[Union[List[Any], Any]],
        model: Optional[str] = 'text-similarity-babbage-001',
        user: Optional[str] = None,
        **kwargs,
    ) -> EmbeddingResult:
        """
        Get a vector representation of a given input that can be easily 
        consumed by machine learning models and algorithms.

        Usage:

        ```python
        >>> result = OpenAI.embeddings.create(
        >>>    model = 'text-similarity-babbage-001',
        >>>    input = 'The food was delicious and the waiter...'
        >>> )
        ```

        **Parameters:**

        * **input** - *(required)* Input text to get embeddings for, encoded as a string 
        or array of tokens. To get embeddings for multiple inputs in a single request, 
        pass an array of strings or array of token arrays. Each input must not exceed 2048 
        tokens in length.
        Unless you are embedding code, we suggest replacing newlines (\n) in your input with 
        a single space, as we have observed inferior results when newlines are present.

        * **model** - *(optional)* ID of the model to use. You can use the List models API 
        to see all of your available models,  or see our Model overview for descriptions of them.
        Default: `text-similarity-babbage-001`

        * **user** - *(optional)* A unique identifier representing your end-user, which can help OpenAI to 
        monitor and detect abuse.
        Default: `None`

        Returns: `EmbeddingResult`
        """
        request = EmbeddingRequest(
            model = model,
            input = input,
            user = user,
        )
        response = self.send(
            **request.create_embeddings_endpoint.get_params(**kwargs)
        )
        result = EmbeddingResult(
            _raw_request = request,
            _raw_response = response,
        )
        result.parse_result()
        return result
    
    async def async_create(
        self,
        input: Optional[Union[List[Any], Any]],
        model: Optional[str] = 'text-similarity-babbage-001',
        user: Optional[str] = None,
        **kwargs,
    ) -> EmbeddingResult:
        """
        Get a vector representation of a given input that can be easily 
        consumed by machine learning models and algorithms.

        Usage:

        ```python
        >>> result = await OpenAI.embeddings.async_create(
        >>>    model = 'text-similarity-babbage-001',
        >>>    input = 'The food was delicious and the waiter...'
        >>> )
        ```

        **Parameters:**

        * **input** - *(required)* Input text to get embeddings for, encoded as a string 
        or array of tokens. To get embeddings for multiple inputs in a single request, 
        pass an array of strings or array of token arrays. Each input must not exceed 2048 
        tokens in length.
        Unless you are embedding code, we suggest replacing newlines (\n) in your input with 
        a single space, as we have observed inferior results when newlines are present.

        * **model** - *(optional)* ID of the model to use. You can use the List models API 
        to see all of your available models,  or see our Model overview for descriptions of them.
        Default: `text-similarity-babbage-001`

        * **user** - *(optional)* A unique identifier representing your end-user, which can help OpenAI to 
        monitor and detect abuse.
        Default: `None`

        Returns: `EmbeddingResult`
        """
        request = EmbeddingRequest(
            model = model,
            input = input,
            user = user,
        )
        response = await self.async_send(
            **await request.create_embeddings_endpoint.async_get_params(**kwargs)
        )
        result = EmbeddingResult(
            _raw_request = request,
            _raw_response = response,
        )
        result.parse_result()
        return result

    


