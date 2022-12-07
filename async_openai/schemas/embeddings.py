from typing import Optional, Union, List, Any
from async_openai.schemas.base import BaseSchema
from async_openai.schemas.types.embeddings import *


class EmbeddingSchema(BaseSchema):

    def create(
        self,
        model: Optional[str],
        input: Optional[Union[List[Any], Any]],
        user: Optional[str] = None,
        **kwargs,
    ) -> EmbeddingResult:
        """
        Create an Embedding.
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
        model: Optional[str],
        input: Optional[Union[List[Any], Any]],
        user: Optional[str] = None,
        **kwargs,
    ) -> EmbeddingResult:
        """
        [Async] Create an Embedding.
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

    


