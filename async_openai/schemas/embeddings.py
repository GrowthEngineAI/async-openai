from typing import Optional, Type, Any, Union, List
from lazyops.types import validator, lazyproperty

from async_openai.types.options import OpenAIModel, get_consumption_cost
from async_openai.types.resources import BaseResource
from async_openai.types.responses import BaseResponse
from async_openai.types.routes import BaseRoute


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
    model: Optional[Union[str, OpenAIModel, Any]] = "text-embedding-ada-002"
    input: Optional[Union[List[Any], Any]] = None
    user: Optional[str] = None

    @validator('model', pre=True, always=True)
    def validate_model(cls, v) -> OpenAIModel:
        """
        Validate the model
        """
        if isinstance(v, OpenAIModel):
            return v
        if isinstance(v, dict):
            return OpenAIModel(**v)
        return OpenAIModel(value = v, mode = 'embedding')
    

    def dict(self, *args, exclude: Any = None, **kwargs):
        """
        Returns the dict representation of the response
        """
        data = super().dict(*args, exclude = exclude, **kwargs)
        if data.get('model'):
            data['model'] = data['model'].value
        return data
    


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
        return self.headers.get('openai-model', self.input_object.model.value)

    @lazyproperty
    def consumption(self) -> int:
        """
        Returns the consumption for the completions
        """ 
        return get_consumption_cost(
            model_name = self.openai_model,
            mode = 'embedding',
            prompt_tokens = self.usage.prompt_tokens,
            completion_tokens = self.usage.completion_tokens,
            total_tokens = self.usage.total_tokens,
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
        return super().create(input_object=input_object, **kwargs)
    

    async def async_create(
        self, 
        input_object: Optional[EmbeddingObject] = None,
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
        return await super().async_create(input_object=input_object, **kwargs)