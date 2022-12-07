from typing import Optional, Any, Union, List, Dict
from async_openai.schemas.base import BaseSchema
from async_openai.schemas.types.completions import *


class CompletionSchema(BaseSchema):

    def create(
        self,
        model: Optional[Union[str, Any]] = CompletionModels.curie,
        prompt: Optional[str] = '<|endoftext|>' ,
        suffix: Optional[str] = None,
        max_tokens: Optional[int] = 16,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 1.0,
        n: Optional[int] = 1,
        stream: Optional[bool] = False,
        logprobs: Optional[int] = None,
        echo: Optional[bool] = False,
        stop: Optional[Union[str, List[str]]] = None,
        presence_penalty: Optional[float] = 0.0,
        frequency_penalty: Optional[float] = 0.0,
        best_of: Optional[int] = 1,
        logit_bias: Optional[Dict[str, float]] = None,
        user: Optional[str] = None,
        **kwargs,
    ) -> CompletionResult:
        """
        Creates a completion for the provided prompt and parameters

        Usage:

        ```python
        >>> result = OpenAI.completions.create(
        >>>    prompt = 'say this is a test',
        >>>    max_tokens = 4,
        >>>    stream = True
        >>> )
        ```

        **Parameters:**

        * **model** - *(required)* ID of the model to use. You can use the List models API 
        to see all of your available models,  or see our Model overview for descriptions of them.
        
        * **prompt** - *(optional)* The prompt(s) to generate completions for, encoded as a string, 
        array of strings, array of tokens, or array of token arrays.
        Note that <|endoftext|> is the document separator that the model sees during 
        training, so if a prompt is not specified the model will generate as if from 
        the beginning of a new document.
        Default: `<|endoftext|>`

        * **suffix** - *(optional)* The suffix that comes after a completion of inserted text.
        Default: `None`

        """
        request = CompletionRequest(
            model = model,
            prompt = prompt,
            suffix = suffix,
            max_tokens = max_tokens,
            temperature = temperature,
            top_p = top_p,
            n = n,
            stream = stream,
            logprobs = logprobs,
            echo = echo,
            stop = stop,
            presence_penalty = presence_penalty,
            frequency_penalty = frequency_penalty,
            best_of = best_of,
            logit_bias = logit_bias,
            user = user,
        )
        response = self.send(
            **request.create_endpoint.get_params(**kwargs)
        )
        result = CompletionResult(
            _raw_request = request,
            _raw_response = response,
        )
        result.parse_result()
        return result
    
    async def async_create(
        self,
        model: Optional[Union[str, Any]] = CompletionModels.curie,
        prompt: Optional[str] = '<|endoftext|>' ,
        suffix: Optional[str] = None,
        max_tokens: Optional[int] = 16,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 1.0,
        n: Optional[int] = 1,
        stream: Optional[bool] = False,
        logprobs: Optional[int] = None,
        echo: Optional[bool] = False,
        stop: Optional[Union[str, List[str]]] = None,
        presence_penalty: Optional[float] = 0.0,
        frequency_penalty: Optional[float] = 0.0,
        best_of: Optional[int] = 1,
        logit_bias: Optional[Dict[str, float]] = None,
        user: Optional[str] = None,
        **kwargs,
    ) -> CompletionResult:
        """
        [Async] Create a completion.
        """
        request = CompletionRequest(
            model = model,
            prompt = prompt,
            suffix = suffix,
            max_tokens = max_tokens,
            temperature = temperature,
            top_p = top_p,
            n = n,
            stream = stream,
            logprobs = logprobs,
            echo = echo,
            stop = stop,
            presence_penalty = presence_penalty,
            frequency_penalty = frequency_penalty,
            best_of = best_of,
            logit_bias = logit_bias,
            user = user,
        )
        response = await self.async_send(
            **await request.create_endpoint.async_get_params(**kwargs)
        )
        result = CompletionResult(
            _raw_request = request,
            _raw_response = response,
        )
        result.parse_result()
        return result

    


