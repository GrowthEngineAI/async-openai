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
        Default: `text-curie-001`
        
        * **prompt** - *(optional)* The prompt(s) to generate completions for, encoded as 
        a string, array of strings, array of tokens, or array of token arrays.
        Note that `<|endoftext|>` is the document separator that the model sees during 
        training, so if a prompt is not specified the model will generate as if from 
        the beginning of a new document.
        Default: `<|endoftext|>`

        * **suffix** - *(optional)* The suffix that comes after a completion of inserted text.
        Default: `None`

        * **max_tokens** - *(optional)* The maximum number of tokens to generate in the completion.
        The token count of your prompt plus `max_tokens` cannot exceed the model's context length. 
        Most models have a context length of 2048 tokens (except for the newest models, which 
        support 4096).
        Default: `16`

        * **temperature** - *(optional)* What sampling temperature to use. Higher values means 
        the model will take more risks. Try 0.9 for more creative applications, and 0 (argmax sampling) 
        for ones with a well-defined answer. We generally recommend altering this or `top_p` but not both.
        Default: `1.0`

        * **top_p** - *(optional)* An alternative to sampling with `temperature`, called nucleus 
        sampling, where the model considers the results of the tokens with `top_p` probability mass. 
        So `0.1` means only  the tokens comprising the top 10% probability mass are considered.
        We generally recommend altering this or `temperature` but not both
        Default: `1.0`

        * **n** - *(optional)* How many completions to generate for each prompt.
        Note: Because this parameter generates many completions, it can quickly 
        consume your token quota. Use carefully and ensure that you have reasonable 
        settings for `max_tokens` and stop.
        Default: `1`

        * **stream** - *(optional)* Whether to stream back partial progress. 
        If set, tokens will be sent as data-only server-sent events as they become 
        available, with the stream terminated by a `data: [DONE]` message. This is 
        handled automatically by the Client and enables faster response processing.
        Default: `False`

        * **logprobs** - *(optional)* Include the log probabilities on the `logprobs` 
        most likely tokens, as well the chosen tokens. For example, if `logprobs` is 5, 
        the API will return a list of the 5 most likely tokens. The API will always 
        return the logprob of the sampled token, so there may be up to `logprobs+1` 
        elements in the response. The maximum value for `logprobs` is 5.
        Default: `None`

        * **echo** - *(optional)* Echo back the prompt in addition to the completion
        Default: `False`

        * **stop** - *(optional)* Up to 4 sequences where the API will stop generating 
        further tokens. The returned text will not contain the stop sequence.
        Default: `None`

        * **presence_penalty** - *(optional)* Number between `-2.0` and `2.0`. Positive values 
        penalize new tokens based on whether they appear in the text so far, increasing the 
        model's likelihood to talk about new topics
        Default: `0.0`

        * **frequency_penalty** - *(optional)* Number between `-2.0` and `2.0`. Positive 
        values penalize new tokens based on their existing frequency in the text so 
        far, decreasing the model's likelihood to repeat the same line verbatim.
        Default: `0.0`

        * **best_of** - *(optional)* Generates `best_of` completions server-side and returns 
        the "best" (the one with the highest log probability per token). Results cannot be streamed.
        When used with `n`, `best_of` controls the number of candidate completions and n specifies how 
        many to return – `best_of` must be greater than `n`.
        Note: Because this parameter generates many completions, it can quickly consume your token quota. 
        Use carefully and ensure that you have reasonable settings for `max_tokens` and `stop`.
        Default: `1`

        * **logit_bias** - *(optional)* Modify the likelihood of specified tokens appearing in the completion.
        Accepts a json object that maps tokens (specified by their token ID in the GPT tokenizer) to an associated 
        bias value from -100 to 100. You can use this tokenizer tool (which works for both GPT-2 and GPT-3) to 
        convert text to token IDs. Mathematically, the bias is added to the logits generated by the model prior 
        to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase 
        likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the 
        relevant token.
        As an example, you can pass `{"50256": -100}` to prevent the `<|endoftext|>` token from being generated.
        Default: `None`

        * **user** - *(optional)* A unique identifier representing your end-user, which can help OpenAI to 
        monitor and detect abuse.
        Default: `None`

        Returns: `CompletionResult`
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
        Creates a completion for the provided prompt and parameters

        Usage:

        ```python
        >>> result = await OpenAI.completions.async_create(
        >>>    prompt = 'say this is a test',
        >>>    max_tokens = 4,
        >>>    stream = True
        >>> )
        ```

        **Parameters:**

        * **model** - *(required)* ID of the model to use. You can use the List models API 
        to see all of your available models,  or see our Model overview for descriptions of them.
        Default: `text-curie-001`
        
        * **prompt** - *(optional)* The prompt(s) to generate completions for, encoded as 
        a string, array of strings, array of tokens, or array of token arrays.
        Note that `<|endoftext|>` is the document separator that the model sees during 
        training, so if a prompt is not specified the model will generate as if from 
        the beginning of a new document.
        Default: `<|endoftext|>`

        * **suffix** - *(optional)* The suffix that comes after a completion of inserted text.
        Default: `None`

        * **max_tokens** - *(optional)* The maximum number of tokens to generate in the completion.
        The token count of your prompt plus `max_tokens` cannot exceed the model's context length. 
        Most models have a context length of 2048 tokens (except for the newest models, which 
        support 4096).
        Default: `16`

        * **temperature** - *(optional)* What sampling temperature to use. Higher values means 
        the model will take more risks. Try 0.9 for more creative applications, and 0 (argmax sampling) 
        for ones with a well-defined answer. We generally recommend altering this or `top_p` but not both.
        Default: `1.0`

        * **top_p** - *(optional)* An alternative to sampling with `temperature`, called nucleus 
        sampling, where the model considers the results of the tokens with `top_p` probability mass. 
        So `0.1` means only  the tokens comprising the top 10% probability mass are considered.
        We generally recommend altering this or `temperature` but not both
        Default: `1.0`

        * **n** - *(optional)* How many completions to generate for each prompt.
        Note: Because this parameter generates many completions, it can quickly 
        consume your token quota. Use carefully and ensure that you have reasonable 
        settings for `max_tokens` and stop.
        Default: `1`

        * **stream** - *(optional)* Whether to stream back partial progress. 
        If set, tokens will be sent as data-only server-sent events as they become 
        available, with the stream terminated by a `data: [DONE]` message. This is 
        handled automatically by the Client and enables faster response processing.
        Default: `False`

        * **logprobs** - *(optional)* Include the log probabilities on the `logprobs` 
        most likely tokens, as well the chosen tokens. For example, if `logprobs` is 5, 
        the API will return a list of the 5 most likely tokens. The API will always 
        return the logprob of the sampled token, so there may be up to `logprobs+1` 
        elements in the response. The maximum value for `logprobs` is 5.
        Default: `None`

        * **echo** - *(optional)* Echo back the prompt in addition to the completion
        Default: `False`

        * **stop** - *(optional)* Up to 4 sequences where the API will stop generating 
        further tokens. The returned text will not contain the stop sequence.
        Default: `None`

        * **presence_penalty** - *(optional)* Number between `-2.0` and `2.0`. Positive values 
        penalize new tokens based on whether they appear in the text so far, increasing the 
        model's likelihood to talk about new topics
        Default: `0.0`

        * **frequency_penalty** - *(optional)* Number between `-2.0` and `2.0`. Positive 
        values penalize new tokens based on their existing frequency in the text so 
        far, decreasing the model's likelihood to repeat the same line verbatim.
        Default: `0.0`

        * **best_of** - *(optional)* Generates `best_of` completions server-side and returns 
        the "best" (the one with the highest log probability per token). Results cannot be streamed.
        When used with `n`, `best_of` controls the number of candidate completions and n specifies how 
        many to return – `best_of` must be greater than `n`.
        Note: Because this parameter generates many completions, it can quickly consume your token quota. 
        Use carefully and ensure that you have reasonable settings for `max_tokens` and `stop`.
        Default: `1`

        * **logit_bias** - *(optional)* Modify the likelihood of specified tokens appearing in the completion.
        Accepts a json object that maps tokens (specified by their token ID in the GPT tokenizer) to an associated 
        bias value from -100 to 100. You can use this tokenizer tool (which works for both GPT-2 and GPT-3) to 
        convert text to token IDs. Mathematically, the bias is added to the logits generated by the model prior 
        to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase 
        likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the 
        relevant token.
        As an example, you can pass `{"50256": -100}` to prevent the `<|endoftext|>` token from being generated.
        Default: `None`

        * **user** - *(optional)* A unique identifier representing your end-user, which can help OpenAI to 
        monitor and detect abuse.
        Default: `None`

        Returns: `CompletionResult`
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

    


