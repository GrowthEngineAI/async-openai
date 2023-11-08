import enum
import json
import time
import asyncio
import aiohttpx
import contextlib
from typing import Optional, Type, Any, Union, List, Dict, Iterator, AsyncIterator, Generator, AsyncGenerator, TYPE_CHECKING
from lazyops.types import validator, lazyproperty
from lazyops.types.models import root_validator, pre_root_validator, Field

from async_openai.types.context import ModelContextHandler
from async_openai.types.resources import BaseResource, Usage
from async_openai.types.responses import BaseResponse
from async_openai.types.routes import BaseRoute
from async_openai.types.errors import RateLimitError, APIError, MaxRetriesExceeded, InvalidMaxTokens, InvalidRequestError
from async_openai.utils import logger, parse_stream, aparse_stream


__all__ = [
    'CompletionChoice',
    'CompletionObject',
    'CompletionResponse',
    'CompletionRoute',
]


class CompletionKind(str, enum.Enum):
    # Only have one choice
    CONTENT = 'content'

class StreamedCompletionChoice(BaseResource):
    kind: CompletionKind = CompletionKind.CONTENT
    value: str


class CompletionChoice(BaseResource):
    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None



class CompletionObject(BaseResource):
    prompt: Union[List[str], str] = '<|endoftext|>'
    model: Optional[str] = "gpt-3.5-turbo-instruct"
    # prompt: Optional[Union[List[str], str]] = '<|endoftext|>'
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[List[str], str]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    best_of: Optional[int] = 1
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

    validate_max_tokens: Optional[bool] = Field(default = False, exclude = True)
    # validate_model_aliases: Optional[bool] = Field(default = False, exclude = True)

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

    
    @validator('temperature')
    def validate_temperature(cls, v: float) -> float:
        """
        Min Temperature is 0.0
        https://beta.openai.com/docs/api-reference/completions/create#completions/create-temperature
        """
        return None if v is None else max(0.0, v)
    
    @validator('top_p')
    def validate_top_p(cls, v: float) -> float:
        """
        Min Top Probability is 0.0
        https://beta.openai.com/docs/api-reference/completions/create#completions/create-top-probability
        """
        return None if v is None else max(0.0, v)

    @validator('logprobs')
    def validate_logprobs(cls, v: int) -> int:
        """
        Max logprobs is 5
        https://beta.openai.com/docs/api-reference/completions/create#completions/create-logprobs
        """
        return None if v is None else max(0, min(v, 5))
    
    @validator('presence_penalty')
    def validate_presence_penalty(cls, v: float) -> float:
        """
        Min Presence Penalty is -2.0, Max is 2.0
        https://beta.openai.com/docs/api-reference/completions/create#completions/create-presence-penalty
        """
        return None if v is None else max(0.0, min(v, 2.0))
    
    @validator('frequency_penalty')
    def validate_frequency_penalty(cls, v: float) -> float:
        """
        Min Frequency Penalty is -2.0, Max is 2.0
        https://beta.openai.com/docs/api-reference/completions/create#completions/create-frequency-penalty
        """
        return None if v is None else max(0.0, min(v, 2.0))
    

    def dict(self, *args, exclude: Any = None, **kwargs):
        """
        Returns the dict representation of the response
        """
        return super().dict(*args, exclude = exclude, **kwargs)


    

class CompletionResponse(BaseResponse):
    choices: Optional[List[CompletionChoice]] = None
    choice_model: Optional[Type[BaseResource]] = CompletionChoice
    input_object: Optional[CompletionObject] = None

    @lazyproperty
    def text(self) -> str:
        """
        Returns the text for the completions
        """
        if self.choices_results:
            return ''.join([choice.text for choice in self.choices])
        return self.response.text
    
    @lazyproperty
    def openai_model(self):
        """
        Returns the model for the completions
        """
        return self.headers.get('openai-model', self.completion_model)

    @lazyproperty
    def completion_model(self):
        """
        Returns the model for the completions
        """
        return self.input_object.model or None

    def _validate_usage(self):
        """
        Validate usage
        """
        if self.usage and self.usage.total_tokens: return
        if self.response.status_code == 200:
            self.usage = Usage(
                prompt_tokens = ModelContextHandler.count_tokens(self.input_object.prompt),
                completion_tokens = ModelContextHandler.count_tokens(self.text),
            )
            self.usage.total_tokens = self.usage.prompt_tokens + self.usage.completion_tokens

    @lazyproperty
    def consumption(self) -> int:
        """
        Returns the consumption for the completions
        """ 
        self._validate_usage()
        return ModelContextHandler.get_consumption_cost(
            model_name = self.openai_model,
            usage = self.usage,
        )

    def dict(self, *args, exclude: Any = None, **kwargs):
        """
        Returns the dict representation of the response
        """
        return super().dict(*args, exclude = exclude, **kwargs)

    def parse_stream_item(self, item: Union[Dict, Any], **kwargs) -> Optional[StreamedCompletionChoice]:
        """
        Parses a single stream item
        """
        choice = item['choices'][0]
        if choice['finish_reason'] == 'stop':
            return None
        return StreamedCompletionChoice(
            value = choice['text']
        )

    def handle_stream(
        self,
        response: aiohttpx.Response,
        streaming: Optional[bool] = False,
    ) -> Iterator[Dict]:
        """
        Handles the streaming response
        """
        texts = {}
        for line in parse_stream(response):
            try:
                item = json.loads(line)
                if streaming:
                    yield item
                self.handle_stream_metadata(item)
                for n, choice in enumerate(item['choices']):
                    if not texts.get(n):
                        texts[n] = {
                            'index': choice['index'],
                            'text': choice['text'],
                        }
                        self.usage.completion_tokens += 1
                    elif choice['finish_reason'] != 'stop':
                        texts[n]['text'] += choice['text']
                        self.usage.completion_tokens += 1

                    else:
                        texts[n]['finish_reason'] = choice['finish_reason']
                        compl = texts.pop(n)
                        if streaming:
                            self.handle_resource_item(item = compl)
                        else:
                            yield compl

            except Exception as e:
                logger.error(f'Error: {line}: {e}')

        self._stream_consumed = True
        for remaining_text in texts.values():
            if streaming:
                self.handle_resource_item(item = remaining_text)
            else:
                yield remaining_text
        self.usage.total_tokens = self.usage.completion_tokens
    

    async def ahandle_stream(
        self,
        response: aiohttpx.Response,
        streaming: Optional[bool] = False,
    ) -> AsyncIterator[Dict]:
        """
        Handles the streaming response
        """
        texts = {}
        async for line in aparse_stream(response):
            try:
                item = json.loads(line)
                if streaming:
                    yield item
                self.handle_stream_metadata(item)
                for n, choice in enumerate(item['choices']):
                    if not texts.get(n):
                        texts[n] = {
                            'index': choice['index'],
                            'text': choice['text'],
                        }
                        self.usage.completion_tokens += 1
                    elif choice['finish_reason'] != 'stop':
                        texts[n]['text'] += choice['text']
                        self.usage.completion_tokens += 1

                    else:
                        texts[n]['finish_reason'] = choice['finish_reason']
                        compl = texts.pop(n)
                        if streaming:
                            self.handle_resource_item(item = compl)
                        else:
                            yield compl

            except Exception as e:
                logger.error(f'Error: {line}: {e}')

        self._stream_consumed = True
        for remaining_text in texts.values():
            if streaming:
                self.handle_resource_item(item = remaining_text)
            else:
                yield remaining_text
        self.usage.total_tokens = self.usage.completion_tokens

    if TYPE_CHECKING:
        def stream(self, **kwargs) -> Generator[StreamedCompletionChoice, None, None]:
            """
            Streams the response
            """
            ...

        async def astream(self, **kwargs) -> AsyncGenerator[StreamedCompletionChoice, None]:
            """
            Streams the response
            """
            ...
    

class CompletionRoute(BaseRoute):
    input_model: Optional[Type[BaseResource]] = CompletionObject
    response_model: Optional[Type[BaseResource]] = CompletionResponse

    @lazyproperty
    def api_resource(self):
        return 'completions'

    @lazyproperty
    def root_name(self):
        return 'completion'
    
    def create(
        self, 
        input_object: Optional[CompletionObject] = None,
        parse_stream: Optional[bool] = True,
        auto_retry: Optional[bool] = False,
        auto_retry_limit: Optional[int] = None,
        **kwargs
    ) -> CompletionResponse:
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

        :model (required): ID of the model to use. You can use the List models API 
        to see all of your available models,  or see our Model overview for descriptions of them.
        Default: `text-curie-001`
        
        :prompt (optional): The prompt(s) to generate completions for, encoded as 
        a string, array of strings, array of tokens, or array of token arrays.
        Note that `<|endoftext|>` is the document separator that the model sees during 
        training, so if a prompt is not specified the model will generate as if from 
        the beginning of a new document.
        Default: `<|endoftext|>`

        :suffix (optional): The suffix that comes after a completion of inserted text.
        Default: `None`

        :max_tokens (optional): The maximum number of tokens to generate in the completion.
        The token count of your prompt plus `max_tokens` cannot exceed the model's context length. 
        Most models have a context length of 2048 tokens (except for the newest models, which 
        support 4096).
        Default: `16`

        :temperature (optional): What sampling temperature to use. Higher values means 
        the model will take more risks. Try 0.9 for more creative applications, and 0 (argmax sampling) 
        for ones with a well-defined answer. We generally recommend altering this or `top_p` but not both.
        Default: `1.0`

        :top_p (optional): An alternative to sampling with `temperature`, called nucleus 
        sampling, where the model considers the results of the tokens with `top_p` probability mass. 
        So `0.1` means only  the tokens comprising the top 10% probability mass are considered.
        We generally recommend altering this or `temperature` but not both
        Default: `1.0`

        :n (optional): How many completions to generate for each prompt.
        Note: Because this parameter generates many completions, it can quickly 
        consume your token quota. Use carefully and ensure that you have reasonable 
        settings for `max_tokens` and stop.
        Default: `1`

        :stream (optional): Whether to stream back partial progress. 
        If set, tokens will be sent as data-only server-sent events as they become 
        available, with the stream terminated by a `data: [DONE]` message. This is 
        handled automatically by the Client and enables faster response processing.
        Default: `False`

        :logprobs (optional): Include the log probabilities on the `logprobs` 
        most likely tokens, as well the chosen tokens. For example, if `logprobs` is 5, 
        the API will return a list of the 5 most likely tokens. The API will always 
        return the logprob of the sampled token, so there may be up to `logprobs+1` 
        elements in the response. The maximum value for `logprobs` is 5.
        Default: `None`

        :echo (optional): Echo back the prompt in addition to the completion
        Default: `False`

        :stop (optional): Up to 4 sequences where the API will stop generating 
        further tokens. The returned text will not contain the stop sequence.
        Default: `None`

        :presence_penalty (optional): Number between `-2.0` and `2.0`. Positive values 
        penalize new tokens based on whether they appear in the text so far, increasing the 
        model's likelihood to talk about new topics
        Default: `0.0`

        :frequency_penalty (optional): Number between `-2.0` and `2.0`. Positive 
        values penalize new tokens based on their existing frequency in the text so 
        far, decreasing the model's likelihood to repeat the same line verbatim.
        Default: `0.0`

        :best_of (optional): Generates `best_of` completions server-side and returns 
        the "best" (the one with the highest log probability per token). Results cannot be streamed.
        When used with `n`, `best_of` controls the number of candidate completions and n specifies how 
        many to return – `best_of` must be greater than `n`.
        Note: Because this parameter generates many completions, it can quickly consume your token quota. 
        Use carefully and ensure that you have reasonable settings for `max_tokens` and `stop`.
        Default: `1`

        :logit_bias (optional): Modify the likelihood of specified tokens appearing in the completion.
        Accepts a json object that maps tokens (specified by their token ID in the GPT tokenizer) to an associated 
        bias value from -100 to 100. You can use this tokenizer tool (which works for both GPT-2 and GPT-3) to 
        convert text to token IDs. Mathematically, the bias is added to the logits generated by the model prior 
        to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase 
        likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the 
        relevant token.
        As an example, you can pass `{"50256": -100}` to prevent the `<|endoftext|>` token from being generated.
        Default: `None`

        :user (optional): A unique identifier representing your end-user, which can help OpenAI to 
        monitor and detect abuse.
        Default: `None`

        :auto_retry (optional): Whether to automatically retry the request if it fails due to a rate limit error.

        :auto_retry_limit (optional): The maximum number of times to retry the request if it fails due to a rate limit error.

        Returns: `CompletionResult`
        """
        current_attempt = kwargs.pop('_current_attempt', 0)
        if not auto_retry:
            return super().create(input_object = input_object, parse_stream = parse_stream, **kwargs)
        
        # Handle Auto Retry Logic
        if not auto_retry_limit: auto_retry_limit = self.settings.max_retries
        try:
            return super().create(input_object = input_object, parse_stream = parse_stream, **kwargs)
        except RateLimitError as e:
            if current_attempt >= auto_retry_limit:
                raise MaxRetriesExceeded(name = self.name, attempts = current_attempt, base_exception = e) from e
            sleep_interval = e.retry_after_seconds * 1.5 if e.retry_after_seconds else 15.0
            logger.warning(f'[{self.name}: {current_attempt}/{auto_retry_limit}] Rate Limit Error. Sleeping for {sleep_interval} seconds')
            time.sleep(sleep_interval)
            current_attempt += 1
            return self.create(
                input_object = input_object,
                parse_stream = parse_stream,
                auto_retry = auto_retry,
                auto_retry_limit = auto_retry_limit,
                _current_attempt = current_attempt,
                **kwargs
            )
        except APIError as e:
            if current_attempt >= auto_retry_limit:
                raise MaxRetriesExceeded(name = self.name, attempts = current_attempt, base_exception = e) from e
            logger.warning(f'[{self.name}: {current_attempt}/{auto_retry_limit}] API Error: {e}. Sleeping for 10 seconds')
            time.sleep(10.0)
            current_attempt += 1
            return self.create(
                input_object = input_object,
                parse_stream = parse_stream,
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
                parse_stream = parse_stream,
                auto_retry = auto_retry,
                auto_retry_limit = auto_retry_limit,
                _current_attempt = current_attempt,
                **kwargs
            )


    async def async_create(
        self, 
        input_object: Optional[CompletionObject] = None,
        parse_stream: Optional[bool] = True,
        auto_retry: Optional[bool] = False,
        auto_retry_limit: Optional[int] = None,
        **kwargs
    ) -> CompletionResponse:
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

        :model (required): ID of the model to use. You can use the List models API 
        to see all of your available models,  or see our Model overview for descriptions of them.
        Default: `text-curie-001`
        
        :prompt (optional): The prompt(s) to generate completions for, encoded as 
        a string, array of strings, array of tokens, or array of token arrays.
        Note that `<|endoftext|>` is the document separator that the model sees during 
        training, so if a prompt is not specified the model will generate as if from 
        the beginning of a new document.
        Default: `<|endoftext|>`

        :suffix (optional): The suffix that comes after a completion of inserted text.
        Default: `None`

        :max_tokens (optional): The maximum number of tokens to generate in the completion.
        The token count of your prompt plus `max_tokens` cannot exceed the model's context length. 
        Most models have a context length of 2048 tokens (except for the newest models, which 
        support 4096).
        Default: `16`

        :temperature (optional): What sampling temperature to use. Higher values means 
        the model will take more risks. Try 0.9 for more creative applications, and 0 (argmax sampling) 
        for ones with a well-defined answer. We generally recommend altering this or `top_p` but not both.
        Default: `1.0`

        :top_p (optional): An alternative to sampling with `temperature`, called nucleus 
        sampling, where the model considers the results of the tokens with `top_p` probability mass. 
        So `0.1` means only  the tokens comprising the top 10% probability mass are considered.
        We generally recommend altering this or `temperature` but not both
        Default: `1.0`

        :n (optional): How many completions to generate for each prompt.
        Note: Because this parameter generates many completions, it can quickly 
        consume your token quota. Use carefully and ensure that you have reasonable 
        settings for `max_tokens` and stop.
        Default: `1`

        :stream (optional): Whether to stream back partial progress. 
        If set, tokens will be sent as data-only server-sent events as they become 
        available, with the stream terminated by a `data: [DONE]` message. This is 
        handled automatically by the Client and enables faster response processing.
        Default: `False`

        :logprobs (optional): Include the log probabilities on the `logprobs` 
        most likely tokens, as well the chosen tokens. For example, if `logprobs` is 5, 
        the API will return a list of the 5 most likely tokens. The API will always 
        return the logprob of the sampled token, so there may be up to `logprobs+1` 
        elements in the response. The maximum value for `logprobs` is 5.
        Default: `None`

        :echo (optional): Echo back the prompt in addition to the completion
        Default: `False`

        :stop (optional): Up to 4 sequences where the API will stop generating 
        further tokens. The returned text will not contain the stop sequence.
        Default: `None`

        :presence_penalty (optional): Number between `-2.0` and `2.0`. Positive values 
        penalize new tokens based on whether they appear in the text so far, increasing the 
        model's likelihood to talk about new topics
        Default: `0.0`

        :frequency_penalty (optional): Number between `-2.0` and `2.0`. Positive 
        values penalize new tokens based on their existing frequency in the text so 
        far, decreasing the model's likelihood to repeat the same line verbatim.
        Default: `0.0`

        :best_of (optional): Generates `best_of` completions server-side and returns 
        the "best" (the one with the highest log probability per token). Results cannot be streamed.
        When used with `n`, `best_of` controls the number of candidate completions and n specifies how 
        many to return – `best_of` must be greater than `n`.
        Note: Because this parameter generates many completions, it can quickly consume your token quota. 
        Use carefully and ensure that you have reasonable settings for `max_tokens` and `stop`.
        Default: `1`

        :logit_bias (optional): Modify the likelihood of specified tokens appearing in the completion.
        Accepts a json object that maps tokens (specified by their token ID in the GPT tokenizer) to an associated 
        bias value from -100 to 100. You can use this tokenizer tool (which works for both GPT-2 and GPT-3) to 
        convert text to token IDs. Mathematically, the bias is added to the logits generated by the model prior 
        to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase 
        likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the 
        relevant token.
        As an example, you can pass `{"50256": -100}` to prevent the `<|endoftext|>` token from being generated.
        Default: `None`

        :user (optional): A unique identifier representing your end-user, which can help OpenAI to 
        monitor and detect abuse.
        Default: `None`

        :auto_retry (optional): Whether to automatically retry the request if it fails due to a rate limit error.

        :auto_retry_limit (optional): The maximum number of times to retry the request if it fails due to a rate limit error.

        Returns: `CompletionResult`
        """
        
        current_attempt = kwargs.pop('_current_attempt', 0)
        if not auto_retry:
            return await super().async_create(input_object = input_object, parse_stream = parse_stream,  **kwargs)

        # Handle Auto Retry Logic
        if not auto_retry_limit: auto_retry_limit = self.settings.max_retries
        try:
            return await super().async_create(input_object = input_object, parse_stream = parse_stream, **kwargs)
        except RateLimitError as e:
            if current_attempt >= auto_retry_limit:
                raise MaxRetriesExceeded(name = self.name, attempts = current_attempt, base_exception = e) from e
            sleep_interval = e.retry_after_seconds * 1.5 if e.retry_after_seconds else 15.0
            logger.warning(f'[{self.name}: {current_attempt}/{auto_retry_limit}] Rate Limit Error. Sleeping for {sleep_interval} seconds')
            await asyncio.sleep(sleep_interval)
            current_attempt += 1
            return await self.async_create(
                input_object = input_object,
                parse_stream = parse_stream,
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
                parse_stream = parse_stream,
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
                parse_stream = parse_stream,
                auto_retry = auto_retry,
                auto_retry_limit = auto_retry_limit,
                _current_attempt = current_attempt,
                **kwargs
            )


    
