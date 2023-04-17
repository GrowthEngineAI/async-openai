import json
import aiohttpx
from pydantic import root_validator, Field
from typing import Optional, Type, Any, Union, List, Dict, Iterator
from lazyops.types import validator, lazyproperty

from async_openai.types.options import OpenAIModel, get_consumption_cost
from async_openai.types.resources import BaseResource, Usage
from async_openai.types.responses import BaseResponse
from async_openai.types.routes import BaseRoute
from async_openai.utils import logger, get_max_tokens, get_token_count



__all__ = [
    'ChatMessage',
    'ChatChoice',
    'ChatObject',
    'CompletionResponse',
    'CompletionRoute',
]

class ChatMessage(BaseResource):
    content: str
    role: Optional[str] = "user"
     
class ChatChoice(BaseResource):
    message: ChatMessage
    index: int
    logprobs: Optional[Any]
    finish_reason: Optional[str]




class ChatObject(BaseResource):
    messages: Union[List[ChatMessage], str]
    model: Optional[Union[OpenAIModel, str, Any]] = "gpt-3.5-turbo"

    max_tokens: Optional[int] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    validate_max_tokens: Optional[bool] = Field(default = True, exclude = True)

    @validator('messages', pre = True, always = True)
    def validate_messages(cls, v) -> List[ChatMessage]:
        vals = []
        if not isinstance(v, list):
            v = [v]
        for i in v:
            if isinstance(i, dict):
                vals.append(ChatMessage.parse_obj(i))
            elif isinstance(i, str):
                vals.append(ChatMessage(content = i))
            else:
                vals.append(i)
        return vals

    @validator('model', pre=True, always=True)
    def validate_model(cls, v, values: Dict[str, Any]) -> OpenAIModel:
        """
        Validate the model
        """
        if not v and values.get('engine'):
            v = values.get('engine')
        if isinstance(v, OpenAIModel):
            return v
        if isinstance(v, dict):
            return OpenAIModel(**v)
        return OpenAIModel(value = v, mode = 'chat')

    # @validator('max_tokens')
    # def validate_max_tokens(cls, v: int) -> int:
    #     """
    #     Max tokens is 4,096 / 8,192 / 32,768
    #     https://beta.openai.com/docs/api-reference/completions/create#completions/create-max-tokens
    #     """
    #     return None if v is None else max(0, min(v, 8192))
    
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
        data = super().dict(*args, exclude = exclude, **kwargs)
        # data['stream'] = False
        if data.get('model'):
            data['model'] = data['model'].src_value
        # if data['max_tokens'] is None:
        #     del data['max_tokens']
        return data
    
    @root_validator()
    def validate_obj(cls, data: Dict):
        """
        Validate the object
        """
        input_text = '\n'.join([f'{msg.role}: {msg.content}' for msg in data['messages']])
        if data['max_tokens'] is not None and data['max_tokens'] <= 0:
            data['max_tokens'] = get_max_tokens(
                text = input_text,
                model_name = data['model'].src_value,
            ) - 10
        elif data['validate_max_tokens'] and data['max_tokens']:
            data['max_tokens'] = get_max_tokens(
                text = input_text,
                model_name = data['model'].src_value,
                max_tokens = data['max_tokens']
            ) - 10
            # data['max_tokens'] = min(
            #     data['max_tokens'], 
            #     (get_max_tokens(
            #         text = input_text, 
            #         model_name = data['model'].src_value
            #     ) - 10 )
        # )
        return data



class ChatResponse(BaseResponse):
    choices: Optional[List[ChatChoice]]
    choice_model: Optional[Type[BaseResource]] = ChatChoice
    _input_object: Optional[ChatObject] = None

    @lazyproperty
    def messages(self) -> List[ChatMessage]:
        if self.choices_results:
            return [choice.message for choice in self.choices]
        return self._response.text

    @lazyproperty
    def input_text(self) -> str:
        """
        Returns the input text for the input prompt
        """
        return '\n'.join([f'{msg.role}: {msg.content}' for msg in self._input_object.messages])

    @lazyproperty
    def text(self) -> str:
        """
        Returns the text for the chat response
        """
        if self.choices_results:
            return '\n'.join([f'{msg.role}: {msg.content}' for msg in self.messages])
        return self._response.text

    @lazyproperty
    def chat_model(self):
        """
        Returns the model for the completions
        """
        return self._input_object.model or None
        # return OpenAIModel(value=self.model, mode='chat') if self.model else None
    
    @lazyproperty
    def openai_model(self):
        """
        Returns the model for the completions
        """
        return self.headers.get('openai-model', self.model)
    
    def _validate_usage(self):
        """
        Validate usage
        """
        if self.usage and self.usage.total_tokens and self.usage.prompt_tokens: return
        if self._response.status_code == 200:
            self.usage = Usage(
                prompt_tokens = get_token_count(self.input_text),
                completion_tokens = get_token_count(self.text),
            )
            self.usage.total_tokens = self.usage.prompt_tokens + self.usage.completion_tokens


    @lazyproperty
    def consumption(self) -> int:
        """
        Returns the consumption for the completions
        """ 
        self._validate_usage()
        return get_consumption_cost(
            model_name = self.openai_model,
            mode = 'chat',
            prompt_tokens = self.usage.prompt_tokens,
            completion_tokens = self.usage.completion_tokens,
            total_tokens = self.usage.total_tokens,
        )


    def dict(self, *args, exclude: Any = None, **kwargs):
        """
        Returns the dict representation of the response
        """
        data = super().dict(*args, exclude = exclude, **kwargs)
        if data.get('chat_model'):
            data['chat_model'] = data['chat_model'].dict()
        return data
    
    def handle_stream(
        self,
        response: aiohttpx.Response
    ) -> Iterator[Dict]:

        results = {}
        for line in response.iter_lines():
            if not line: continue
            if "data: [DONE]" in line:
                # return here will cause GeneratorExit exception in urllib3
                # and it will close http connection with TCP Reset
                continue
            if line.startswith("data: "):
                line = line[len("data: ") :]
            if not line.strip(): continue
            try:
                item = json.loads(line)
                self.handle_stream_metadata(item)
                for n, choice in enumerate(item['choices']):
                    if not results.get(n):
                        results[n] = {
                            'index': choice['index'],
                            'message': {
                                'role': choice['delta'].get('role', ''),
                                'content': choice['delta'].get('content', ''),
                            }
                        }
                        # every message follows <im_start>{role/name}\n{content}<im_end>\n
                        self.usage.completion_tokens += 4

                    elif choice['finish_reason'] != 'stop':
                        for k,v in choice['delta'].items():
                            if v: results[n]['message'][k] += v
                            self.usage.completion_tokens += 1

                    else:
                        results[n]['finish_reason'] = choice['finish_reason']
                        self.usage.completion_tokens += 2  # every reply is primed with <im_start>assistant
                        yield results.pop(n)

            except Exception as e:
                logger.error(f'Error: {line}: {e}')
        
        yield from results.values()
        self.usage.total_tokens = self.usage.completion_tokens
        if not self.usage.prompt_tokens:
            self.usage.prompt_tokens = get_token_count(
                text = self.input_text,
                model_name = self.chat_model.src_value,
            )


class ChatRoute(BaseRoute):
    input_model: Optional[Type[BaseResource]] = ChatObject
    response_model: Optional[Type[BaseResource]] = ChatResponse

    @lazyproperty
    def api_resource(self):
        return 'chat/completions'

    @lazyproperty
    def root_name(self):
        return 'chat'
    
    def create(
        self, 
        input_object: Optional[ChatObject] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Creates a chat response for the provided prompt and parameters

        Usage:

        ```python
        >>> result = OpenAI.chat.create(
        >>>    messages = [{'content': 'say this is a test'}],
        >>>    max_tokens = 4,
        >>>    stream = True
        >>> )
        ```

        **Parameters:**

        :model (required): ID of the model to use. You can use the List models API 
        to see all of your available models,  or see our Model overview for descriptions of them.
        Default: `gpt-3.5-turbo`
        
        :messages: The messages to generate chat completions for, in the chat format.

        :max_tokens (optional): The maximum number of tokens to generate in the completion.
        The token count of your prompt plus `max_tokens` cannot exceed the model's context length. 
        Most models have a context length of 2048 tokens (except for the newest models, which 
        support 4096 / 8182 / 32,768). If max_tokens is not provided, the model will use the maximum number of tokens
        Default: None

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

        :stream (optional): CURRENTLY NOT SUPPORTED
        Whether to stream back partial progress. 
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

        Returns: `ChatResponse`
        """
        return super().create(input_object=input_object, **kwargs)

    async def async_create(
        self, 
        input_object: Optional[ChatObject] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Creates a chat response for the provided prompt and parameters

        Usage:

        ```python
        >>> result = await OpenAI.chat.async_create(
        >>>    messages = [{'content': 'say this is a test'}],
        >>>    max_tokens = 4,
        >>>    stream = True
        >>> )
        ```

        **Parameters:**

        :model (required): ID of the model to use. You can use the List models API 
        to see all of your available models,  or see our Model overview for descriptions of them.
        Default: `gpt-3.5-turbo`
        
        :messages: The messages to generate chat completions for, in the chat format.

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

        :stream (optional): CURRENTLY NOT SUPPORTED
        Whether to stream back partial progress. 
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

        Returns: `CompletionResult`
        """
        return await super().async_create(input_object=input_object, **kwargs)