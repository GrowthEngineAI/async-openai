import json
import enum
import time
import asyncio
import aiohttpx
import contextlib
from typing import Optional, Type, Any, Union, List, Dict, Iterator, TypeVar, AsyncIterator, Generator, AsyncGenerator, TYPE_CHECKING
from lazyops.types import validator, lazyproperty, Literal
from lazyops.types.models import root_validator, pre_root_validator, Field, BaseModel, PYD_VERSION, get_pyd_schema

from async_openai.types.context import ModelContextHandler
from async_openai.types.resources import BaseResource, Usage
from async_openai.types.responses import BaseResponse
from async_openai.types.routes import BaseRoute
from async_openai.types.errors import RateLimitError, InvalidMaxTokens, InvalidRequestError, APIError, MaxRetriesExceeded, ServiceTimeoutError
from async_openai.utils import logger, parse_stream, aparse_stream
from async_openai.utils.fixjson import resolve_json


__all__ = [
    'ChatMessage',
    'ChatChoice',
    'Function',
    'ChatObject',
    'CompletionResponse',
    'CompletionRoute',
]

SchemaObj = TypeVar("SchemaObj", bound=BaseModel)
SchemaType = TypeVar("SchemaType", bound=Type[BaseModel])

class MessageKind(str, enum.Enum):
    CONTENT = 'content'
    ROLE = 'role'
    FUNCTION_CALL = 'function_call'

    @classmethod
    def from_choice(cls, choice: Dict[str, Any]) -> 'MessageKind':
        """
        Returns the message kind from the choice
        """
        if 'role' in choice['delta']:
            return cls.ROLE
        elif 'content' in choice['delta']:
            return cls.CONTENT
        elif 'function_call' in choice['delta']:
            return cls.FUNCTION_CALL
        raise ValueError(f'Invalid choice: {choice}')



class StreamedChatMessage(BaseResource):
    kind: MessageKind
    value: Union[Dict[str, Any], str]

class FunctionCall(BaseResource):
    name: str
    arguments: Optional[Union[str, Dict[str, Any]]] = None

    @validator('arguments', pre = True, always = True)
    def validate_arguments(cls, v) -> Dict[str, Any]:
        """
        Try to load the arguments as json
        """
        if isinstance(v, dict):
            return v
        elif isinstance(v, str):
            with contextlib.suppress(Exception):
                return json.loads(v)
        return v
    
# TODO Add support for name
class ChatMessage(BaseResource):
    content: Optional[str] = None
    role: Optional[str] = "user"
    function_call: Optional[FunctionCall] = None
    name: Optional[str] = None

    def dict(self, *args, exclude_none: bool = True, **kwargs):
        return super().dict(*args, exclude_none = exclude_none, **kwargs)


class ChatChoice(BaseResource):
    message: ChatMessage
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None

    def __getitem__(self, key: str) -> Any:
        """
        Mimic dict
        """
        return getattr(self, key)

class Function(BaseResource):
    """
    Represents a function
    """
    # Must be a-z, A-Z, 0-9, or contain underscores and dashes
    if PYD_VERSION == 2:
        name: str = Field(..., max_length = 64, pattern = r'^[a-zA-Z0-9_]+$')
    else:
        name: str = Field(..., max_length = 64, regex = r'^[a-zA-Z0-9_]+$')
    parameters: Union[Dict[str, Any], SchemaType, str]
    description: Optional[str] = None
    source_object: Optional[Union[SchemaType, Any]] = Field(default = None, exclude = True)

    @root_validator(pre = True)
    def validate_parameters(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the parameters
        """
        if params := values.get('parameters'):
            if isinstance(params, dict):
                pass
            elif issubclass(params, BaseModel) or isinstance(params, type(BaseModel)):
                values['parameters'] = get_pyd_schema(params)
                # params.schema()
                values['source_object'] = params
            elif isinstance(params, str):
                try:
                    values['parameters'] = json.loads(params)
                except Exception as e:
                    raise ValueError(f'Invalid JSON: {params}, {e}. Must be a dict or pydantic BaseModel.') from e
            else:
                # logger.warning(f'Invalid parameters: {params}. Must be a dict or pydantic BaseModel.')
                raise ValueError(f'Parameters must be a dict or pydantic BaseModel. Provided: {type(params)}')
        return values


class Tool(BaseResource):
    """
    Represents a tool 
    """
    type: Optional[str] = 'function'
    function: Optional[Function] = None


class ChatObject(BaseResource):
    messages: Union[List[ChatMessage], str]
    model: Optional[str] = "gpt-3.5-turbo"

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

    functions: Optional[List[Function]] = None
    function_call: Optional[Union[str, Dict[str, str]]] = None

    # v2 Params
    response_format: Optional[Dict[str, Literal['json_object', 'text']]] = None
    seed: Optional[int] = None

    # tools: Optional[Union[List[Function], List[Dict[str, Union[str, Function]]]]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Union[str, Dict[str, str]]]] = None

    # Extra Params
    validate_max_tokens: Optional[bool] = Field(default = False, exclude = True)
    validate_model_aliases: Optional[bool] = Field(default = False, exclude = True)
    # api_version: Optional[str] = None

    @validator('messages', pre = True, always = True)
    def validate_messages(cls, v) -> List[ChatMessage]:
        """
        Validate the Input Messages
        """
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

    @root_validator(pre = True)
    def validate_obj(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the object
        """

        is_azure = values.pop('is_azure', False)
        
        if values.get('functions'):
            if not all(isinstance(f, Function) for f in values['functions']):
                values['functions'] = [Function(**f) for f in values['functions']]
            if not values.get('function_call'):
                values['function_call'] = 'auto'
            
            # Auto set to json_object if functions are present
            # if values.get('response_format') is None:
            #     values['response_format'] = {'type': 'json_object'}
        
        if values.get('tools'):
            tools = []
            for tool in values['tools']:
                if isinstance(tool, Tool):
                    tools.append(tool)
                elif isinstance(tool, dict):
                    # This should be the correct format
                    if tool.get('function'):
                        tools.append(Tool(**tool))
                    else:
                        # This is previously supported format
                        tools.append(Tool(function = Function(**tool)))
                else:
                    raise ValueError(f'Invalid tool: {tool}')
            values['tools'] = tools
            if not values.get('tool_choice'):
                values['tool_choice'] = 'auto'

            # Auto set to json_object if tools are present
            # if values.get('response_format') is None:
            #     values['response_format'] = {'type': 'json_object'}

        # Validate for Azure
        if is_azure and values.get('tools') and not values.get('functions'):
            # Convert tools to functions
            values['functions'] = [tool.function for tool in values['tools']]
            if not values.get('function_call'):
                values['function_call'] = 'auto'

        return values


class ChatResponse(BaseResponse):
    choices: Optional[List[ChatChoice]] = None
    choice_model: Optional[Type[BaseResource]] = ChatChoice
    input_object: Optional[ChatObject] = None

    @lazyproperty
    def messages(self) -> List[ChatMessage]:
        """
        Returns the messages for the completions
        """
        if self.choices_results:
            return [choice.message for choice in self.choices]
        return self.response.text
    
    @lazyproperty
    def function_results(self) -> List[FunctionCall]:
        """
        Returns the function results for the completions
        """
        return [msg.function_call for msg in self.messages if msg.function_call]
    
    @lazyproperty
    def function_result_objects(self) -> List[Union[SchemaObj, Dict[str, Any]]]:
        """
        Returns the function result objects for the completions
        """
        results = []
        source_function: Function = self.input_object.functions[0] if self.input_object.function_call == "auto" else (
            [
                f for f in self.input_object.functions if f.name == self.input_object.function_call['name']
            ]
        )[0]

        for func_result in self.function_results:
            if source_function.source_object:
                if not isinstance(func_result.arguments, dict):
                    try:
                        func_result.arguments = resolve_json(func_result.arguments)
                    except Exception as e:
                        logger.error('Could not resolve function arguments. Skipping.')
                        continue
                try:
                    results.append(source_function.source_object(**func_result.arguments))
                    continue
                except Exception as e:
                    logger.error(e)
            if isinstance(func_result.arguments, dict):
                results.append(func_result.arguments)
            else:
                try:
                    results.append(resolve_json(func_result.arguments))
                except Exception as e:

                    logger.error(e)
                    results.append(func_result.arguments)

        return results
    
    @lazyproperty
    def has_functions(self) -> bool:
        """
        Returns whether the response has functions
        """
        return bool(self.input_object.functions)
    

    @lazyproperty
    def tool_results(self) -> List[FunctionCall]:
        """
        Returns the tool results for the completions
        """
        return [msg.function_call for msg in self.messages if msg.function_call]
    
    @lazyproperty
    def tool_result_objects(self) -> List[Union[SchemaObj, Dict[str, Any]]]:
        """
        Returns the tool result objects for the completions
        """
        results = []
        source_function: Function = self.input_object.tools[0].function.source_object if self.input_object.tool_choice == "auto" else (
            [
                t.function for t in self.input_object.tools if t.function.name == self.input_object.tool_choice['function']['name']
            ]
        )[0]

        for tool_result in self.tool_results:
            if source_function.source_object:
                if not isinstance(tool_result.arguments, dict):
                    try:
                        tool_result.arguments = resolve_json(tool_result.arguments)
                    except Exception as e:
                        logger.error('Could not resolve function arguments. Skipping.')
                        continue
                try:
                    results.append(source_function.source_object(**tool_result.arguments))
                    continue
                except Exception as e:
                    logger.error(e)
            if isinstance(tool_result.arguments, dict):
                results.append(tool_result.arguments)
            else:
                try:
                    results.append(resolve_json(tool_result.arguments))
                except Exception as e:

                    logger.error(e)
                    results.append(tool_result.arguments)

        return results

    @lazyproperty
    def has_tools(self) -> bool:
        """
        Returns whether the response has tools
        """
        return bool(self.input_object.tools)

    @lazyproperty
    def input_text(self) -> str:
        """
        Returns the input text for the input prompt
        """
        return '\n'.join([f'{msg.role}: {msg.content}' for msg in self.input_object.messages])

    @lazyproperty
    def input_messages(self) -> List[ChatMessage]:
        """
        Returns the input messages for the input prompt
        """
        return self.input_object.messages

    @lazyproperty
    def text(self) -> str:
        """
        Returns the text for the chat response
        """
        if self.choices_results:
            return '\n'.join([f'{msg.role}: {msg.content}' for msg in self.messages])
        return self.response.text


    @lazyproperty
    def only_text(self) -> str:
        """
        Returns the text for the chat response without the role
        """
        if self.has_tools:
            data = []
            for tool_obj in self.tool_result_objects:
                if isinstance(tool_obj, BaseModel):
                    data.append(tool_obj.dict())
                else:
                    data.append(tool_obj)
            return json.dumps(data, indent = 2)
        if self.has_functions:
            data = []
            for func_obj in self.function_result_objects:
                if isinstance(func_obj, BaseModel):
                    data.append(func_obj.dict())
                else:
                    data.append(func_obj)
            return json.dumps(data, indent = 2)
        if self.choices_results:
            return '\n'.join([msg.content for msg in self.messages])
        return self.response.text

    @lazyproperty
    def chat_model(self) -> Optional[str]:
        """
        Returns the model for the completions
        """
        return self.input_object.model or None
        # return OpenAIModel(value=self.model, mode='chat') if self.model else None
    
    @lazyproperty
    def openai_model(self):
        """
        Returns the model for the completions
        """
        return self.headers.get('openai-model', self.chat_model)
    
    def _validate_usage(self):
        """
        Validate usage
        """
        if self.usage and self.usage.total_tokens and self.usage.prompt_tokens: return
        if self.response.status_code == 200:
            self.usage = Usage(
                prompt_tokens = ModelContextHandler.count_chat_tokens(self.input_messages, model_name = self.openai_model),
                completion_tokens = ModelContextHandler.count_chat_tokens(self.messages, model_name = self.openai_model),
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
    

    def __getitem__(self, key: str) -> Any:
        """
        Mimic dict
        """
        return getattr(self, key)
    

    def parse_stream_item(self, item: Union[Dict, Any], **kwargs) -> Optional[StreamedChatMessage]:
        """
        Parses a single stream item
        """
        # logger.info(f'Item: {item}')
        if not item['choices']: return None
        choice = item['choices'][0]
        if choice['finish_reason'] in ['stop', 'function_call']:
            return None
        kind = MessageKind.from_choice(choice)
        return StreamedChatMessage(
            kind = kind,
            value = choice['delta'].get(kind.value, '')
        )
    
    def handle_stream(
        self,
        response: aiohttpx.Response,
        streaming: Optional[bool] = False,
    ) -> Iterator[Dict]:  # sourcery skip: low-code-quality
        """
        Handle the stream response
        """
        results = {}
        for line in parse_stream(response):
            try:
                item = json.loads(line)
                if streaming:
                    yield item
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
                        if 'function_call' in choice['delta']:
                            results[n]['message']['function_call'] = choice['delta']['function_call']
                            self.usage.completion_tokens += 4
                        
                        # every message follows <im_start>{role/name}\n{content}<im_end>\n
                        self.usage.completion_tokens += 4

                    elif choice['finish_reason'] != 'stop':
                        for k,v in choice['delta'].items():
                            if k == 'function_call' and v:
                                for fck, fcv in v.items():
                                    if not results[n]['message'][k].get(fck):
                                        results[n]['message'][k][fck] = fcv
                                    else:
                                        results[n]['message'][k][fck] += fcv

                            elif v: results[n]['message'][k] += v
                            self.usage.completion_tokens += 1

                    else:
                        results[n]['finish_reason'] = choice['finish_reason']
                        self.usage.completion_tokens += 2  # every reply is primed with <im_start>assistant
                        compl = results.pop(n)
                        if streaming:
                            self.handle_resource_item(item = compl)
                        else:
                            yield compl

            except Exception as e:
                logger.error(f'Error: {line}: {e}')
        self._stream_consumed = True
        for remaining_result in results.values():
            if streaming:
                self.handle_resource_item(item = remaining_result)
            else:
                yield remaining_result
        if not self.usage.prompt_tokens:
            self.usage.prompt_tokens = ModelContextHandler.count_chat_tokens(
                messages = self.input_messages,
                model_name = self.openai_model
            )
        self.usage.total_tokens = self.usage.completion_tokens + self.usage.prompt_tokens
        
    
    async def ahandle_stream(
        self,
        response: aiohttpx.Response,
        streaming: Optional[bool] = False,
    ) -> AsyncIterator[Dict]:  # sourcery skip: low-code-quality
        """
        Handles the streaming response
        """
        results = {}
        async for line in aparse_stream(response):
            # logger.info(f'line: {line}')
            try:
                item = json.loads(line)
                self.handle_stream_metadata(item)
                if streaming:
                    yield item

                # logger.info(f'item: {item}')
                for n, choice in enumerate(item['choices']):
                    if not results.get(n):
                        results[n] = {
                            'index': choice['index'],
                            'message': {
                                'role': choice['delta'].get('role', ''),
                                'content': choice['delta'].get('content', ''),
                            }
                        }
                        if 'function_call' in choice['delta']:
                            results[n]['message']['function_call'] = choice['delta']['function_call']
                            self.usage.completion_tokens += 4
                        
                        # every message follows <im_start>{role/name}\n{content}<im_end>\n
                        self.usage.completion_tokens += 4

                    elif choice['finish_reason'] != 'stop':
                        for k,v in choice['delta'].items():
                            if k == 'function_call' and v:
                                for fck, fcv in v.items():
                                    if not results[n]['message'][k].get(fck):
                                        results[n]['message'][k][fck] = fcv
                                    else:
                                        results[n]['message'][k][fck] += fcv
                            elif v: results[n]['message'][k] += v
                            self.usage.completion_tokens += 1

                    else:
                        results[n]['finish_reason'] = choice['finish_reason']
                        self.usage.completion_tokens += 2  # every reply is primed with <im_start>assistant
                        compl = results.pop(n)
                        if streaming:
                            self.handle_resource_item(item = compl)
                        else:
                            yield compl


            except Exception as e:
                logger.trace(f'Error: {line}', e)
        # self.ctx.stream_consumed = True
        self._stream_consumed = True
        for remaining_result in results.values():
            if streaming:
                self.handle_resource_item(item = remaining_result)
            else:
                yield remaining_result
        
        if not self.usage.prompt_tokens:
            self.usage.prompt_tokens = ModelContextHandler.count_chat_tokens(
                messages = self.input_messages,
                model_name = self.openai_model
            )
        self.usage.total_tokens = self.usage.completion_tokens + self.usage.prompt_tokens

    if TYPE_CHECKING:
        def stream(self, **kwargs) -> Generator[StreamedChatMessage, None, None]:
            """
            Streams the response
            """
            ...

        async def astream(self, **kwargs) -> AsyncGenerator[StreamedChatMessage, None]:
            """
            Streams the response
            """
            ...
    


class ChatRoute(BaseRoute):
    input_model: Optional[Type[BaseResource]] = ChatObject
    response_model: Optional[Type[BaseResource]] = ChatResponse

    @lazyproperty
    def api_resource(self):
        return 'chat/completions'

    @lazyproperty
    def root_name(self):
        return 'chat'
    

    # def encode_data(self, data: Dict[str, Any]) -> str:
    #     """
    #     Encodes the data
    #     """
    #     # response_format isn't supported atm
    #     if self.is_azure:
    #         _ = data.pop('response_format', None)
    #     return super().encode_data(data = data)
        
    
    def create(
        self, 
        input_object: Optional[ChatObject] = None,
        parse_stream: Optional[bool] = True,
        timeout: Optional[int] = None,
        auto_retry: Optional[bool] = False,
        auto_retry_limit: Optional[int] = None,
        header_cache_keys: Optional[List[str]] = None,
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

        :functions (optional): A list of dictionaries representing the functions to call
        
        :function_call (optional): The name of the function to call. Default: `auto` if functions are provided

        :response_format (optional): The format of the response. Default: `text`

        :seed (optional): An integer seed for random sampling. Must be between 0 and 2**32 - 1

        :tools (optional): A list of dictionaries representing the tools to use

        :tool_choice (optional): The name of the tool to use. Default: `auto` if tools are provided

        :auto_retry (optional): Whether to automatically retry the request if it fails due to a rate limit error.

        :auto_retry_limit (optional): The maximum number of times to retry the request if it fails due to a rate limit error.

        Returns: `ChatResponse`
        """
        if self.is_azure and self.azure_model_mapping and kwargs.get('model') and kwargs['model'] in self.azure_model_mapping:
            kwargs['model'] = self.azure_model_mapping[kwargs['model']]

        current_attempt = kwargs.pop('_current_attempt', 0)
        if not auto_retry:
            return super().create(input_object=input_object, parse_stream = parse_stream, timeout = timeout, **kwargs)
        
        # Handle Auto Retry Logic
        if not auto_retry_limit: auto_retry_limit = self.settings.max_retries
        try:
            return super().create(input_object = input_object, parse_stream = parse_stream, timeout = timeout, **kwargs)
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
                timeout = timeout,
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
            if header_cache_keys and kwargs.get('headers'):
                headers = kwargs.pop('headers')
                _ = [headers.pop(k) for k in header_cache_keys if k in headers]
                kwargs['headers'] = headers
            return self.create(
                input_object = input_object,
                parse_stream = parse_stream,
                timeout = timeout,
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
            logger.warning(f'[{self.name}: {current_attempt}/{auto_retry_limit}] Unknown Error: ({type(e)}) {e}. Sleeping for 10 seconds')
            time.sleep(10.0)
            current_attempt += 1
            if header_cache_keys and kwargs.get('headers'):
                headers = kwargs.pop('headers')
                _ = [headers.pop(k) for k in header_cache_keys if k in headers]
                kwargs['headers'] = headers
            return self.create(
                input_object = input_object,
                parse_stream = parse_stream,
                timeout = timeout, 
                auto_retry = auto_retry,
                auto_retry_limit = auto_retry_limit,
                _current_attempt = current_attempt,
                **kwargs
            )



    async def async_create(
        self, 
        input_object: Optional[ChatObject] = None,
        parse_stream: Optional[bool] = True,
        timeout: Optional[int] = None,
        auto_retry: Optional[bool] = False,
        auto_retry_limit: Optional[int] = None,
        header_cache_keys: Optional[List[str]] = None,
        **kwargs
    ) -> ChatResponse:  # sourcery skip: low-code-quality
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

        :functions (optional): A list of dictionaries representing the functions to call
        
        :function_call (optional): The name of the function to call. Default: `auto` if functions are provided

        :response_format (optional): The format of the response. Default: `text`

        :seed (optional): An integer seed for random sampling. Must be between 0 and 2**32 - 1

        :tools (optional): A list of dictionaries representing the tools to use

        :tool_choice (optional): The name of the tool to use. Default: `auto` if tools are provided

        :auto_retry (optional): Whether to automatically retry the request if it fails due to a rate limit error.

        :auto_retry_limit (optional): The maximum number of times to retry the request if it fails due to a rate limit error.

        Default: `None`

        Returns: `ChatResponse`
        """
        if self.is_azure and self.azure_model_mapping and kwargs.get('model') and kwargs['model'] in self.azure_model_mapping:
            kwargs['model'] = self.azure_model_mapping[kwargs['model']]
        current_attempt = kwargs.pop('_current_attempt', 0)
        if not auto_retry:
            return await super().async_create(input_object = input_object, parse_stream = parse_stream, timeout = timeout, **kwargs)

        # Handle Auto Retry Logic
        if not auto_retry_limit: auto_retry_limit = self.settings.max_retries
        try:
            return await super().async_create(input_object = input_object, parse_stream = parse_stream, timeout = timeout, **kwargs)
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
                timeout = timeout, 
                auto_retry = auto_retry,
                auto_retry_limit = auto_retry_limit,
                _current_attempt = current_attempt,
                **kwargs
            )
        except ServiceTimeoutError as e:
            if current_attempt > 1:
                logger.warning(f'[{self.name}: {current_attempt}/{auto_retry_limit}] Service Timeout Error. Not retrying as issue is likely not transient.')
                raise e
            logger.warning(f'[{self.name}: {current_attempt}/{auto_retry_limit}] Service Timeout Error. Sleeping for 2 seconds and setting timeout to 5 seconds')
            await asyncio.sleep(2.0)
            current_attempt += 1
            timeout = 5.0
            return await self.async_create(
                input_object = input_object,
                parse_stream = parse_stream,
                timeout = timeout, 
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
            if header_cache_keys and kwargs.get('headers'):
                headers = kwargs.pop('headers')
                _ = [headers.pop(k) for k in header_cache_keys if k in headers]
                kwargs['headers'] = headers
            return await self.async_create(
                input_object = input_object,
                parse_stream = parse_stream,
                timeout = timeout, 
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
            logger.warning(f'[{self.name}: {current_attempt}/{auto_retry_limit}] Unknown Error: ({type(e)}) {e}. Sleeping for 10 seconds')
            await asyncio.sleep(10.0)
            current_attempt += 1
            if header_cache_keys and kwargs.get('headers'):
                headers = kwargs.pop('headers')
                _ = [headers.pop(k) for k in header_cache_keys if k in headers]
                kwargs['headers'] = headers
            return await self.async_create(
                input_object = input_object,
                parse_stream = parse_stream,
                timeout = timeout, 
                auto_retry = auto_retry,
                auto_retry_limit = auto_retry_limit,
                _current_attempt = current_attempt,
                **kwargs
            )


    
