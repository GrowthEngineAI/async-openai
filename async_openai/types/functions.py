from __future__ import annotations

"""
OpenAI Functions Base Class
"""

import jinja2
import functools
import inspect
from abc import ABC
from pydantic import Field, BaseModel
# from lazyops.types import BaseModel
from lazyops.utils.times import Timer
from lazyops.libs.proxyobj import ProxyObject
from lazyops.types.models import schema_extra, PYD_VERSION
from async_openai.utils.fixjson import resolve_json
from . import errors

from typing import Optional, Any, Set, Dict, List, Union, Type, Tuple, Awaitable, Generator, AsyncGenerator, TypeVar, TYPE_CHECKING

if PYD_VERSION == 2:
    from pydantic import ConfigDict

if TYPE_CHECKING:
    from async_openai import ChatResponse, ChatRoute
    from async_openai.types.resources import Usage
    from async_openai.manager import OpenAIManager as OpenAISessionManager
    from lazyops.utils.logs import Logger
    from lazyops.libs.persistence import PersistentDict
    

FT = TypeVar('FT', bound = BaseModel)
SchemaT = TypeVar('SchemaT', bound = BaseModel)


class BaseFunctionModel(BaseModel):
    function_name: Optional[str] = Field(None, hidden = True)
    function_model: Optional[str] = Field(None, hidden = True)
    function_duration: Optional[float] = Field(None, hidden = True)

    if TYPE_CHECKING:
        function_usage: Optional[Usage]
    else:
        function_usage: Optional[Any] = Field(None, hidden = True)

    def update(
        self,
        values: 'BaseFunctionModel',
    ):
        """
        Updates the values
        """
        pass

    def _setup_item(
        self,
        item: 'SchemaT',
        **kwargs
    ) -> 'SchemaT':
        """
        Updates the Reference Item
        """
        return item
    

    def update_values(
        self,
        item: 'SchemaT',
        **kwargs
    ) -> 'SchemaT':
        """
        Updates the Reference Item with the values
        """
        return item
    

    def update_data(
        self,
        item: 'SchemaT',
        **kwargs
    ) -> 'SchemaT':
        """
        Updates the Reference Item with the values
        """
        item = self._setup_item(item = item, **kwargs)
        item = self.update_values(item = item, **kwargs)
        return item
    

    def is_valid(self) -> bool:
        """
        Returns whether the function data is valid
        """
        return True
    
    def _set_values_from_response(
        self,
        response: 'ChatResponse',
        name: Optional[str] = None,
        **kwargs
    ) -> 'BaseFunctionModel':
        """
        Sets the values from the response
        """
        if name: self.function_name = name
        self.function_usage = response.usage
        if response.response_ms: self.function_duration = response.response_ms / 1000
        self.function_model = response.model

    @property
    def function_cost(self) -> Optional[float]:
        """
        Returns the function consumption
        """
        if not self.function_model: return None
        if not self.function_usage: return None
        from async_openai.types.context import ModelContextHandler
        return ModelContextHandler.get_consumption_cost(self.function_model, self.function_usage)
    
    @property
    def function_cost_string(self) -> Optional[str]:
        """
        Returns the function consumption as a pretty string
        """
        return f"${self.function_cost:.2f}" if self.function_cost else None


    if PYD_VERSION == 2:
        model_config = ConfigDict(json_schema_extra = schema_extra, arbitrary_types_allowed = True)
    else:
        class Config:
            json_schema_extra = schema_extra
            arbitrary_types_allowed = True



FunctionSchemaT = TypeVar('FunctionSchemaT', bound = BaseFunctionModel)
FunctionResultT = TypeVar('FunctionResultT', bound = BaseFunctionModel)

class BaseFunction(ABC):
    """
    Base Class for OpenAI Functions
    """
    
    name: Optional[str] = None
    function_name: Optional[str] = None
    description: Optional[str] = None
    schema: Optional[Type[FunctionSchemaT]] = None
    schemas: Optional[Dict[str, Dict[str, Union[str, Type[FunctionSchemaT]]]]] = None
    
    prompt_template: Optional[str] = None
    system_template: Optional[str] = None

    default_model: Optional[str] = 'gpt-35-turbo'
    default_larger_model: Optional[bool] = None
    cachable: Optional[bool] = True
    result_buffer: Optional[int] = 1000
    retry_limit: Optional[int] = 5
    max_attempts: Optional[int] = 2

    default_model_local: Optional[str] = None
    default_model_develop: Optional[str] = None
    default_model_production: Optional[str] = None

    auto_register_function: Optional[bool] = True

    def __init_subclass__(cls, **kwargs):
        """
        Subclass Hook
        """
        if cls.auto_register_function:
            OpenAIFunctions.register_function(cls, initialize = False)


    def __init__(
        self, 
        api: Optional['OpenAISessionManager'] = None,
        debug_enabled: Optional[bool] = None,
        **kwargs
    ):
        """
        This gets initialized from the Enrichment Handler
        """
        from async_openai.manager import ModelContextHandler
        from async_openai.utils.logs import logger, null_logger
        self.ctx: Type['ModelContextHandler'] = ModelContextHandler
        if api is None:
            from async_openai.client import OpenAIManager
            api = OpenAIManager
        
        self.api: 'OpenAISessionManager' = api
        self.pool = self.api.pooler
        self.kwargs = kwargs
        self.logger = logger
        self.null_logger = null_logger
        self.settings = self.api.settings
        if debug_enabled is not None:
            self.debug_enabled = debug_enabled
        else:
            self.debug_enabled = self.settings.debug_enabled
        self.build_funcs(**kwargs)
        self.build_templates(**kwargs)
        self.post_init(**kwargs)

    @property
    def default_model_func(self) -> str:
        """
        Returns the default model
        """
        if self.settings.is_local_env:
            return self.default_model_local or self.default_model
        if self.settings.is_development_env:
            return self.default_model_develop or self.default_model
        return self.default_model_production or self.default_model

    @property
    def autologger(self) -> 'Logger':
        """
        Returns the logger
        """
        return self.logger if \
            (self.debug_enabled or self.settings.is_development_env) else self.null_logger


    @property
    def has_diff_model_than_default(self) -> bool:
        """
        Returns True if the default model is different than the default model
        """
        return self.default_model_func != self.default_model


    def build_templates(self, **kwargs):
        """
        Construct the templates
        """
        self.template = self.create_template(self.prompt_template)
        # Only create the system template if it's a jinja template
        if self.system_template and '{%' in self.system_template:
            self.system_template = self.create_template(self.system_template)

    def build_funcs(self, **kwargs):
        """
        Builds the functions
        """
        # Handles multi functions
        if self.schemas:
            self.functions = []
            self.functions.extend(
                {
                    "name": name,
                    "description": data.get('description', self.description),
                    "parameters": data.get('schema', self.schema),
                }
                for name, data in self.schemas.items()
            )
        else:
            self.functions = [
                {
                    "name": self.function_name or self.name,
                    "description": self.description,
                    "parameters": self.schema,
                }
            ]


    def post_init(self, **kwargs):
        """
        Post Init Hook
        """
        pass
    
    def pre_call_hook(self, *args, **kwargs):
        """
        Pre Call Hook
        """
        pass

    async def apre_call_hook(self, *args, **kwargs):
        """
        Pre Call Hook
        """
        self.pre_call_hook(*args, **kwargs)

    def pre_validate(self, *args, **kwargs) -> bool:
        """
        Validate the input before running
        """
        return True

    async def apre_validate(self, *args, **kwargs) -> bool:
        """
        Validate the input before running
        """
        return self.pre_validate(*args, **kwargs)
    

    def pre_validate_model(self, prompt: str, model: str, *args, **kwargs) -> str:
        """
        Validates the model before running
        """
        return model

    async def apre_validate_model(self, prompt: str, model: str, *args, **kwargs) -> str:
        """
        Validates the model before running
        """
        return self.pre_validate_model(prompt = prompt, model = model, *args, **kwargs)
    


    def call(
        self,
        *args,
        model: Optional[str] = None, 
        **kwargs
    ) -> Optional[FunctionSchemaT]:
        """
        Call the function
        """
        if not self.pre_validate(*args, **kwargs):
            return None
        self.pre_call_hook(*args, **kwargs)
        return self.run_function(*args, model = model, **kwargs)
    
    async def acall(
        self,
        *args,
        model: Optional[str] = None, 
        **kwargs
    ) -> Optional[FunctionSchemaT]:
        """
        Call the function
        """
        if not await self.apre_validate(*args, **kwargs):
            return None
        await self.apre_call_hook(*args, **kwargs)
        return await self.arun_function(*args, model = model, **kwargs)

    def __call__(
        self,
        *args,
        model: Optional[str] = None, 
        is_async: Optional[bool] = True,
        **kwargs
    ) -> Optional[FunctionSchemaT]:
        """
        Call the function
        """
        if is_async: return self.acall(*args, model = model, **kwargs)
        return self.call(*args, model = model, **kwargs)

    def get_chat_client(self, model: str, **kwargs) -> 'ChatRoute':
        """
        Gets the chat client
        """
        return self.api.get_chat_client(model = model, **kwargs)
    
    def get_completion_client(self, model: str, **kwargs) -> 'ChatRoute':
        """
        Gets the chat client
        """
        return self.api.get_chat_client(model = model, **kwargs)
    
    async def arun_chat_function(
        self, 
        messages: List[Dict[str, Any]],
        chat: Optional['ChatRoute'] = None, 
        cachable: Optional[bool] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_name: Optional[str] = None,
        property_meta: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        excluded_clients: Optional[List[str]] = None,
        **kwargs,
    ) -> ChatResponse:  # sourcery skip: low-code-quality
        """
        Runs the chat function
        """
        current_attempt = kwargs.pop('_current_attempt', 0)
        last_chat_name = kwargs.pop('_last_chat_name', None)
        if current_attempt and current_attempt > self.retry_limit:
            raise errors.MaxRetriesExhausted(
                name = last_chat_name,
                func_name = function_name or self.name,
                model = model,
                attempts = current_attempt,
                max_attempts = self.retry_limit,
            )

        disable_cache = not cachable if cachable is not None else not self.cachable
        if not chat: 
            if last_chat_name:
                if not excluded_clients: excluded_clients = []
                excluded_clients.append(last_chat_name)
            chat = self.get_chat_client(model = model, excluded_clients = excluded_clients, **kwargs)
        if not headers and 'noproxy' not in chat.name:
            headers = {
                'Helicone-Cache-Enabled': 'false' if disable_cache else 'true',
                'Helicone-Property-FunctionName': function_name or self.name,
            }
            if property_meta:
                property_meta = {f'Helicone-Property-{k}': str(v) for k, v in property_meta.items()}
                headers.update(property_meta)
            
        elif headers and 'noproxy' in chat.name:
            headers = None
        functions = functions or self.functions
        try:
            if headers: chat.client.headers.update(headers)
            return await chat.async_create(
                model = model,
                messages = messages,
                functions = functions,
                headers = headers,
                auto_retry = True,
                auto_retry_limit = 2,
                function_call = {'name': function_name or self.name},
                header_cache_keys = ['Helicone-Cache-Enabled'],
                **kwargs,
            )
        except errors.InvalidRequestError as e:
            self.logger.info(f"[{current_attempt}/{self.retry_limit}] [{self.name} - {model}] Invalid Request Error. |r|{e}|e|", colored=True)
            raise e
        except errors.MaxRetriesExceeded as e:
            self.autologger.info(f"[{current_attempt}/{self.retry_limit}] [{self.name} - {model}] Retrying...", colored=True)
            return await self.arun_chat_function(
                messages = messages,
                cachable = cachable,
                functions = functions,
                function_name = function_name,
                property_meta = property_meta,
                model = model,
                headers = headers,
                excluded_clients = excluded_clients,
                _current_attempt = current_attempt + 1,
                _last_chat_name = chat.name,
                **kwargs,
            )
        except Exception as e:
            self.autologger.info(f"[{current_attempt}/{self.retry_limit}] [{self.name} - {model}] Unknown Error Trying to run chat function: |r|{e}|e|", colored=True)
            return await self.arun_chat_function(
                messages = messages,
                cachable = cachable,
                functions = functions,
                function_name = function_name,
                property_meta = property_meta,
                model = model,
                headers = headers,
                excluded_clients = excluded_clients,
                _current_attempt = current_attempt + 1,
                _last_chat_name = chat.name,
                **kwargs,
            )

    def run_chat_function(
        self, 
        chat: 'ChatRoute', 
        messages: List[Dict[str, Any]],
        cachable: Optional[bool] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_name: Optional[str] = None,
        property_meta: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ChatResponse:
        """
        Runs the chat function
        """
        disable_cache = not cachable if cachable is not None else not self.cachable
        headers = None
        if 'noproxy' not in chat.name:
            headers = {
                'Helicone-Cache-Enabled': 'false' if disable_cache else 'true',
                'Helicone-Property-FunctionName': function_name or self.name,
            }
            if property_meta:
                property_meta = {f'Helicone-Property-{k.strip()}': str(v).strip() for k, v in property_meta.items()}
                headers.update(property_meta)
        if headers: chat.client.headers.update(headers)
        functions = functions or self.functions
        return chat.create(
            messages = messages,
            functions = functions,
            headers = headers,
            auto_retry = True,
            auto_retry_limit = self.retry_limit,
            function_call = {'name': function_name or self.name},
            header_cache_keys=['Helicone-Cache-Enabled'],
            **kwargs,
        )

    def parse_response(
        self,
        response: 'ChatResponse',
        schema: Optional[Type[FunctionSchemaT]] = None,
        include_name: Optional[bool] = True,
    ) -> Optional[FunctionSchemaT]:  # sourcery skip: extract-duplicate-method
        """
        Parses the response
        """
        schema = schema or self.schema
        try:
            result = schema.model_validate(response.function_results[0].arguments, from_attributes = True)
            result._set_values_from_response(response, name = self.name if include_name else None)
            return result
        except Exception as e:
            self.autologger.error(f"[{self.name} - {response.model} - {response.usage}] Failed to parse object: {e}\n{response.text}\n{response.function_results[0].arguments}")
            try:
                result = schema.model_validate(resolve_json(response.function_results[0].arguments), from_attributes = True)
                result._set_values_from_response(response, name = self.name if include_name else None)
                return result
            except Exception as e:
                self.autologger.error(f"[{self.name} - {response.model} - {response.usage}] Failed to parse object after fixing")
                return None
    

    def is_valid_response(self, response: FT) -> bool:
        """
        Returns True if the response is valid
        """
        return True
    
    def apply_text_cleaning(self, text: str) -> str:
        """
        Applies text cleaning
        """
        from lazyops.utils.format_utils import clean_html, clean_text, cleanup_dots
        if "..." in text: text = cleanup_dots(text)
        return clean_html(clean_text(text))
    
    @staticmethod
    def create_template(template: str, enable_async: Optional[bool] = False, **kwargs) -> jinja2.Template:
        """
        Creates the template
        """
        return jinja2.Template(template, enable_async = enable_async, **kwargs)
    
    def truncate_documents(
        self, 
        documents: Dict[str, str],
        max_length: Optional[int] = None,
        buffer_size: Optional[int] = None,
        model: Optional[str] = None,
        truncation_length: Optional[int] = None,
    ) -> Dict[str, str]:
        """
        Helper Function to truncate supporting docs
        """
        current_length = 0
        if max_length is None:
            model = model or self.default_model_func
            max_length = self.ctx.get(model).context_length
        if buffer_size is None: buffer_size = self.result_buffer
        max_length -= buffer_size

        truncation_length = truncation_length or (max_length // len(documents))
        new_documents = {}
        for file_name, file_text in documents.items():
            if not file_text: continue
            file_text = self.apply_text_cleaning(file_text)[:truncation_length]
            current_length += len(file_text)
            new_documents[file_name] = file_text
            if current_length > max_length: break
        return new_documents
    
    """
    Function Handlers
    """

    def prepare_function_inputs(
        self,
        model: Optional[str] = None,
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Prepare the Function Inputs for the function
        """
        model = model or self.default_model_func
        prompt = self.template.render(**kwargs)
        prompt = self.api.truncate_to_max_length(prompt, model = model, buffer_length = self.result_buffer)
        messages = []
        if self.system_template:
            if isinstance(self.system_template, jinja2.Template):
                system_template = self.system_template.render(**kwargs)
            else:
                system_template = self.system_template
            messages.append({
                "role": "system",
                "content": system_template,
            })
        messages.append({
            "role": "user", 
            "content": prompt,
        })
        return messages, model
    
    async def aprepare_function_inputs(
        self,
        model: Optional[str] = None,
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Prepare the Function Inputs for the function
        """
        model = model or self.default_model_func
        prompt = self.template.render(**kwargs)
        prompt = await self.api.atruncate_to_max_length(prompt, model = model, buffer_length = self.result_buffer)
        messages = []
        if self.system_template:
            if isinstance(self.system_template, jinja2.Template):
                system_template = self.system_template.render(**kwargs)
            else:
                system_template = self.system_template
            messages.append({
                "role": "system",
                "content": system_template,
            })
        messages.append({
            "role": "user", 
            "content": prompt,
        })
        return messages, model

    def run_function(
        self,
        *args,
        model: Optional[str] = None, 
        **kwargs
    ) -> Optional[FunctionSchemaT]:
        """
        Returns the Function Result
        """
        messages, model = self.prepare_function_inputs(model = model, **kwargs)
        return self.run_function_loop(messages = messages, model = model, **kwargs)

    async def arun_function(
        self,
        *args,
        model: Optional[str] = None, 
        **kwargs
    ) -> Optional[FunctionSchemaT]:
        """
        Returns the Function Result
        """
        messages, model = await self.aprepare_function_inputs(model = model, **kwargs)
        return await self.arun_function_loop(messages = messages, model = model, **kwargs)
    
    def get_function_kwargs(self) -> Dict[str, Any]:
        """
        Returns the function kwargs
        """
        sig = inspect.signature(self.arun_function)
        return [p.name for p in sig.parameters.values() if p.kind in {p.KEYWORD_ONLY, p.VAR_KEYWORD, p.POSITIONAL_OR_KEYWORD} and p.name not in {'kwargs', 'args', 'model'}]

    """
    Handle a Loop
    """

    def run_function_loop(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        raise_errors: Optional[bool] = True,
        **kwargs,
    ) -> Optional[FunctionSchemaT]:
        """
        Runs the function loop
        """
        chat = self.get_chat_client(model = model, **kwargs)
        response = self.run_chat_function(
            chat = chat,
            messages = messages,
            model = model,
            **kwargs,
        )

        result = self.parse_response(response, include_name = True)
        if result is not None: return result

        # Try Again
        attempts = 1
        _ = kwargs.pop('cachable', None)
        while attempts < self.max_attempts:
            chat = self.get_chat_client(model = model, **kwargs)
            response = self.run_chat_function(
                chat = chat,
                messages = messages,
                model = model,
                **kwargs,
            )
            result = self.parse_response(response, include_name = True)
            if result is not None: return result
            attempts += 1
        self.autologger.error(f"Unable to parse the response for {self.name} after {self.max_attempts} attempts.")
        if raise_errors: raise errors.MaxRetriesExhausted(
            name = self.name, 
            func_name = self.name,
            model = model,
            attempts = self.max_attempts,
        )
        return None

    async def arun_function_loop(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        raise_errors: Optional[bool] = True,
        **kwargs,
    ) -> Optional[FunctionSchemaT]:
        """
        Runs the function loop
        """
        chat = self.get_chat_client(model = model, **kwargs)
        response = await self.arun_chat_function(
            chat = chat,
            messages = messages,
            model = model,
            **kwargs,
        )

        result = self.parse_response(response, include_name = True)
        if result is not None: return result

        # Try Again
        attempts = 1
        _ = kwargs.pop('cachable', None)
        while attempts < self.max_attempts:
            chat = self.get_chat_client(model = model, **kwargs)
            response = await self.arun_chat_function(
                chat = chat,
                messages = messages,
                model = model,
                **kwargs,
            )
            result = self.parse_response(response, include_name = True)
            if result is not None: return result
            attempts += 1
        self.autologger.error(f"Unable to parse the response for {self.name} after {self.max_attempts} attempts.")
        if raise_errors: raise errors.MaxRetriesExhausted(
            name = self.name, 
            func_name = self.name,
            model = model,
            attempts = self.max_attempts,
        )
        return None

    

FunctionT = TypeVar('FunctionT', bound = BaseFunction)



class FunctionManager(ABC):
    """
    The Functions Manager Class that handles registering and managing functions

    - Additionally supports caching through `kvdb`
    """

    name: Optional[str] = 'functions'

    def __init__(
        self,
        **kwargs,
    ):
        from async_openai.utils.config import settings
        from async_openai.utils.logs import logger, null_logger
        self.logger = logger
        self.null_logger = null_logger
        self.settings = settings
        self.debug_enabled = self.settings.debug_enabled
        self.cache_enabled = self.settings.function_cache_enabled

        self._api: Optional['OpenAISessionManager'] = None
        self._cache: Optional['PersistentDict'] = None
        self.functions: Dict[str, BaseFunction] = {}
        self._kwargs = kwargs
        try:
            import xxhash
            self._hash_func = xxhash.xxh64
        except ImportError:
            from hashlib import md5
            self._hash_func = md5
        
        try:
            import cloudpickle
            self._pickle = cloudpickle
        except ImportError:
            import pickle
            self._pickle = pickle

    @property
    def api(self) -> 'OpenAISessionManager':
        """
        Returns the API
        """
        if self._api is None:
            from async_openai.client import OpenAIManager
            self._api = OpenAIManager
        return self._api
    
    @property
    def autologger(self) -> 'Logger':
        """
        Returns the logger
        """
        return self.logger if \
            (self.debug_enabled or self.settings.is_development_env) else self.null_logger

    @property
    def cache(self) -> 'PersistentDict':
        """
        Gets the cache
        """
        if self._cache is None:
            serializer_kwargs = {
                'compression': self._kwargs.get('serialization_compression', None),
                'compression_level': self._kwargs.get('serialization_compression_level', None),
                'raise_errors': True,
            }
            kwargs = {
                'base_key': f'openai.functions.{self.api.settings.app_env.name}.{self.api.settings.proxy.proxy_app_name or "default"}',
                'expiration': self._kwargs.get('cache_expiration', 60 * 60 * 24 * 3),
                'serializer': self._kwargs.get('serialization', 'json'),
                'serializer_kwargs': serializer_kwargs,
            }
            try:
                import kvdb
                self._cache = kvdb.create_persistence(session_name = 'openai', **kwargs)
            except ImportError:
                from lazyops.libs.persistence import PersistentDict
                self._cache = PersistentDict(**kwargs)
        return self._cache

    def register_function(
        self,
        func: Union[BaseFunction, Type[BaseFunction], str],
        name: Optional[str] = None,
        overwrite: Optional[bool] = False,
        raise_error: Optional[bool] = False,
        initialize: Optional[bool] = True,
        **kwargs,
    ):
        """
        Registers the function
        """
        if isinstance(func, str):
            from lazyops.utils.lazy import lazy_import
            func = lazy_import(func)
        if isinstance(func, type) and initialize:
            func = func(**kwargs)
        name = name or func.name
        if not overwrite and name in self.functions:
            if raise_error: raise ValueError(f"Function {name} already exists")
            return
        self.functions[name] = func
        self.autologger.info(f"Registered Function: |g|{name}|e|", colored=True)

    def create_hash(self, *args, **kwargs) -> str:
        """
        Creates a hash
        """
        key = args or ()
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        sorted_items = sorted(kwargs.items())
        for item in sorted_items:
            key += item
        key = ':'.join(str(k) for k in key)
        return self._hash_func(key).hexdigest()
    
    async def acreate_hash(self, *args, **kwargs) -> str:
        """
        Creates a hash
        """
        return await self.api.pooler.asyncish(self.create_hash, *args, **kwargs)
    
    def _get_function(self, name: str) -> Optional[BaseFunction]:
        """
        Gets the function
        """
        func = self.functions.get(name)
        if not func: return None
        if isinstance(func, type):
            func = func(**self._kwargs)
            self.functions[name] = func
        return func

    def get(self, name: Union[str, 'FunctionT']) -> Optional['FunctionT']:
        """
        Gets the function
        """
        return name if isinstance(name, BaseFunction) else self._get_function(name)
        
    
    def parse_iterator_func(
        self,
        function: 'FunctionT',
        *args,
        with_index: Optional[bool] = False,
        **function_kwargs,
    ) -> Tuple[int, Set, Dict[str, Any]]:
        """
        Parses the iterator function kwargs
        """
        func_iter_arg = args[0]
        args = args[1:]
        idx = None
        if with_index: idx, item = func_iter_arg
        else: item = func_iter_arg
        _func_kwargs = function.get_function_kwargs()
        if isinstance(item, dict) and any(k in _func_kwargs for k in item):
            function_kwargs.update(item)
        else:
            # Get the missing function kwargs
            _added = False
            for k in _func_kwargs:
                if k not in function_kwargs:
                    function_kwargs[k] = item
                    # self.autologger.info(f"Added missing function kwarg: {k} = {item}", prefix = function.name, colored = True)
                    _added = True
                    break
            if not _added:
                # If not, then add the item as the first argument
                args = (item,) + args
        return idx, args, function_kwargs

    def execute(
        self,
        function: Union['FunctionT', str],
        *args,
        item_hashkey: Optional[str] = None,
        cachable: Optional[bool] = True,
        overrides: Optional[List[str]] = None,
        with_index: Optional[bool] = False,
        **function_kwargs
    ) -> Union[Optional['FunctionSchemaT'], Tuple[int, Optional['FunctionSchemaT']]]:
        """
        Runs the function
        """
        overwrite = overrides and 'functions' in overrides
        function = self.get(function)
        if overwrite and self.check_value_present(overrides, f'{function.name}.cachable'):
            cachable = False
        
        # Iterators
        is_iterator = function_kwargs.pop('_is_iterator', False)
        if is_iterator:
            idx, args, function_kwargs = self.parse_iterator_func(function, *args, with_index = with_index, **function_kwargs)
        
        if item_hashkey is None: item_hashkey = self.create_hash(**function_kwargs)
        key = f'{item_hashkey}.{function.name}'
        if function.has_diff_model_than_default:
            key += f'.{function.default_model_func}'

        t = Timer()
        result = None
        cache_hit = False
        if self.cache_enabled and not overwrite:
            result: 'FunctionResultT' = self.cache.fetch(key)
            if result:
                if isinstance(result, dict): result = function.schema.model_validate(result)
                result.function_name = function.name
                cache_hit = True
        
        if result is None:
            result = function(*args, cachable = cachable, is_async = False, **function_kwargs)
            if self.cache_enabled and function.is_valid_response(result):
                self.cache.set(key, result)
        
        self.autologger.info(f"Function: {function.name} in {t.total_s} (Cache Hit: {cache_hit})", prefix = key, colored = True)
        if is_iterator and with_index:
            return idx, result if function.is_valid_response(result) else (idx, None)
        return result if function.is_valid_response(result) else None

    
    async def aexecute(
        self,
        function: Union['FunctionT', str],
        *args,
        item_hashkey: Optional[str] = None,
        cachable: Optional[bool] = True,
        overrides: Optional[List[str]] = None,
        with_index: Optional[bool] = False,
        **function_kwargs
    ) -> Union[Optional['FunctionSchemaT'], Tuple[int, Optional['FunctionSchemaT']]]:
        # sourcery skip: low-code-quality
        """
        Runs the function
        """
        overwrite = overrides and 'functions' in overrides
        function = self.get(function)
        if overwrite and self.check_value_present(overrides, f'{function.name}.cachable'):
            cachable = False
        
        # Iterators
        is_iterator = function_kwargs.pop('_is_iterator', False)
        if is_iterator:
            idx, args, function_kwargs = self.parse_iterator_func(function, *args, with_index = with_index, **function_kwargs)

        if item_hashkey is None: item_hashkey = await self.acreate_hash(*args, **function_kwargs)
        key = f'{item_hashkey}.{function.name}'
        if function.has_diff_model_than_default:
            key += f'.{function.default_model_func}'

        t = Timer()
        result = None
        cache_hit = False
        if self.cache_enabled and not overwrite:
            result: 'FunctionResultT' = await self.cache.afetch(key)
            if result:
                if isinstance(result, dict): result = function.schema.model_validate(result)
                result.function_name = function.name
                cache_hit = True
        
        if result is None:
            result = await function(*args, cachable = cachable, is_async = True, **function_kwargs)
            if self.cache_enabled and function.is_valid_response(result):
                await self.cache.aset(key, result)
        
        self.autologger.info(f"Function: {function.name} in {t.total_s} (Cache Hit: {cache_hit})", prefix = key, colored = True)
        if is_iterator and with_index:
            return idx, result if function.is_valid_response(result) else (idx, None)
        return result if function.is_valid_response(result) else None

    
    
    @property
    def function_names(self) -> List[str]:
        """
        Returns the function names
        """
        return list(self.functions.keys())
    
    def __getitem__(self, name: str) -> Optional['FunctionT']:
        """
        Gets the function
        """
        return self.get(name)

    def __setitem__(self, name: str, value: Union[FunctionT, Type[FunctionT], str]):
        """
        Sets the function
        """
        return self.register_function(value, name = name)
    
    def append(self, value: Union[FunctionT, Type[FunctionT], str]):
        """
        Appends the function
        """
        return self.register_function(value)
    

    def check_value_present(
        self, items: List[str], *values: str,
    ) -> bool:
        """
        Checks if the value is present
        """
        if not values:
            return any(self.name in item for item in items)
        for value in values:
            key = f'{self.name}.{value}' if value else self.name
            if any((key in item or value in item) for item in items):
                return True
        return False
    
    def map(
        self,
        function: Union['FunctionT', str],
        iterable_kwargs: List[Union[Dict[str, Any], Any]],
        *args,
        cachable: Optional[bool] = True,
        overrides: Optional[List[str]] = None,
        return_ordered: Optional[bool] = True,
        with_index: Optional[bool] = False,
        **function_kwargs
    ) -> List[Union[Optional['FunctionSchemaT'], Tuple[int, Optional['FunctionSchemaT']]]]:
        """
        Maps the function to the iterable in parallel
        """
        partial = functools.partial(
            self.execute, 
            function, 
            # *args,
            cachable = cachable, 
            overrides = overrides, 
            _is_iterator = True,
            with_index = with_index,
            **function_kwargs
        )
        if with_index: iterable_kwargs = list(enumerate(iterable_kwargs))
        return self.api.pooler.map(partial, iterable_kwargs, *args, return_ordered = return_ordered)
    
    async def amap(
        self,
        function: Union['FunctionT', str],
        iterable_kwargs: List[Dict[str, Any]],
        *args,
        cachable: Optional[bool] = True,
        overrides: Optional[List[str]] = None,
        return_ordered: Optional[bool] = True,
        concurrency_limit: Optional[int] = None,
        with_index: Optional[bool] = False,
        **function_kwargs
    ) -> List[Union[Optional['FunctionSchemaT'], Tuple[int, Optional['FunctionSchemaT']]]]:
        """
        Maps the function to the iterable in parallel
        """
        partial = functools.partial(
            self.aexecute, 
            function, 
            # *args,
            cachable = cachable, 
            overrides = overrides, 
            _is_iterator = True,
            with_index = with_index,
            **function_kwargs
        )
        if with_index: iterable_kwargs = list(enumerate(iterable_kwargs))
        return await self.api.pooler.amap(partial, iterable_kwargs, *args, return_ordered = return_ordered, concurrency_limit = concurrency_limit)
    
    def iterate(
        self,
        function: Union['FunctionT', str],
        iterable_kwargs: List[Dict[str, Any]],
        *args,
        cachable: Optional[bool] = True,
        overrides: Optional[List[str]] = None,
        return_ordered: Optional[bool] = False,
        with_index: Optional[bool] = False,
        **function_kwargs
    ) -> Generator[Union[Optional['FunctionSchemaT'], Tuple[int, Optional['FunctionSchemaT']]], None, None]:
        """
        Maps the function to the iterable in parallel
        """
        partial = functools.partial(
            self.execute, 
            function, 
            # *args, 
            cachable = cachable, 
            overrides = overrides, 
            _is_iterator = True,
            with_index = with_index,
            **function_kwargs
        )
        if with_index: iterable_kwargs = list(enumerate(iterable_kwargs))
        return self.api.pooler.iterate(partial, iterable_kwargs, *args, return_ordered = return_ordered)

    def aiterate(
        self,
        function: Union['FunctionT', str],
        iterable_kwargs: List[Dict[str, Any]],
        *args,
        cachable: Optional[bool] = True,
        overrides: Optional[List[str]] = None,
        return_ordered: Optional[bool] = False,
        concurrency_limit: Optional[int] = None,
        with_index: Optional[bool] = False,
        **function_kwargs
    ) -> AsyncGenerator[Union[Optional['FunctionSchemaT'], Tuple[int, Optional['FunctionSchemaT']]], None]:
        """
        Maps the function to the iterable in parallel
        """
        partial = functools.partial(
            self.aexecute, 
            function, 
            # *args, 
            cachable = cachable, 
            overrides = overrides, 
            _is_iterator = True,
            with_index = with_index,
            **function_kwargs
        )
        if with_index: iterable_kwargs = list(enumerate(iterable_kwargs))
        return self.api.pooler.aiterate(partial, iterable_kwargs, *args, return_ordered = return_ordered, concurrency_limit = concurrency_limit)
    
    def __call__(
        self,
        function: Union['FunctionT', str],
        *args,
        item_hashkey: Optional[str] = None,
        cachable: Optional[bool] = True,
        overrides: Optional[List[str]] = None,
        is_async: Optional[bool] = True,
        **function_kwargs
    ) -> Union[Awaitable['FunctionSchemaT'], 'FunctionSchemaT']:
        """
        Runs the function
        """
        method = self.aexecute if is_async else self.execute
        return method(
            function = function,
            *args,
            item_hashkey = item_hashkey,
            cachable = cachable,
            overrides = overrides,
            **function_kwargs
        )
    




OpenAIFunctions: FunctionManager = ProxyObject(FunctionManager)