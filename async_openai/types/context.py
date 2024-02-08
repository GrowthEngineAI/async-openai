from __future__ import annotations

"""
OpenAI Cost Functions and Handler
"""
import abc
import tiktoken
from pathlib import Path
from lazyops.types import BaseModel, validator, Field, lazyproperty
from lazyops.libs.proxyobj import ProxyObject
from typing import Optional, Union, Dict, Any, List, Tuple, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .resources import Usage
    from async_openai.schemas.chat import ChatMessage
    from transformers import PreTrainedTokenizer
    from async_openai.utils.external_config import ExternalProviderSettings, ProviderModel

pricing_file_path = Path(__file__).parent.joinpath('pricing.yaml')

class ModelCosts(BaseModel):
    """
    Represents a model's costs
    """
    unit: Optional[int] = 1000
    input: Optional[float] = 0.0
    output: Optional[float] = 0.0
    total: Optional[float] = 0.0


class ModelCostItem(BaseModel):
    """
    Represents a model's Cost Item
    """
    name: str
    aliases: Optional[List[str]] = None
    context_length: Optional[int] = 0
    costs: Optional[ModelCosts] = Field(default_factory=ModelCosts)
    endpoints: Optional[List[str]] = None

    def get_costs(
        self, 
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        usage: Optional['Usage'] = None,
        **kwargs
    ) -> float:
        """
        Gets the costs
        """
        if usage is not None:
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
        if kwargs.get('prompt_tokens'):
            input_tokens = kwargs['prompt_tokens']
        if kwargs.get('completion_tokens'):
            output_tokens = kwargs['completion_tokens']
        
        assert input_tokens is not None or output_tokens is not None or total_tokens is not None, "Must provide either input_tokens, output_tokens, or total_tokens"
        if self.costs is None: return 0.0
        cost = 0.0
        if self.costs.input:
            cost += self.costs.input * input_tokens / self.costs.unit
        if self.costs.output:
            cost += self.costs.output * output_tokens / self.costs.unit
        if self.costs.total and total_tokens is not None:
            cost += self.costs.total * total_tokens / self.costs.unit
        return cost
        

class ModelCostHandlerClass(abc.ABC):
    """
    The Model Cost Handler
    """
    def __init__(
        self,
        **kwargs,
    ):
        """
        Initializes the Model Cost Handler
        """
        self._models: Optional[Dict[str, ModelCostItem]] = None
        self._model_aliases: Optional[Dict[str, str]] = None
        self.tokenizers: Optional[Dict[str, tiktoken.Encoding]] = {}

        self.external_models: Optional[Dict[str, 'ProviderModel']] = {}
        self.external_model_aliases: Optional[Dict[str, str]] = {}
        self.external_tokenizers: Optional[Dict[str, 'PreTrainedTokenizer']] = {}

    @staticmethod
    def load_models() -> Dict[str, ModelCostItem]:
        """
        Loads the models
        """
        import yaml
        models: Dict[str, Dict[str, Any]] = yaml.safe_load(pricing_file_path.read_text())
        return {k: ModelCostItem(name = k, **v) for k, v in models.items()}

    @property
    def models(self) -> Dict[str, ModelCostItem]:
        """
        Gets the models
        """
        if self._models is None: self._models = self.load_models()
        return self._models
    
    @property
    def model_aliases(self) -> Dict[str, str]:
        """
        Gets the model aliases
        """
        if self._model_aliases is None:
            self._model_aliases = {alias: model for model, item in self.models.items() for alias in item.aliases or []}
        return self._model_aliases


    def get_external_model(self, name: str) -> Optional['ProviderModel']:
        """
        Gets the model
        """
        if name not in self.external_model_aliases and name not in self.external_models:
            raise KeyError(f"Model {name} not found: {self.external_model_aliases} / {list(self.external_models.keys())}")
            # print(f"Model {name} not found: {self.external_model_aliases} / {self.external_models}")
            # return None
        if name in self.external_model_aliases:
            name = self.external_model_aliases[name]
        return self.external_models[name]
    
    def resolve_external_model_name(self, model_name: str) -> str:
        """
        Resolves the Model Name from the model aliases
        """
        # Try to remove the provider name
        model = self.get_external_model(model_name)
        return model.name

    def resolve_model_name(self, model_name: str) -> str:
        """
        Resolves the Model Name from the model aliases
        """
        # Try to remove the version number
        if model_name in self.models:
            return model_name
        
        # if model_name in self.model_aliases:
        return self.model_aliases.get(model_name, model_name)
        
        # key = model_name.rsplit('-', 1)[0].strip()
        # if key in self.model_aliases:
        #     return self.model_aliases[key]
        # elif key in self.models:
        #     self.model_aliases[model_name] = key
        #     return key
        # raise KeyError(f"Model {key}/{model_name} not found in {self.model_aliases} / {list(self.models.keys())}")
    
    def __getitem__(self, key: str) -> ModelCostItem:
        """
        Gets a model by name
        """
        if '/' in key or key in self.external_model_aliases: return self.get_external_model(key)
        if key not in self.model_aliases and key not in self.models:
            return self.models[self.resolve_model_name(key)]
        if key in self.model_aliases:
            key = self.model_aliases[key]
        return self.models[key]
    
    def get(self, key: str, default: Optional[str] = None) -> Optional[ModelCostItem]:
        """
        Gets a model by name
        """
        try:
            return self[key]
        except KeyError:
            if default is None:
                raise KeyError(f"Model {key} not found") from None
            return self[default]
        
    def add_provider(self, provider: 'ExternalProviderSettings'):
        """
        Adds a provider to the handler
        """
        for model in provider.models:
            model_name = f'{provider.name}/{model.name}'
            self.external_models[model_name] = model
            if model.name not in self.external_model_aliases:
                self.external_model_aliases[model.name] = model_name
            for alias in model.aliases or []:
                model_alias = f'{provider.name}/{alias}'
                self.external_model_aliases[model_alias] = model_name
                if alias not in self.external_model_aliases:
                    self.external_model_aliases[alias] = model_name
        
    def add_model(self, model: str, source_model: str):
        """
        Add a model to the handler
        
        Args:
            model (str): The model name
            source_model (str): The source model name
        """
        if model in self.model_aliases or model in self.models:
            return
        
        src_model = self[source_model]
        # Add to the model aliases
        self.model_aliases[model] = src_model.name

    def get_external_tokenizer(self, name: str) -> Optional['PreTrainedTokenizer']:
        """
        Gets the tokenizer
        """
        # Remove the provider name
        model = self.get_external_model(name)
        tokenizer_name = model.tokenizer or model.name
        if tokenizer_name not in self.external_tokenizers:
            try:
                from transformers.models.auto.tokenization_auto import AutoTokenizer
            except ImportError as e:
                raise ImportError("transformers is not installed, please install it to use this feature") from e            
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.external_tokenizers[tokenizer_name] = tokenizer
        return self.external_tokenizers[tokenizer_name]

    def get_tokenizer(self, name: str) -> Optional[Union[tiktoken.Encoding, 'PreTrainedTokenizer']]:
        """
        Gets the tokenizer
        """
        # Switch the 35 -> 3.5
        # OpenAI Models don't have / in the name
        if '/' in name or name in self.external_model_aliases \
            or name in self.external_models: return self.get_external_tokenizer(name)
        
        if '35' in name: name = name.replace('35', '3.5')    
        if name not in self.tokenizers:
            if name in {'text-embedding-3-small', 'text-embedding-3-large'}:
                enc_name = 'cl100k_base'
                self.tokenizers[name] = tiktoken.get_encoding(enc_name)
            else:
                self.tokenizers[name] = tiktoken.encoding_for_model(name)
        return self.tokenizers[name]
    
    def count_chat_tokens(
        self, 
        messages: List[Union[Dict[str, str], 'ChatMessage']],
        model_name: str,
        reply_padding_token_count: Optional[int] = 3,
        message_padding_token_count: Optional[int] = 4,
        **kwargs
    ) -> int:
        """
        Returns the number of tokens in the chat.
        """
        num_tokens = 0
        tokenizer = self.get_tokenizer(model_name)
        for message in messages:
            if message.get('name'):
                num_tokens -= 1
            num_tokens += message_padding_token_count + len(tokenizer.encode(message.get('content', '')))
        num_tokens += reply_padding_token_count  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def count_tokens(
        self,
        text: Union[str, List[str]],
        model_name: str,
        **kwargs
    ) -> int:
        """
        Returns the number of tokens in the text.
        """
        tokenizer = self.get_tokenizer(model_name)
        return (
            sum(len(tokenizer.encode(t)) for t in text)
            if isinstance(text, list)
            else len(tokenizer.encode(text))
        )
    
    def get_consumption_cost(self, model_name: str, usage: 'Usage', **kwargs) -> float:
        """
        Gets the consumption cost
        """
        # Switch the 35 -> 3.5
        if '35' in model_name: model_name = model_name.replace('35', '3.5')
        model = self[model_name]
        if isinstance(usage, dict):
            from .resources import Usage
            usage = Usage(**usage)
        return model.get_costs(usage = usage, **kwargs)
    
    
    def truncate_to_max_length(self, text: str, model_name: str, context_length: Optional[int] = None, **kwargs) -> str:
        """
        Truncates the text to the max length
        """
        tokenizer = self.get_tokenizer(model_name)
        if context_length is None:
            context_length = self[model_name].context_length
        tokens = tokenizer.encode(text)
        if len(tokens) > context_length:
            tokens = tokens[-context_length:]
            decoded = tokenizer.decode(tokens)
            text = text[-len(decoded):]
        return text




class ModelContextHandlerMetaClass(type):
    """
    The Model Cost Handler
    """

    _models: Optional[Dict[str, ModelCostItem]] = None
    _model_aliases: Optional[Dict[str, str]] = None
    tokenizers: Optional[Dict[str, tiktoken.Encoding]] = {}

    def load_models(cls) -> Dict[str, ModelCostItem]:
        """
        Loads the models
        """
        import yaml
        models: Dict[str, Dict[str, Any]] = yaml.safe_load(pricing_file_path.read_text())
        return {k: ModelCostItem(name = k, **v) for k, v in models.items()}

    @property
    def models(cls) -> Dict[str, ModelCostItem]:
        """
        Gets the models
        """
        if cls._models is None:
            cls._models = cls.load_models()
        return cls._models

    @property
    def model_aliases(cls) -> Dict[str, str]:
        """
        Gets the model aliases
        """
        if cls._model_aliases is None:
            cls._model_aliases = {alias: model for model, item in cls.models.items() for alias in item.aliases or []}
        return cls._model_aliases
    
    def resolve_model_name(cls, model_name: str) -> str:
        """
        Resolves the Model Name from the model aliases
        """
        # Try to remove the version number
        key = model_name.rsplit('-', 1)[0].strip()
        if key in cls.model_aliases:
            cls.model_aliases[model_name] = cls.model_aliases[key]
        if key in cls.models:
            cls.model_aliases[model_name] = key
            return key
        raise KeyError(f"Model {model_name} not found")
    
    def __getitem__(cls, key: str) -> ModelCostItem:
        """
        Gets a model by name
        """
        if key not in cls.model_aliases and key not in cls.models:
            return cls.resolve_model_name(key)
        if key in cls.model_aliases:
            key = cls.model_aliases[key]
        return cls.models[key]
    
    def get(cls, key: str, default: Optional[str] = None) -> Optional[ModelCostItem]:
        """
        Gets a model by name
        """
        try:
            return cls[key]
        except KeyError:
            if default is None:
                raise KeyError(f"Model {key} not found") from None
            return cls[default]
        
    def add_model(cls, model: str, source_model: str):
        """
        Add a model to the handler
        
        Args:
            model (str): The model name
            source_model (str): The source model name
        """
        if model in cls.model_aliases or model in cls.models:
            return
        
        src_model = cls[source_model]
        # Add to the model aliases
        cls.model_aliases[model] = src_model.name


    def get_tokenizer(cls, name: str) -> Optional[tiktoken.Encoding]:
        """
        Gets the tokenizer
        """
        # Switch the 35 -> 3.5
        if '35' in name: name = name.replace('35', '3.5')    
        if name not in cls.tokenizers:
            if name in {'text-embedding-3-small', 'text-embedding-3-large'}:
                enc_name = 'cl100k_base'
                cls.tokenizers[name] = tiktoken.get_encoding(enc_name)
            else:
                cls.tokenizers[name] = tiktoken.encoding_for_model(name)
        return cls.tokenizers[name]
    
    def count_chat_tokens(
        cls, 
        messages: List[Union[Dict[str, str], 'ChatMessage']],
        model_name: str,
        reply_padding_token_count: Optional[int] = 3,
        message_padding_token_count: Optional[int] = 4,
        **kwargs
    ) -> int:
        """
        Returns the number of tokens in the chat.
        """
        num_tokens = 0
        tokenizer = cls.get_tokenizer(model_name)
        for message in messages:
            if message.get('name'):
                num_tokens -= 1
            num_tokens += message_padding_token_count + len(tokenizer.encode(message.get('content', '')))
        num_tokens += reply_padding_token_count  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def count_tokens(
        cls,
        text: Union[str, List[str]],
        model_name: str,
        **kwargs
    ) -> int:
        """
        Returns the number of tokens in the text.
        """
        tokenizer = cls.get_tokenizer(model_name)
        return (
            sum(len(tokenizer.encode(t)) for t in text)
            if isinstance(text, list)
            else len(tokenizer.encode(text))
        )
    
    def get_consumption_cost(cls, model_name: str, usage: 'Usage', **kwargs) -> float:
        """
        Gets the consumption cost
        """
        # Switch the 35 -> 3.5
        if '35' in model_name: model_name = model_name.replace('35', '3.5')
        model = cls[model_name]
        if isinstance(usage, dict):
            from .resources import Usage
            usage = Usage(**usage)
        return model.get_costs(usage = usage, **kwargs)
    
    def resolve_model_name(cls, model_name: str) -> str:
        """
        Resolves the Model Name from the model aliases
        """
        return cls.model_aliases.get(model_name, model_name)
    
    def truncate_to_max_length(cls, text: str, model_name: str, context_length: Optional[int] = None, **kwargs) -> str:
        """
        Truncates the text to the max length
        """
        tokenizer = cls.get_tokenizer(model_name)
        if context_length is None:
            context_length = cls[model_name].context_length

        tokens = tokenizer.encode(text)
        if len(tokens) > context_length:
            tokens = tokens[-context_length:]
            decoded = tokenizer.decode(tokens)
            text = text[-len(decoded):]
        
        return text



class ModelContextHandlerV1(metaclass = ModelContextHandlerMetaClass):
    """
    The Model Cost Handler
    """
    pass


ModelContextHandler: ModelCostHandlerClass = ProxyObject(ModelCostHandlerClass)