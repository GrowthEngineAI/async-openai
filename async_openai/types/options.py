from enum import Enum
from typing import Optional, Union
from lazyops.types import BaseModel, lazyproperty

"""
Pricing Options

# https://openai.com/api/pricing/
"""

_image_prices = {
    'small': 0.016,
    'medium': 0.018,
    'large': 0.02,
}

# price per 1k/tokens
_completion_prices = {
    'ada': 0.0004,
    'babbage': 0.0005,
    'curie': 0.002,
    'davinci': 0.02,
}

_finetune_training_prices = {
    'ada': 0.0004,
    'babbage': 0.0006,
    'curie': 0.003,
    'davinci': 0.03,
}

_finetune_usage_prices = {
    'ada': 0.0016,
    'babbage': 0.0024,
    'curie': 0.012,
    'davinci': 0.12,
}

_embedding_prices = {
    # 'ada': 0.004,
    'ada': 0.0001,
    'babbage': 0.005,
    'curie': 0.02,
    'davinci': 0.2,
}

_chat_prices = {
    'gpt-3.5-turbo': 0.002,

}

_chat_gpt_prices = {
    'gpt-4-32k': {
        'prompt': 0.06,
        'completion': 0.12,
    },
    'gpt-3.5-turbo-16k': {
        'prompt': 0.003,
        'completion': 0.004,
    },
    'gpt-35-turbo-16k': {
        'prompt': 0.003,
        'completion': 0.004,
    },
    'gpt-3-turbo-16k': {
        'prompt': 0.003,
        'completion': 0.004,
    },
    'gpt-4': {
        'prompt': 0.03,
        'completion': 0.06,
    },
    'gpt-3.5-turbo': {
        'prompt': 0.0015,
        'completion': 0.002,
    },
    'gpt-35-turbo': {
        'prompt': 0.0015,
        'completion': 0.002,
    },
    'gpt-3-turbo': {
        'prompt': 0.0015,
        'completion': 0.002,
    },
    'gpt-3.5-turbo-instruct': {
        'prompt': 0.0015,
        'completion': 0.002,
    },
    'gpt-35-turbo-instruct': {
        'prompt': 0.0015,
        'completion': 0.002,
    },
    'gpt-3-turbo-instruct': {
        'prompt': 0.0015,
        'completion': 0.002,
    },

}

# TODO rework this module

_cost_modes = {
    'embedding': _embedding_prices,
    'train': _finetune_training_prices,
    'finetune': _finetune_usage_prices,
    'completion': _completion_prices,
}

def get_arch(
    model_name: str,
) -> str:
    """
    Get the arch
    """
    for arch in {
        'babbage',
        'curie',
        'davinci',
        'ada',
    }:
        if arch in model_name:
            return arch

def get_consumption_cost(
    model_name: str,
    total_tokens: int = 1,
    default_token_cost: Optional[float] = 0.00001,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    mode: Optional[str] = None,
) -> float:
    """
    Returns the total cost of the model
    usage
    """
    if prompt_tokens and completion_tokens:
        total_tokens = prompt_tokens + completion_tokens
    if (not mode or mode == 'chat') and any(
        arch in model_name for arch in {
            'gpt-3.5',
            'gpt-35',
            'gpt-3',
            'gpt-4',
        }):
            return next(
                (
                    (
                        prompt_tokens * (_chat_gpt_prices[gpt_model]['prompt'] / 1000)
                    )
                    + (
                        completion_tokens * (_chat_gpt_prices[gpt_model]['completion'] / 1000)
                    )
                    if prompt_tokens and completion_tokens
                    else total_tokens
                    * (
                        (
                            (_chat_gpt_prices[gpt_model]['prompt'] + _chat_gpt_prices[gpt_model]['completion']) / 2
                        )
                        / 1000
                    )
                    for gpt_model in _chat_gpt_prices
                    if gpt_model in model_name
                ),
                total_tokens * (_chat_prices['gpt-3.5-turbo'] / 1000),
            )

    arch = get_arch(model_name)
    return  total_tokens * (_cost_modes[mode][arch] / 1000) if mode in _cost_modes else total_tokens * default_token_cost



class ApiType(str, Enum):
    azure = "azure"
    openai = "openai"
    open_ai = "openai"
    azure_ad = "azure_ad"
    azuread = "azure_ad"

    def get_version(
        self, 
        version: Optional[str] = None
    ):
        if self.value in {"azure", "azure_ad", "azuread"} and not version:
            return "2023-07-01-preview"
        return version


class FilePurpose(str, Enum):
    """
    File Purpose
    """

    finetune = "fine-tune"
    fine_tune = "fine-tune"
    train = "fine-tune-train"
    search = "search"

    @classmethod
    def parse_str(cls, value: Union[str, 'FilePurpose'], raise_error: bool = True):
        if isinstance(value, cls): return value
        if "train" in value:
            return cls.train
        elif "finetune" in value:
            return cls.finetune
        elif "fine-tune" in value:
            return cls.fine_tune
        elif "search" in value:
            return cls.search
        if not raise_error: return None
        raise ValueError(f"Cannot convert {value} to FilePurpose")


class OpenAIModelType(str, Enum):
    """
    OpenAI Model Types
    """
    text = "text"
    audio = "audio"
    code = "code"
    chat = "chat"
    custom = "custom"

    @classmethod
    def parse(cls, value: Union[str, 'OpenAIModelType'], raise_error: bool = True):
        if isinstance(value, cls): return value
        if "text" in value:
            return cls.text
        elif "audio" in value:
            return cls.audio
        elif "code" in value:
            return cls.code
        elif "gpt-3.5" in value or "gpt-4" in value or "chat" in value:
            return cls.chat
        return cls.custom


class OpenAIModelArch(str, Enum):
    """
    OpenAI Model Architectures
    """

    davinci = "davinci"
    curie = "curie"
    babbage = "babbage"
    ada = "ada"
    chat = "gpt-3.5"
    chat_gpt4 = "gpt-4"
    custom = "custom"

    @classmethod
    def parse(cls, value: Union[str, 'OpenAIModelArch'], raise_error: bool = True):
        if isinstance(value, cls): return value
        if "davinci" in value:
            return cls.davinci
        elif "curie" in value:
            return cls.curie
        elif "babbage" in value:
            return cls.babbage
        elif "ada" in value:
            return cls.ada
        elif "gpt-4" in value:
            return cls.chat_gpt4
        elif "gpt-3.5" in value or "gpt-3" in value or "chat" in value:
            return cls.chat
        return cls.custom


    @lazyproperty
    def model_version(self):
        return "003" if self.value == "davinci" else "001"

    @lazyproperty
    def edit_model(self):
        return f"text-{self.value}-edit-{self.model_version}"
    
    @lazyproperty
    def completion_model(self):
        return f"text-{self.value}-{self.model_version}"
    
    @lazyproperty
    def embedding_model(self):
        return f"text-similarity-{self.value}-{self.model_version}"
    
    @lazyproperty
    def chat_model(self):
        return 'gpt-3.5-turbo' if self.value == 'gpt-3.5' else self.value
    
    @lazyproperty
    def finetune_model(self):
        return self.value
    

class ModelMode(str, Enum):
    """
    Model Mode
    """

    completion = "completion"
    edit = "edit"
    finetune = "finetune"
    fine_tune = "finetune"
    train = "train"
    embedding = "embedding"
    similiarity = "similiarity"
    search = "search"
    chat = "chat"

    @classmethod
    def parse(cls, value: Union[str, 'ModelMode'], raise_error: bool = True):
        if isinstance(value, cls): return value
        if "completion" in value:
            return cls.completion
        if "edit" in value:
            return cls.edit
        if "finetune" in value:
            return cls.finetune
        if "fine-tune" in value:
            return cls.fine_tune
        if "train" in value:
            return cls.train
        if "embedding" in value:
            return cls.embedding
        if "search" in value:
            return cls.search
        if "similiarity" in value:
            return cls.similiarity
        if "gpt-3.5" in value or "gpt-35" in value or 'gpt-4' in value or "chat" in value:
            return cls.chat
        if "text" in value:
            return cls.completion
        if not raise_error: return None
        raise ValueError(f"Cannot convert {value} to ModelMode")
    
    @classmethod
    def get_text_modes(cls):
        return [
            cls.completion, 
            cls.edit,
            cls.embedding,
            cls.similiarity, 
            cls.search, 
            # cls.chat
        ]

class OpenAIModel(object):

    def __init__(
        self, 
        value: str, 
        **kwargs
    ):
        self.src_value = value
        self.src_splits = value.split("-")
        self.mode: ModelMode = kwargs.get("mode")
        self.model_arch: OpenAIModelArch = kwargs.get("model_arch")
        self.model_type: OpenAIModelType = kwargs.get("model_type")
        self.version: str = kwargs.get("version")
        self.parse_values()

    def parse_values(self):
        """
        Parse the source values into the correct parts
        """
        self.mode = ModelMode.parse((self.mode or self.src_value), raise_error = False) or ModelMode.completion
        self.model_arch = OpenAIModelArch.parse((self.model_arch or self.src_value), raise_error = False)
        self.model_type = OpenAIModelType.parse(
            (self.model_type or \
                ("text" if self.mode in ModelMode.get_text_modes() else self.src_value)
            ), raise_error = False)
        if not self.version:
            ver_values = [x for x in self.src_splits if (x[0].isdigit() and x[-1].isdigit())]
            if ver_values:
                self.version = '-'.join(ver_values)
                if self.mode in {ModelMode.chat}:
                    if self.version in {'35', '3.5', '4', '3', '16k', '32k'}:
                        self.version = None
                    else:
                        self.version = self.version.rsplit('-', 1)[-1]

            elif self.mode == ModelMode.completion:
                self.version = "003" if self.model_arch == "davinci" else "001"
            elif self.mode == ModelMode.chat:
                pass
            elif self.model_type != OpenAIModelType.custom:
                self.version = "001"


    @lazyproperty
    def value(self) -> str:
        """
        The value of the model
        """
        if self.model_arch == OpenAIModelArch.custom or self.model_type == OpenAIModelType.custom:
            return self.src_value
        if self.mode == ModelMode.chat:
            return f'{self.src_value}-{self.version}' if self.version else self.src_value
        t = f'{self.model_type.value}'
        if self.mode != ModelMode.completion:
            t += f'-{self.mode.value}'
        t += f'-{self.model_arch.value}'
        if self.version:
            t += f'-{self.version}'
        return t

    def dict(self, *args, **kwargs):
        return {
            "value": self.value,
            "mode": self.mode.value,
            "model_arch": self.model_arch.value,
            "model_type": self.model_type.value,
            "version": self.version,
        }
    
    def __str__(self):
        return f'OpenAIModel(value="{self.value}", mode="{self.mode}", model_arch="{self.model_arch}", model_type="{self.model_type}", version="{self.version}")'

    def __repr__(self) -> str:
        return f'OpenAIModel(value="{self.value}", mode="{self.mode}", model_arch="{self.model_arch}", model_type="{self.model_type}", version="{self.version}")'

    def __json__(self):
        return self.value
    
    # def __dict__(self):
    #     return self.value

    def get_cost(
        self,
        total_tokens: int = 1,
        mode: Optional[str] = None,
        raise_error: bool = True,
        default_token_cost: Optional[float] = 0.00001,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
    ) -> float:
        """
        Returns the total cost of the model
        usage
        """
        if prompt_tokens and completion_tokens:
            total_tokens = prompt_tokens + completion_tokens

        mode = mode or self.mode.value
        if mode in ['completion', 'edit']:
            return total_tokens * (_completion_prices[self.model_arch.value] / 1000)
        if mode in ['chat']:
            return next(
                (
                    (
                        prompt_tokens * (_chat_gpt_prices[gpt_model]['prompt'] / 1000)
                    )
                    + (
                        completion_tokens * (_chat_gpt_prices[gpt_model]['completion'] / 1000)
                    )
                    if prompt_tokens and completion_tokens
                    else total_tokens
                    * (
                        (
                            (_chat_gpt_prices[gpt_model]['prompt'] + _chat_gpt_prices[gpt_model]['completion']) / 2
                        )
                        / 1000
                    )
                    for gpt_model in _chat_gpt_prices
                    if gpt_model in self.src_value
                ),
                total_tokens * (_chat_prices['gpt-3.5-turbo'] / 1000),
            )
        if 'embedding' in mode:
            return total_tokens * (_embedding_prices[self.model_arch.value]  / 1000)
        if 'train' in mode:
            return total_tokens * (_finetune_training_prices[self.model_arch.value] / 1000)
        if 'finetune' in mode or 'fine-tune' in mode:
            return total_tokens * (_finetune_usage_prices[self.model_arch.value] / 1000)
        if raise_error: raise ValueError(f"Invalid mode {mode}")
        return total_tokens * default_token_cost


class EditModels(str, Enum):
    """
    Just the base models available
    """
    davinci = "text-davinci-edit-003"
    curie = "text-curie-edit-001"
    babbage = "text-babbage-edit-001"
    ada = "text-ada-edit-001"

    @lazyproperty
    def model_type(self) -> str:
        return self.value.split("-")[1]

class EmbeddingModels(str, Enum):
    """
    Just the base models available
    """
    davinci = "text-similarity-davinci-003"
    curie = "text-similarity-curie-001"
    babbage = "text-similarity-babbage-001"
    ada = "text-similarity-ada-001"

    @lazyproperty
    def model_type(self) -> str:
        return self.value.split("-")[2]

class CompletionModels(str, Enum):
    """
    Just the base models available
    """
    davinci = "text-davinci-003"
    curie = "text-curie-001"
    babbage = "text-babbage-001"
    ada = "text-ada-001"

    @lazyproperty
    def model_type(self) -> str:
        return self.value.split("-")[1]


class FinetuneModels(str, Enum):
    """
    Supported finetune models.
    """
    ada = "ada"
    babbage = "babbage"
    curie = "curie"
    davici = "davici"

    @lazyproperty
    def model_type(self):
        return self.value

class ImageSize(str, Enum):
    """
    Size of the image
    """

    small = "256x256"
    medium = "512x512"
    large = "1024x1024"

    @lazyproperty
    def image_type(self):
        if self.value == "256x256":
            return 'small'
        if self.value == "512x512":
            return 'medium'
        if self.value == "1024x1024":
            return 'large'
        raise ValueError(f"Cannot convert {self.value} to Kind")

    @classmethod
    def from_str(cls, value: str) -> "ImageSize":
        """
        :param value: Size of the image
        :type value: str
        :return: ImageSize
        :rtype: ImageSize
        """
        if value == "256x256":
            return cls.small
        if value == "512x512":
            return cls.medium
        if value == "1024x1024":
            return cls.large
        try:
            return cls(value)
        except ValueError as e:
            raise ValueError(f"Cannot convert {value} to ImageSize") from e
    
    def get_cost(
        self,
        total_images: int = 1,
    ) -> float:
        """
        Returns the total cost of the model
        usage
        """
        return total_images * _image_prices[self.value]

class ImageFormat(str, Enum):
    """
    Format of the image
    """

    url = "url"
    b64 = "b64_json"
    b64_json = "b64_json"