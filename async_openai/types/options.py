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
    'ada': 0.004,
    'babbage': 0.005,
    'curie': 0.02,
    'davinci': 0.2,
}


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
        if self.value in {"azure", "azure_ad", "azuread"}:
            return "2022-03-01-preview"
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
    def parse_str(cls, value: str):
        if "train" in value:
            return cls.train
        elif "finetune" in value:
            return cls.finetune
        elif "fine-tune" in value:
            return cls.fine_tune
        elif "search" in value:
            return cls.search
        raise ValueError(f"Cannot convert {value} to FilePurpose")

class OpenAIModelType(str, Enum):
    """
    OpenAI Model Types
    """

    davinci = "davinci"
    curie = "curie"
    babbage = "babbage"
    ada = "ada"
    custom = "custom"

    @classmethod
    def parse_str(cls, value: str):
        if "davinci" in value:
            return cls.davinci
        elif "curie" in value:
            return cls.curie
        elif "babbage" in value:
            return cls.babbage
        elif "ada" in value:
            return cls.ada
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
    search = "search"

    @classmethod
    def parse_str(cls, value: str):
        if "completion" in value:
            return cls.completion
        elif "edit" in value:
            return cls.edit
        elif "finetune" in value:
            return cls.finetune
        elif "fine-tune" in value:
            return cls.fine_tune
        elif "train" in value:
            return cls.train
        elif "embedding" in value:
            return cls.embedding
        elif "search" in value:
            return cls.search
        raise ValueError(f"Cannot convert {value} to ModelMode")


class OpenAIModel(BaseModel):
    value: Union[str, OpenAIModelType]
    mode: Optional[Union[str, ModelMode]] = ModelMode.completion

    @lazyproperty
    def model_type(self):
        if isinstance(self.value, OpenAIModelType):
            return self.value
        return OpenAIModelType.parse_str(self.value)
    
    @lazyproperty
    def model_mode(self):
        if isinstance(self.mode, ModelMode):
            return self.mode
        return ModelMode.parse_str(self.mode)
    
    @lazyproperty
    def model(self):
        if self.model_type == OpenAIModelType.custom:
            return self.value
        return self.model_type.value
    
    @property
    def model_version(self):
        return self.model_type.model_version
    
    @property
    def edit_model(self):
        return self.model_type.edit_model
    
    @property
    def completion_model(self):
        return self.model_type.completion_model
    
    @property
    def embedding_model(self):
        return self.model_type.embedding_model
    
    @property
    def finetune_model(self):
        return self.model_type.finetune_model

    def get_cost(
        self,
        total_tokens: int = 1,
        mode: Optional[str] = None,
    ) -> float:
        """
        Returns the total cost of the model
        usage
        """
        mode = mode or self.model_mode.value
        if mode in {'completion', 'edit'}:
            return total_tokens * (_completion_prices[self.model] / 1000)
        if 'embedding' in mode:
            return total_tokens * (_embedding_prices[self.model]  / 1000)
        if 'train' in mode:
            return total_tokens * (_finetune_training_prices[self.model] / 1000)
        if 'finetune' in mode or 'fine-tune' in mode:
            return total_tokens * (_finetune_usage_prices[self.model] / 1000)
        raise ValueError(f"Invalid mode {mode}")
    
    def dict(self, **kwargs):
        if self.mode == ModelMode.completion:
            return self.completion_model
        if self.mode == ModelMode.edit:
            return self.edit_model
        if self.mode == ModelMode.embedding:
            return self.embedding_model
        return self.finetune_model if self.mode == ModelMode.finetune else self.value


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