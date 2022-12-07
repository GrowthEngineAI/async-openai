from enum import Enum
from pydantic import validator
from typing import List, Any, Optional, Union, Dict, Type
from async_openai.types import BaseModel, lazyproperty
from async_openai.schemas.types.base import BaseResult, BaseEndpoint, Method

__all__ = [
    'CompletionChoice',
    'CompletionModels',
    'CompletionRequest',
    'CompletionResult',
]

class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Any]
    finish_reason: Optional[str]

class CompletionModels(str, Enum):
    """
    Just the base models available
    """
    davinci = "text-davinci-003"
    curie = "text-curie-001"
    babbage = "text-babbage-001"
    ada = "text-ada-001"

class CompletionRequest(BaseModel):
    model: Optional[Union[str, Any]] = CompletionModels.curie
    prompt: Optional[str] = '<|endoftext|>'
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    best_of: Optional[int] = 1
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

    @validator('max_tokens')
    def validate_max_tokens(cls, v: int) -> int:
        """
        Max tokens is 4096
        https://beta.openai.com/docs/api-reference/completions/create#completions/create-max-tokens
        """
        return None if v is None else max(0, min(v, 4096))
    
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
    
    @property
    def create_endpoint(self) -> BaseEndpoint:
        return BaseEndpoint(
            method = Method.POST,
            url = '/completions',
            data = self.dict(exclude_none = True)
        )


class CompletionResult(BaseResult):
    choices: Optional[List[CompletionChoice]]

    _request: Optional[CompletionRequest] = None
    _choice_model: Optional[Type[CompletionChoice]] = CompletionChoice

    @property
    def completion_text(self) -> str:
        return ''.join(choice.text for choice in self.choices) if self.choices else ''











