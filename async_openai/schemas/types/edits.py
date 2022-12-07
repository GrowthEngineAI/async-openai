from typing import List, Any, Optional, Union, Dict, Type
from async_openai.types import BaseModel, lazyproperty
from async_openai.schemas.types.base import BaseResult, Method, BaseEndpoint

__all__ = [
    'EditChoice',
    'EditRequest',
    'EditResult',
]

class EditChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Any]
    finish_reason: Optional[str]

class EditRequest(BaseModel):
    model: Optional[str]
    instruction: Optional[str]
    input: Optional[str] = ""
    n: Optional[int] = 1
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    user: Optional[str] = None

    @property
    def create_edit_endpoint(self) -> BaseEndpoint:
        return BaseEndpoint(
            method = Method.POST,
            url ='/edits',
            data = self.dict(
                exclude_none = True
            )
        )


class EditResult(BaseResult):
    choices: Optional[List[EditChoice]]
    _choice_model: Optional[Type[EditChoice]] = EditChoice
    _request: Optional[EditRequest] = None




