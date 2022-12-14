from typing import Optional, Type, Any, Union, List, Dict
from lazyops.types import validator, lazyproperty

from async_openai.types.options import OpenAIModel, OpenAIModelType
from async_openai.types.resources import BaseResource
from async_openai.types.responses import BaseResponse
from async_openai.types.routes import BaseRoute


__all__ = [
    'EditChoice',
    'EditObject',
    'EditResponse',
    'EditRoute',
]


class EditChoice(BaseResource):
    text: str
    index: int
    logprobs: Optional[Any]
    finish_reason: Optional[str]

class EditObject(BaseResource):
    model: Optional[Union[str, OpenAIModel, Any]] = OpenAIModelType.curie
    instruction: Optional[str]
    input: Optional[str] = ""
    n: Optional[int] = 1
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    user: Optional[str] = None

    @validator('model', pre=True, always=True)
    def validate_model(cls, v) -> OpenAIModel:
        """
        Validate the model
        """
        return OpenAIModel(value = v, mode = 'edit')


class EditResponse(BaseResponse):
    choices: Optional[List[EditChoice]]
    choice_model: Optional[Type[BaseResource]] = EditChoice


    @lazyproperty
    def text(self) -> str:
        """
        Returns the text for the edits
        """
        if self.choices:
            return ''.join([choice.text for choice in self.choices])
        return self._response.text


class EditRoute(BaseRoute):
    input_model: Optional[Type[BaseResource]] = EditObject
    response_model: Optional[Type[BaseResource]] = EditResponse

    @lazyproperty
    def api_resource(self):
        return 'edits'

    def create(
        self, 
        input_object: Optional[Type[BaseResource]] = None,
        **kwargs
    ) -> EditResponse:
        """
        
        """
        return super().create(input_object = input_object, **kwargs)
    
    async def async_create(
        self, 
        input_object: Optional[Type[BaseResource]] = None,
        **kwargs
    ) -> EditResponse:
        """
        
        """
        return await super().async_create(input_object = input_object, **kwargs)

    



