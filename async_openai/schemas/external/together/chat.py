from __future__ import annotations

"""
Together.xyz Chat Route
"""
import json
from ...chat import (
    ChatRoute as BaseChatRoute,
    ChatObject as BaseChatObject,
    ChatResponse as BaseChatResponse,
    ChatChoice as BaseChatChoice,
    ChatMessage as BaseChatMessage, 
    Function, FunctionCall, Tool, logger
)
from lazyops.types import validator, root_validator, BaseModel, lazyproperty, Field, PYD_VERSION
from async_openai.types.context import ModelContextHandler
from typing import Any, Dict, List, Optional, Union, Set, Type, TYPE_CHECKING

# if PYD_VERSION == 2:
#     from pydantic import model_validator
# else:
#     from lazyops.types.models import root_validator

class ChatObject(BaseChatObject):
    model: Optional[str] = "mistralai/Mistral-7B-Instruct-v0.1"
    response_format: Optional[Dict[str, Union[str, Dict[str, Any]]]] = None
    is_json_mode: Optional[bool] = Field(None, exclude = True)

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
        
        v = ModelContextHandler.resolve_external_model_name(v)
        return v


    """
    Handle Validation for JSON Mode

    JSON mode corrals the LLM into outputting JSON conforming to a provided schema. 
    To activate JSON mode, provide the response_format parameter to the Chat Completions 
    API with {"type": "json_object"}. The JSON Schema can be specified with the schema 
    property of response_format. The schema property should be a JSON Schema object.
    """

    # if PYD_VERSION == 2:
    #     @model_validator(mode = 'after')
    #     def validate_obj(self):
    #         """
    #         Validate the object
    #         """
        
    # else:

    @root_validator(pre = True)
    def validate_obj(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the object
        """
        if values.get('functions'):
            if not all(isinstance(f, Function) for f in values['functions']):
                values['functions'] = [Function(**f) for f in values['functions']]
            if not values.get('function_call'):
                values['function_call'] = 'auto'
        
        if values.get('response_format', {}).get('type') == 'json_object':
            values['is_json_mode'] = True
            if not values['response_format'].get('schema') and  values.get('functions'):
                func = values['functions'][0] if \
                    len(values['functions']) == 1 or \
                    values.get('function_call') == 'auto' else \
                    next((f for f in values['functions'] if f.name == values['function_call']))
                
                assert func, 'No function found'
                schema = func.model_json_schema()
                # _ = schema.pop('additionalProperties', None)
                values['response_format']['schema'] = schema
                # logger.info(values["response_format"], prefix = 'JSON Mode')

                
        # Disable tools if response format is json_object
        elif values.get('tools'):
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
        return values
    
    def dict(self, **kwargs) -> Dict[str, Any]:
        """
        Return the dict
        """
        exclude: Set[str] = kwargs.pop('exclude', None) or set()
        if self.is_json_mode:
            exclude.add('tools')
            exclude.add('tool_choice')
            exclude.add('functions')
            exclude.add('function_call')
        
        return super().dict(exclude = exclude, **kwargs)
        

class ChatMessage(BaseChatMessage):
    
    """
    Handle some validation here
    """

    @root_validator(pre = True)
    def validate_message(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the object
        """
        # if values.get('content') and '"arguments"' in values['content']:
        #     content = values.pop('content')
        #     try:
        #         data = json.loads(content)
        #     except Exception as e:
        #         try:
        #             content = content.split('\n', 1)[-1].strip()
        #             data = json.loads(content)
        #         except Exception as e:
        #             logger.error(f'Invalid JSON: {content}: {e}')
        #             raise e
        #     values['function_call'] = FunctionCall(**data)
        if values.get('tool_calls'):
            for tc in values['tool_calls']:
                if tc.get('type') == 'function' and tc.get('function'):
                    func = FunctionCall(**tc['function'])
                    values['function_call'] = func
                    break
        return values



class ChatChoice(BaseChatChoice):
    message: ChatMessage


class ChatResponse(BaseChatResponse):

    input_object: Optional[ChatObject] = None
    choice_model: Optional[Type[ChatChoice]] = ChatChoice


class ChatRoute(BaseChatRoute):
    input_model: Optional[Type[ChatObject]] = ChatObject
    response_model: Optional[Type[ChatResponse]] = ChatResponse