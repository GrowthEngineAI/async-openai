from typing import Optional
from async_openai.schemas.base import BaseSchema
from async_openai.schemas.types.edits import *


class EditSchema(BaseSchema):

    def create(
        self,
        model: Optional[str],
        instruction: Optional[str],
        input: Optional[str] = "",
        n: Optional[int] = 1,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 1.0,
        user: Optional[str] = None,
        **kwargs,
    ) -> EditResult:
        """
        Create an Edit.
        """
        request = EditRequest(
            model = model,
            instruction = instruction,
            input = input,
            n = n,
            temperature = temperature,
            top_p = top_p,
            user = user,
        )
        response = self.send(
            **request.create_edit_endpoint.get_params(**kwargs)
        )
        result = EditResult(
            _raw_request = request,
            _raw_response = response,
        )
        result.parse_result()
        return result
    
    async def async_create(
        self,
        model: Optional[str],
        instruction: Optional[str],
        input: Optional[str] = "",
        n: Optional[int] = 1,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 1.0,
        user: Optional[str] = None,
        **kwargs,
    ) -> EditResult:
        """
        [Async] Create an Edit.
        """
        request = EditRequest(
            model = model,
            instruction = instruction,
            input = input,
            n = n,
            temperature = temperature,
            top_p = top_p,
            user = user,
        )
        response = await self.async_send(
            **await request.create_edit_endpoint.async_get_params(**kwargs)
        )
        result = EditResult(
            _raw_request = request,
            _raw_response = response,
        )
        result.parse_result()
        return result

    


