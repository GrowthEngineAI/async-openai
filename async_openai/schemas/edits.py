from typing import Optional
from async_openai.schemas.base import BaseSchema
from async_openai.schemas.types.edits import *


class EditSchema(BaseSchema):

    def create(
        self,
        instruction: Optional[str],
        input: Optional[str] = "",
        model: Optional[str] = 'text-davinci-edit-001',
        n: Optional[int] = 1,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 1.0,
        user: Optional[str] = None,
        **kwargs,
    ) -> EditResult:
        """
        Creates a new edit for the provided input, instruction, and parameters

        Usage:

        ```python
        >>> result = OpenAI.edits.create(
        >>>    model = 'text-davinci-edit-001',
        >>>    instruction = 'What day of the wek is it?',
        >>>    instruction = 'Fix the spelling mistakes'
        >>> )
        ```

        **Parameters:**

        * **instruction** - *(required)* The instruction that tells the model how to edit the prompt

        * **input** - *(optional)* The input text to use as a starting point for the edit
        Default: `""`

        * **model** - *(optional)* ID of the model to use. You can use the List models API 
        to see all of your available models,  or see our Model overview for descriptions of them.
        Default: `text-davinci-edit-001`

        * **n** - *(optional)* How many edits to generate for the input and instruction.
        Default: `1`

        * **temperature** - *(optional)* What sampling temperature to use. Higher values means 
        the model will take more risks. Try 0.9 for more creative applications, and 0 (argmax sampling) 
        for ones with a well-defined answer. We generally recommend altering this or `top_p` but not both.
        Default: `1.0`

        * **top_p** - *(optional)* An alternative to sampling with `temperature`, called nucleus 
        sampling, where the model considers the results of the tokens with `top_p` probability mass. 
        So `0.1` means only  the tokens comprising the top 10% probability mass are considered.
        We generally recommend altering this or `temperature` but not both
        Default: `1.0`

        Returns: `EditResult`
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
        instruction: Optional[str],
        input: Optional[str] = "",
        model: Optional[str] = 'text-davinci-edit-001',
        n: Optional[int] = 1,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 1.0,
        user: Optional[str] = None,
        **kwargs,
    ) -> EditResult:
        """
        Creates a new edit for the provided input, instruction, and parameters

        Usage:

        ```python
        >>> result = await OpenAI.edits.async_create(
        >>>    model = 'text-davinci-edit-001',
        >>>    instruction = 'What day of the wek is it?',
        >>>    instruction = 'Fix the spelling mistakes'
        >>> )
        ```

        **Parameters:**

        * **instruction** - *(required)* The instruction that tells the model how to edit the prompt

        * **input** - *(optional)* The input text to use as a starting point for the edit
        Default: `""`

        * **model** - *(optional)* ID of the model to use. You can use the List models API 
        to see all of your available models,  or see our Model overview for descriptions of them.
        Default: `text-davinci-edit-001`

        * **n** - *(optional)* How many edits to generate for the input and instruction.
        Default: `1`

        * **temperature** - *(optional)* What sampling temperature to use. Higher values means 
        the model will take more risks. Try 0.9 for more creative applications, and 0 (argmax sampling) 
        for ones with a well-defined answer. We generally recommend altering this or `top_p` but not both.
        Default: `1.0`

        * **top_p** - *(optional)* An alternative to sampling with `temperature`, called nucleus 
        sampling, where the model considers the results of the tokens with `top_p` probability mass. 
        So `0.1` means only  the tokens comprising the top 10% probability mass are considered.
        We generally recommend altering this or `temperature` but not both
        Default: `1.0`

        Returns: `EditResult`
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

    


