
import re
import json
from typing import Optional, Callable, Dict, Union, List, Any
from .logs import logger
from .fixjson import fix_json


_json_pattern = re.compile(r"({[^}]*$|{.*})", flags=re.DOTALL)

def build_stack(json_str: str):
    stack = []
    fixed_str = ""
    open_quotes = False

    # a flag indicating whether we've seen a comma or colon most recently
    # since last opening/closing a dict or list
    last_seen_comma_or_colon = None

    for i, char in enumerate(json_str):
        if not open_quotes:
            # opening a new nested
            if char in "{[":
                stack.append(char)
                last_seen_comma_or_colon = None
            # closing a nested
            elif char in "}]":
                stack.pop()
                last_seen_comma_or_colon = None
            if char in ",:":
                last_seen_comma_or_colon = char
        # opening or closing a string, only it's not escaped
        if char == '"' and i > 0 and json_str[i - 1] != "\\":
            open_quotes = not open_quotes

        fixed_str += char

    return (stack, fixed_str, open_quotes, last_seen_comma_or_colon)



def is_truncated(json_str: str):
    """
    Check if the json string is truncated by checking if the number of opening
    brackets is greater than the number of closing brackets.
    """
    stack, _, _, _ = build_stack(json_str)
    return len(stack) > 0


def find_json_response(full_response: str, verbose: Optional[bool] = False):
    """
    Takes a full response that might contain other strings and attempts to extract the JSON payload.
    Has support for truncated JSON where the JSON begins but the token window ends before the json is
    is properly closed.
    """
    # Deal with fully included responses as well as truncated responses that only have one
    if full_response.startswith("{") and not full_response.endswith("}"):
        full_response += "}"
    
    extracted_responses = list(_json_pattern.finditer(full_response))
    if not extracted_responses:
        logger.error(
            f"Unable to find any responses of the matching type `{full_response}`"
        )
        return None

    if len(extracted_responses) > 1 and verbose:
        logger.error(f"Unexpected response > 1, continuing anyway... {extracted_responses}")

    extracted_response = extracted_responses[0]

    if is_truncated(extracted_response.group(0)):
        # Start at the same location and just expand to the end of the message
        extracted_str = full_response[extracted_response.start() :]
    else:
        extracted_str = extracted_response.group(0)

    return extracted_str

def try_load_json(
    text: str,
    object_hook: Optional[Callable] = None,
    **kwargs,
):
    """
    Attempts to load the text as JSON
    """
    try:
        return json.loads(text, object_hook = object_hook, **kwargs)
    except Exception as e1:
        try:
            return json.loads(fix_json(text), object_hook = object_hook, **kwargs)
        except Exception as e2:
            logger.error(f"Unable to load JSON. Errors: {e1}, {e2}")
            raise e2


def extract_json_response(
    full_response: str, 
    verbose: Optional[bool] = False,
    raise_exceptions: Optional[bool] = False,
    object_hook: Optional[Callable] = None,
) -> Union[Dict[str, Any], List[Any], Any]:
    """
    Returns the extracted JSON response from the full response
    """
    extracted_str = find_json_response(full_response, verbose = verbose)
    if not extracted_str:
        return None
    try:
        return try_load_json(extracted_str, object_hook = object_hook)
    except Exception as e:
        if verbose: logger.trace(f"Unable to extract JSON response from {extracted_str}", error = e)
        if raise_exceptions: raise e
        return None