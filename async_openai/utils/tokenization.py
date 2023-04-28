from __future__ import annotations

import functools
import tiktoken
from typing import Optional, Union, List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from async_openai.schemas.chat import ChatMessage

def modelname_to_contextsize(modelname: str) -> int:
    """
    Calculate the maximum number of tokens possible to generate for a model.
    
    text-davinci-003: 4,097 tokens
    text-curie-001: 2,048 tokens
    text-babbage-001: 2,048 tokens
    text-ada-001: 2,048 tokens
    code-davinci-002: 8,000 tokens
    code-cushman-001: 2,048 tokens
    gpt-3.5-turbo: 4,096 tokens
    gpt-4: 8,192 tokens
    gpt-4-32k: 32,768 tokens

    Args:
        modelname: The modelname we want to know the context size for.

    Returns:
        The maximum context size
    """
    if modelname == "code-davinci-002":
        return 8000
    
    elif modelname in {
        "text-curie-001",
        "text-babbage-001",
        "text-ada-001",
        "code-cushman-001",
    }:
        return 2048

    elif modelname in {
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0314"
    }:
        return 4096
    
    elif modelname in {
        "gpt-4-32k",
        "gpt-4-32k-0314"
    }:
        return 32768
    elif modelname in {
        "gpt-4", 
        "gpt-4-0314"
    }:
        return 8192
    else:
        return 4097
    

def get_encoder(
    model_name: str,
) -> tiktoken.Encoding:
    """
    Returns the correct encoder for the model name.
    """
    if any(name in model_name for name in {'gpt-4', 'gpt-3.5'}):
        return tiktoken.encoding_for_model(model_name)

    encoder = "gpt2"
    if model_name in {"text-davinci-003", "text-davinci-002"}:
        encoder = "p50k_base"
    if model_name.startswith("code"):
        encoder = "p50k_base"
    
    return tiktoken.get_encoding(encoder)

@functools.lru_cache(maxsize = 2048)
def get_token_count(
    text: str,
    model_name: str,
) -> int:
    """
    Returns the number of tokens in the text.
    """
    return len(get_encoder(model_name).encode(text))


def get_max_tokens(
    text: Union[str, List[str]],
    model_name: str,
    max_tokens: Optional[int] = None,
    padding_token_count: Optional[int] = 16 # tokens added to make sure we do not go over the limit
):
    """
    Returns the maximum number of tokens that can be generated for a model.
    """
    max_model_tokens = modelname_to_contextsize(model_name) - padding_token_count
    if isinstance(text, list):
        all_text_tokens = [get_token_count(t, model_name) for t in text]
        text_tokens = max(all_text_tokens)
    else:
        text_tokens = get_token_count(text, model_name)
    max_input_tokens = max_model_tokens - text_tokens
    if max_tokens is None:
        return max_input_tokens
    return min(max_input_tokens, max_tokens)
    # return modelname_to_contextsize(model_name) - get_token_count(text, model_name)

def get_max_chat_tokens(
    messages: List[Union[Dict[str, str], 'ChatMessage']],
    model_name: str,
    max_tokens: Optional[int] = None,
    reply_padding_token_count: Optional[int] = 3,
    message_padding_token_count: Optional[int] = 4,
    padding_token_count: Optional[int] = 16 # tokens added to make sure we do not go over the li
):
    """
    Returns the maximum number of tokens that can be generated for a model.
    """

    num_tokens = 0
    for message in messages:
        if message.get('name'):
            num_tokens -= 1
        num_tokens += message_padding_token_count + get_token_count(message.get('content', ''), model_name)

    num_tokens += reply_padding_token_count  # every reply is primed with <|start|>assistant<|message|>
    max_model_tokens = modelname_to_contextsize(model_name) - padding_token_count
    max_input_tokens = max_model_tokens - num_tokens
    if max_tokens is None:
        return max_input_tokens
    return min(max_input_tokens, max_tokens)
