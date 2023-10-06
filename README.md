# async-openai
 Unofficial Async Python client library for the [OpenAI](https://openai.com) API based on [Documented Specs](https://beta.openai.com/docs/api-reference/making-requests)

 **Latest Version**: [![PyPI version](https://badge.fury.io/py/async-openai.svg)](https://badge.fury.io/py/async-openai)

 **[Official Client](https://github.com/openai/openai-python)**

## Features

- [x] Asyncio based with Sync and Async Support with `httpx`

- [ ] Supports all API endpoints

    - [x] `Completions`: [Docs](https://beta.openai.com/docs/api-reference/completions)
    
    - [x] `Edits`: [Docs](https://beta.openai.com/docs/api-reference/edits)
    
    - [x] `Embeddings`: [Docs](https://beta.openai.com/docs/api-reference/embeddings)

    - [ ] `Images`: [Docs](https://beta.openai.com/docs/api-reference/images)

    - [ ] `Files`: [Docs](https://beta.openai.com/docs/api-reference/files)

    - [ ] `Finetuning`: [Docs](https://beta.openai.com/docs/api-reference/fine-tunes)

    - [x] `Models`: [Docs](https://beta.openai.com/docs/api-reference/models)

    - [ ] `Moderations`: [Docs](https://beta.openai.com/docs/api-reference/moderations)

    - [ ] `Search`: [Docs](#)

- [x] Strongly typed validation of requests and responses with `Pydantic` Models with transparent 
    access to the raw response and object-based results.

- [x] Handles Retries automatically through `backoff`

- [x] Supports Local and Remote Cloud Object Storage File Handling Asyncronously through `file-io`

    - [x] Supports `S3`: `s3://bucket/path/to/file.txt`
    
    - [x] Supports `GCS`: `gs://bucket/path/to/file.txt`

    - [x] Supports `Minio`: `minio://bucket/path/to/file.txt`

- [x] Supports `limited` cost tracking for `Completions` and `Edits` requests (when stream is not enabled)

---
 
## Installation

```bash
# Install from stable
pip install async-openai

# Install from dev/latest
pip install git+https://github.com/GrowthEngineAI/async-openai.git

```

### Quick Usage

```python

import asyncio
from async_openai import OpenAI, settings, CompletionResponse

# Environment variables should pick up the defaults
# however, you can also set them explicitly.

# `api_key` - Your OpenAI API key.                  Env: [`OPENAI_API_KEY`]
# `url` - The URL of the OpenAI API.                Env: [`OPENAI_URL`]
# `api_type` - The OpenAI API type.                 Env: [`OPENAI_API_TYPE`]
# `api_version` - The OpenAI API version.           Env: [`OPENAI_API_VERSION`]
# `organization` - The OpenAI organization.         Env: [`OPENAI_ORGANIZATION`]
# `proxies` - A dictionary of proxies to be used.   Env: [`OPENAI_PROXIES`]
# `timeout` - The timeout in seconds to be used.    Env: [`OPENAI_TIMEOUT`]
# `max_retries` - The number of retries to be used. Env: [`OPENAI_MAX_RETRIES`]

OpenAI.configure(
    api_key = "sk-XXXX",
    organization = "org-XXXX",
    debug_enabled = False,
)

# Alternatively you can configure the settings through environment variables
# settings.configure(
#    api_key = "sk-XXXX",
#     organization = "org-XXXX",
# )


# [Sync] create a completion
# Results return a CompletionResult object
result: CompletionResponse = OpenAI.completions.create(
    prompt = 'say this is a test',
    max_tokens = 4,
    stream = True
)

# print the completion text
# which are concatenated together from the result['choices'][n]['text']

print(result.text)

# print the number of choices returned
print(len(result))

# get the cost consumption for the request
print(result.consumption)

# [Async] create a completion
# All async methods are generally prefixed with `async_`

result: CompletionResponse = asyncio.run(
    OpenAI.completions.async_create(
        prompt = 'say this is a test',
        max_tokens = 4,
        stream = True
    )
)

```

### Initialize Clients Manually

```python

from async_openai import OpenAI

# Configure your primary client (default)


OpenAI.configure(
    api_key = "sk-XXXX",
    organization = "org-XXXX",
    debug_enabled = False,

    # Azure Configuration
    azure_api_base = 'https://....openai.azure.com/',
    azure_api_version = '2023-07-01-preview',
    azure_api_key = '....',
)

# Returns the default client (openai)
oai = OpenAI.init_api_client()

# Configure your secondary client (azure) and use it directly
az = OpenAI.init_api_client('az', set_as_default = False, debug_enabled = True)
result = az.completions.create(
    prompt = 'say this is a test',
    max_tokens = 4,
    stream = True
)


# Use the default client (openai)
result = OpenAI.completions.create(
    prompt = 'say this is a test',
    max_tokens = 4,
    stream = True
)
# Or 
result = oai.completions.create(
    prompt = 'say this is a test',
    max_tokens = 4,
    stream = True
)

```

### Handling Errors, Retries, and Rotations

The below will show you how to rotate between multiple clients when you hit an error.

**Important** Auto-rotation is only supported with `chat_create` and `async_chat_create` methods. Otherwise, you should handle the rotation manually.

```python

import asyncio
from async_openai import OpenAI, ChatResponse
from async_openai.utils import logger

OpenAI.configure(
    api_key = "sk-XXXX",
    organization = "org-XXXX",
    debug_enabled = False,

    # Azure Configuration
    azure_api_base = 'https://....openai.azure.com/',
    azure_api_version = '2023-07-01-preview',
    azure_api_key = '....',

    # This will allow you to auto rotate clients when you hit an error.
    # But only if you have multiple clients configured and are using `OpenAI.chat_create`
    enable_rotating_clients = True, 

    # This will prioritize Azure over OpenAI when using `OpenAI.chat_create`
    prioritize = "azure",
)

# Display the current client
OpenAI.get_current_client_info(verbose = True)

# Rotate to the next client
# OpenAI.rotate_client(verbose = True)

###
# [Sync] create a completion with auto-rotation and auto-retry
###

result: ChatResponse = OpenAI.chat_create(
    model = "gpt-3.5-turbo-16k",
    messages = [
        {"role": "user", "content": "Translate the following English text to French: “Multiple models, each with different capabilities and price points. Prices are per 1,000 tokens. You can think of tokens as pieces of words, where 1,000 tokens is about 750 words. This paragraph is 35 tokens”"}
    ],
    auto_retry = True,

)

logger.info(f'Result Chat Message: {result.messages}')
logger.info(f'Result Usage: {result.usage}')
logger.info(f'Result Consumption: {result.consumption}')

###
# [Async] create a completion with auto-rotation and auto-retry
###

result: ChatResponse = asyncio.run(
    OpenAI.async_chat_create(
        model = "gpt-3.5-turbo-16k",
        messages = [
            {"role": "user", "content": "Translate the following English text to French: “Multiple models, each with different capabilities and price points. Prices are per 1,000 tokens. You can think of tokens as pieces of words, where 1,000 tokens is about 750 words. This paragraph is 35 tokens”"}
        ],
        auto_retry = True,
    )
)

```

### Function Calls

The latest version of the API allows for function calls to be made. This is currently only supported in `Chat` and requires api version: `2023-07-01-preview` for `azure`.

Function calls support using `pydantic` models to auto-generate the schemas

```python

import asyncio
from enum import Enum
from client_rotate import OpenAI
from async_openai.utils import logger
from pydantic import BaseModel, Field

class Unit(str, Enum):
    celsius = "celsius"
    fahrenheit = "fahrenheit"

class Weather(BaseModel):
    location: str = Field(..., description="The city and state, e.g. San Francisco, CA.")
    unit: Unit = Field(Unit.fahrenheit)

functions = [ 
  {
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": Weather,
  }
]

result: ChatResponse = OpenAI.chat_create(
    model = "gpt-3.5-turbo-16k",
    messages = [
        {"role": "user", "content": "What's the weather like in Boston today?"}
    ],
    functions = functions,
    auto_retry = True,
)

logger.info(f'Result Chat Message: {result.messages}')
logger.info(f'Result Chat Function: {result.function_results}')
logger.info(f'Result Usage: {result.usage}')
logger.info(f'Result Consumption: {result.consumption}')

"""
Result:

> Result Chat Message: [ChatMessage(content='', role='assistant', function_call=FunctionCall(name='get_current_weather', arguments={'location': 'Boston, MA'}), name=None)]
> Result Chat Function: [FunctionCall(name='get_current_weather', arguments={'location': 'Boston, MA'})]
> Result Usage: prompt_tokens=16 completion_tokens=19 total_tokens=35
> Result Consumption: 0.00012399999999999998
"""

```



---

### Dependencies

The aim of this library is to be as lightweight as possible. It is built on top of the following libraries:

- [aiohttpx](https://github.com/GrowthEngineAI/aiohttpx): Unified Async / Sync HTTP Client that wraps around `httpx`

    - [httpx](https://www.python-httpx.org/): Async / Sync HTTP Requests

    - [lazyops](https://github.com/trisongz/lazyops): Provides numerous utility functions for working with Async / Sync code and data structures

- [pydantic](https://pydantic-docs.helpmanual.io/): Type Support

- [file-io](https://github.com/trisongz/file-io): Async Cloud-based File Storage I/O

- [backoff](https://github.com/litl/backoff): Retries with Exponential Backoff


