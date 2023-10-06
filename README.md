# async-openai
 Unofficial Async Python client library for the [OpenAI](https://openai.com) API based on [Documented Specs](https://beta.openai.com/docs/api-reference/making-requests)

 **Latest Version**: [![PyPI version](https://badge.fury.io/py/async-openai.svg)](https://badge.fury.io/py/async-openai)

 **[Official Client](https://github.com/openai/openai-python)**

## Features

- [x] Asyncio based with Sync and Async Support with `httpx`

- [ ] Supports all API endpoints

    - [x] `Completions`: [Docs](https://beta.openai.com/docs/api-reference/completions)

      - [x] Supports Streaming

    - [x] `Chat`: [Docs](https://beta.openai.com/docs/api-reference/chat)

      - [x] Supports Streaming

      - [x] Supports `Functions`
    
    - [x] `Edits`: [Docs](https://beta.openai.com/docs/api-reference/edits)
    
    - [x] `Embeddings`: [Docs](https://beta.openai.com/docs/api-reference/embeddings)

    - [x] `Models`: [Docs](https://beta.openai.com/docs/api-reference/models)

- [x] Strongly typed validation of requests and responses with `Pydantic` Models with transparent 
    access to the raw response and object-based results.

- [x] Handles Retries automatically through `backoff` and custom retry logic.
   
   - [x] Handles `rate_limit` errors and retries automatically. (when passing `auto_retry = True`)

- [x] Supports Multiple Clients and Auto-Rotation of Clients

- [x] Supports `Azure` API

- [x] Supports Local and Remote Cloud Object Storage File Handling Asyncronously through `file-io`

    - [x] Supports `S3`: `s3://bucket/path/to/file.txt`
    
    - [x] Supports `GCS`: `gs://bucket/path/to/file.txt`

    - [x] Supports `Minio`: `minio://bucket/path/to/file.txt`

- [x] Supports `limited` cost tracking for `Completions` and `Edits` requests (when stream is not enabled)

- [x] Parallelization Safe with ThreadPools or any `asyncio` compatible event loop. Can handle 100s of requests per second. (If you don't run into rate limits)


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
# however, you can also set them explicitly. See below for more details.

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

### Configuration and Environment Variables

The following environment variables can be used to configure the client.

```

OpenAI Configuration

url: The OpenAI API URL                                     | Env: [`OPENAI_API_URL`]
scheme: The OpenAI API Scheme                               | Env: [`OPENAI_API_SCHEME`]
host: The OpenAI API Host                                   | Env: [`OPENAI_API_HOST`]
port: The OpenAI API Port                                   | Env: [`OPENAI_API_PORT`]
api_base: The OpenAI API Base                               | Env: [`OPENAI_API_BASE`]
api_key: The OpenAI API Key                                 | Env: [`OPENAI_API_KEY`]
api_path: The OpenAI API Path                               | Env: [`OPENAI_API_PATH`]
api_type: The OpenAI API Type                               | Env: [`OPENAI_API_TYPE`]
api_version: The OpenAI API Version                         | Env: [`OPENAI_API_VERSION`]
api_key_path: The API Key Path                              | Env: [`OPENAI_API_KEY_PATH`]
organization: Organization                                  | Env: [`OPENAI_ORGANIZATION`]
proxies: The OpenAI Proxies                                 | Env: [`OPENAI_PROXIES`]
timeout: Timeout in Seconds                                 | Env: [`OPENAI_TIMEOUT`]
max_retries: The OpenAI Max Retries                         | Env: [`OPENAI_MAX_RETRIES`]
ignore_errors: Ignore Errors                                | Env: [`OPENAI_IGNORE_ERRORS`]
disable_retries: Disable Retries                            | Env: [`OPENAI_DISABLE_RETRIES`]
max_connections: Max Connections                            | Env: [`OPENAI_MAX_CONNECTIONS`]
max_keepalive_connections: Max Keepalive Connections        | Env: [`OPENAI_MAX_KEEPALIVE_CONNECTIONS`]
keepalive_expiry: Keepalive Expiry                          | Env: [`OPENAI_KEEPALIVE_EXPIRY`]
custom_headers: Custom Headers                              | Env: [`OPENAI_CUSTOM_HEADERS`]

Azure Configuration

azure_url: The OpenAI API URL                               | Env: [`AZURE_OPENAI_API_URL`]
azure_scheme: The OpenAI API Scheme                         | Env: [`AZURE_OPENAI_API_SCHEME`]
azure_host: The OpenAI API Host                             | Env: [`AZURE_OPENAI_API_HOST`]
azure_port: The OpenAI API Port                             | Env: [`AZURE_OPENAI_API_PORT`]
azure_api_key: The OpenAI API Key                           | Env: [`AZURE_OPENAI_API_KEY`]
azure_api_base: The OpenAI API Base                         | Env: [`AZURE_OPENAI_API_BASE`]
azure_api_path: The OpenAI API Path                         | Env: [`AZURE_OPENAI_API_PATH`]
azure_api_type: The OpenAI API Type                         | Env: [`AZURE_OPENAI_API_TYPE`]
azure_api_version: The OpenAI API Version                   | Env: [`AZURE_OPENAI_API_VERSION`]
azure_api_key_path: The API Key Path                        | Env: [`AZURE_OPENAI_API_KEY_PATH`]
azure_organization: Organization                            | Env: [`AZURE_OPENAI_ORGANIZATION`]
azure_proxies: The OpenAI Proxies                           | Env: [`AZURE_OPENAI_PROXIES`]
azure_timeout: Timeout in Seconds                           | Env: [`AZURE_OPENAI_TIMEOUT`]
azure_max_retries: The OpenAI Max Retries                   | Env: [`AZURE_OPENAI_MAX_RETRIES`]

```


### Initialize Clients Manually, and working with multiple clients

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

# You can select the different clients by name or index
result = OpenAI['az'].completions.create(
    prompt = 'say this is a test',
    max_tokens = 4,
    stream = True
)

# Use the default client (openai)
result = OpenAI['default'].completions.create(
    prompt = 'say this is a test',
    max_tokens = 4,
    stream = True
)

# Will use the `default` client since it was initialized first
result = OpenAI[0].completions.create(
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

### Configure Azure Model Mapping

Your azure models may be named differently than the default mapping. By configuring the mapping, you can automatically map the models to the correct azure model (when using openai model names).

```python

from async_openai import OpenAI

"""
Default Azure Model Mapping
{
    'gpt-3.5-turbo': 'gpt-35-turbo',
    'gpt-3.5-turbo-16k': 'gpt-35-turbo-16k',
    'gpt-3.5-turbo-instruct': 'gpt-35-turbo-instruct',
    'gpt-3.5-turbo-0301': 'gpt-35-turbo-0301',
    'gpt-3.5-turbo-0613': 'gpt-35-turbo-0613',
}
"""

AzureModelMapping = {
    'gpt-3.5-turbo': 'azure-gpt-35-turbo',
    'gpt-3.5-turbo-16k': 'azure-gpt-35-turbo-16k',
    'gpt-3.5-turbo-instruct': 'azure-gpt-35-turbo-instruct',
    'gpt-3.5-turbo-0301': 'azure-gpt-35-turbo-0301',
    'gpt-3.5-turbo-0613': 'azure-gpt-35-turbo-0613',
}

OpenAI.configure(
    api_key = "sk-XXXX",
    organization = "org-XXXX",
    debug_enabled = False,

    # Azure Configuration
    azure_api_base = 'https://....openai.azure.com/',
    azure_api_version = '2023-07-01-preview',
    azure_api_key = '....',
    azure_model_mapping = AzureModelMapping,
)

# This will now use the azure endpoint as the default client
OpenAI.init_api_client('az', set_as_default = True, debug_enabled = True)

# This will automatically map "gpt-3.5-turbo-16k" -> "azure-gpt-35-turbo-16k"
result: ChatResponse = OpenAI.chat.create(
    model = "gpt-3.5-turbo-16k",
    messages = [
        {"role": "user", "content": "Translate the following English text to French: “Multiple models, each with different capabilities and price points. Prices are per 1,000 tokens. You can think of tokens as pieces of words, where 1,000 tokens is about 750 words. This paragraph is 35 tokens”"}
    ],
    auto_retry = True,
)


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


