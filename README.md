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


---

### Dependencies

The aim of this library is to be as lightweight as possible. It is built on top of the following libraries:

- [aiohttpx](https://github.com/GrowthEngineAI/aiohttpx): Unified Async / Sync HTTP Client that wraps around `httpx`

    - [httpx](https://www.python-httpx.org/): Async / Sync HTTP Requests

    - [lazyops](https://github.com/trisongz/lazyops): Provides numerous utility functions for working with Async / Sync code and data structures

- [pydantic](https://pydantic-docs.helpmanual.io/): Type Support

- [file-io](https://github.com/trisongz/file-io): Async Cloud-based File Storage I/O

- [backoff](https://github.com/litl/backoff): Retries with Exponential Backoff


