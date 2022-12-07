# async-openai
 Async Python client library for the OpenAI API based on [Documented Specs](https://beta.openai.com/docs/api-reference/making-requests)

## Features

- [x] Asyncio based with Sync and Async Support

- [ ] Supports all API endpoints

    - [x] [Completions](https://beta.openai.com/docs/api-reference/completions)
    
    - [x] [Edits](https://beta.openai.com/docs/api-reference/edits)
    
    - [x] [Embeddings](https://beta.openai.com/docs/api-reference/embeddings)

- [x] Handles Retries automatically

---
 
## Installation

```bash
pip install async-openai
```

### Quick Usage

```python

import asyncio
from async_openai import OpenAI, settings

# Environment variables should pick up the defaults
# however, you can also set them explicitly.

settings.configure(
    api_key = "sk-XXXX",
    organization = "org-XXXX",
)

# [Sync] create a completion
# Results return a CompletionResult object
result = OpenAI.completions.create(
    prompt = 'say this is a test',
    max_tokens = 4,
    stream = True
)

# print the completion text
# which are concatenated together from the result['choices'][n]['text']

print(result.completion_text)

# print the number of choices returned
print(len(result))

# [Async] create a completion
# All async methods are generally prefixed with `async_`

result = asyncio.run(
    OpenAI.completions.async_create(
        prompt = 'say this is a test',
        max_tokens = 4,
        stream = True
    )
)

```

## Alternative Usage

Following a similar pattern to the original openai python project, you can additionally use the methods as a drop-in replacement for most functions

```python

import asyncio
import async_openapi as openai

# Environment variables should pick up the defaults
# however, you can also set them explicitly.

openai.settings.configure(
    api_key = "sk-XXXX",
    organization = "org-XXXX",
)

# [Sync] create a completion
# Results return a CompletionResult object
# Note, the only difference in this method is that the `Completion` is capitalized to match the OpenAI API

result = openai.Completions.create(
    prompt = 'say this is a test',
    max_tokens = 4,
    stream = True
)

# print the completion text
# which are concatenated together from the result['choices'][n]['text']

print(result.completion_text)

# print the number of choices returned
print(len(result))

# [Async] create a completion
result = asyncio.run(
    openai.Completions.async_create(
        prompt = 'say this is a test',
        max_tokens = 4,
        stream = True
    )
)



```

---

### Dependencies

The aim of this library is to be as lightweight as possible. It is built on top of the following libraries:

- [httpx](https://www.python-httpx.org/): Async / Sync HTTP Requests

- [pydantic](https://pydantic-docs.helpmanual.io/): Type Support

- [loguru](https://github.com/Delgan/loguru): Logging

- [file-io](https://github.com/trisongz/file-io): Async Cloud-based File Storage I/O

- [backoff](https://github.com/litl/backoff): Retries with Exponential Backoff


