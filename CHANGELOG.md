# Changelogs

**Changes**

#### v0.0.32 (2023-08-23)

**Changes**

- Updated `headers` behavior and parameter, allowing it to be passed to each API call.
- Updated `auth` behavior, which now utilizes `httpx.Auth` rather than injecting into the header directly.
- Added `custom_headers` configuration that can be passed to the `OpenAI` client during initialization.
- Added customization of `connection_pool`, controlling the number of concurrent connections to the API.

- Reworked `streaming` implementations, which previously didn't properly work.
- Added `parse_stream` parameter (default: true) which defers parsing of the stream util it is called with `result.stream` or `result.astream`, rather than parsing the stream as it is received.


#### v0.0.31 (2023-08-11)


**Changes**

- Updated some behaviors of the `OpenAI` Client.
  * allow for customization of retry behavior or completely disabling it.

- Routes now take the `is_azure` parameter during init rather than using `@property` to determine the route.
- The `_send` method is better optimized for retry behaviors.

**Fixes**

- Resolved `model` endpoint.
- Resolved handling of `Azure` models.



---

#### v0.0.30 (2023-08-10)

_Potentially breaking changes in this version_

**Changes**

- Refactored the architecture of the `OpenAI` Client to accomodate multi-client initialization. i.e. `OpenAI` can now be initialized with multiple API keys and will automatically rotate between them, as well as switch back and forth between Azure and OpenAI.

- Settings are initialized after first call, rather than globally.

- Routes, Clients are configured after first call, rather than during initialization.


**Fixes**

- Resolved `embedding` endpoints.

**Updates**

- Changed default `api-version` to `2023-03-15-preview`

---

#### v0.0.22 (2023-06-14)
  - Update pricing to reflect OpenAI's new pricing model
    - `gpt-3.5-turbo`
    - `text-embedding-ada-002`
  - Bugfix for consumption and usage validation in `chat` models
  - Added support for `gpt-3.5-turbo-16k`
  - Modified handling of `gpt-3.5-turbo`'s consumption pricing to reflect `prompt` and `completion` usage
  - Modified default `Embedding` model to be `ada`

---
#### 0.0.17 (2023-04-12)
  - Add better support for chatgpt models and `gpt-4`
  - Better validation `max_tokens`

---
#### 0.0.11 (2023-03-07)
  - Added support for GPT-3.5 Turbo through `async_openai.OpenAI.chat`
  - Refactored `async_openai.OpenAI` to utilize a `metaclass` rather than initalizing directly

#### 0.0.7 (2023-02-02)
  - Refactor `async_openai.types.options.OpenAIModel` to handle more robust parsing of model names.

#### 0.0.3 (2022-12-21)
  - Fix proper charge for `babbage` and `ada` models.


  