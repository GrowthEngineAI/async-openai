# Changelogs

#### v0.0.40 (2023-10-18)

**Potentially Breaking Changes**

This version introduces full compatability with `pydantic v1/v2` where previous versions would only work with `pydantic v1`. Auto-detection and handling of deprecated methods of `pydantic` models are handled by `lazyops`, and require `lazyops >= 0.2.60`.

With `pydantic v2` support, there should be a slight performance increase in parsing `pydantic` objects, although the majority of the time is spent waiting for the API to respond.

Additionally, support is added for handling the response like a `dict` object, so you can access the response like `response['choices']` rather than `response.choices`.

#### v0.0.36 (2023-10-11)

**Additions**

- Added auto-parsing of `pydantic` objects in `function_call` parameters and return the same object schema in `chat_response.function_result_objects`.


#### v0.0.35 (2023-10-06)

**Additions**

- Added `auto_retry` option to `OpenAI` client, which will automatically retry failed requests.
- Added `RotatingClients` class which handles the rotation of multiple clients. This can be enabled by passing `rotating_clients=True` to the `OpenAI` client while configuring.
- Added `OpenAI.chat_create` and `OpenAI.async_chat_create` methods which automatically handles rotating clients and retrying failed requests.
- Added `azure_model_mapping` which allows automatically mapping of Azure models to OpenAI models when passing `openai` models as a parameter, it will automatically convert it to the Azure model. This is only done in `chat` implementation.

**Fixes**

- Fixed `api_version` Configuration handling.
- Fixed parsing of `function_call` in streaming implementation.



#### v0.0.34 (2023-10-06)

**Changes** 

- Updated default `api_version` to `2023-07-01-preview`
- Added `__getitem__` attributes to completion and chat objects, allowing them to act like `dict` objects.
- Added `functions` and `function_call` to `Chat` completion routes.
  - `function.properties` can pass through a `pydantic` object which will convert it automatically to a `dict` json schema.
- Added `function_call` attribute in `ChatMessage` objects, allowing for easy access to the function call.
- Streaming is not supported for `functions` at this time.

#### v0.0.33 (2023-08-24)

**Changes**

- Updated auto-configuring `httpx`'s logger to be disabled if `debug_enabled` is set to `False`.


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


  