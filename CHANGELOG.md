# Changelogs
- 0.0.17 (2023-04-12)
  - Add better support for chatgpt models and `gpt-4`
  - Better validation `max_tokens`

- 0.0.11 (2023-03-07)
  - Added support for GPT-3.5 Turbo through `async_openai.OpenAI.chat`
  - Refactored `async_openai.OpenAI` to utilize a `metaclass` rather than initalizing directly

- 0.0.7 (2023-02-02)
  - Refactor `async_openai.types.options.OpenAIModel` to handle more robust parsing of model names.

- 0.0.3 (2022-12-21)
  - Fix proper charge for `babbage` and `ada` models.

  
