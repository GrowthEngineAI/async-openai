from async_openai import OpenAI

org_id = 'org-...'
api_key = 'sk-...'

OpenAI.configure(
    api_key = api_key,
    organization = org_id,
    debug_enabled = True,
)