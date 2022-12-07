

import asyncio
from async_openai import settings
from async_openai.schemas.completion import CompletionSchema


org_id = 'org-...'
api_key = 'sk-...'

settings.api_key = api_key
settings.organization = org_id


async def run_test():
    schema = CompletionSchema()
    result = await schema.async_create(
        prompt = 'say this is a test',
        max_tokens = 4,
        stream = True
    )
    print(result)


asyncio.run(run_test())    


