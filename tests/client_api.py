import asyncio
from async_openai import OpenAI, settings


org_id = 'org-....'
api_key = 'sk-...'

settings.configure(
    api_key = api_key,
    organization = org_id
)

async def run_test():
    result = await OpenAI.completions.async_create(
        prompt = 'say this is a test',
        max_tokens = 4,
        stream = True
    )
    print(result)

    result = OpenAI.completions.create(
        prompt = 'say this is a test',
        max_tokens = 4,
        stream = True
    )
    print(result)


asyncio.run(run_test())    


