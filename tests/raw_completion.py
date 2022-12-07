
import asyncio
import httpx

from async_openai.schemas.types.completions import CompletionResult

org_id = 'org-...'
api_key = 'sk-....'

client = httpx.AsyncClient(
    base_url='https://api.openai.com/v1',
    headers={
        # 'api-key': api_key,
        'Authorization': f'Bearer {api_key}',
        'OpenAI-Organization': org_id,
    }
)

async def run_test():
    async with client:
        response = await client.post(
            '/completions',
            json = {
                'model': 'text-davinci-003',
                'prompt': 'say this is a test',
                'max_tokens': 4,
                'stream': True,
            }
        )
        print(response.headers)
        obj = CompletionResult()
        obj.parse_result(
            raw_response = response
        )
        print(obj)
        print(len(obj))


asyncio.run(run_test())    





