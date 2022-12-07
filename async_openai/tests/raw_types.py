
import asyncio
import httpx

from async_openai.schemas.types import CompletionRequest, CompletionResult
from async_openai.schemas.types import ModelRequest, ModelResult


org_id = 'org-...'
api_key = 'sk-...'

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
        req = CompletionRequest(
            prompt = 'say this is a test',
            max_tokens = 4,
            stream = True,
        )
        response = await client.request(
            **await req.create_endpoint.async_get_params(),
        )
        resp = CompletionResult(
            _raw_request = req,
            _raw_response = response,
        )
        resp.parse_result()
        print(resp)
        print(len(resp))

        req = ModelRequest()
        response = await client.request(
            **await req.list_models_endpoint.async_get_params(),
        )
        resp = ModelResult(
            _raw_request = req,
            _raw_response = response,
        )
        resp.parse_result()
        print(resp)



asyncio.run(run_test())    


