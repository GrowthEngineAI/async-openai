import asyncio
from client import OpenAI
from async_openai.utils import logger

async def run_test():
    result = await OpenAI.completions.async_create(
        prompt = 'say this is a test',
        max_tokens = 4,
        stream = False
    )
    logger.info(f'Result Model: {result}')
    logger.info(f'Result Type: {type(result)}')

    logger.info(f'Result Text: {result.text}')
    logger.info(f'Result Usage: {result.usage}')
    logger.info(f'Result Consumption: {result.consumption}')
    
    

    result = OpenAI.completions.create(
        prompt = 'say this is a test',
        max_tokens = 4,
        stream = True
    )
    

    logger.info(f'Result Model: {result}')
    logger.info(f'Result Type: {type(result)}')

    logger.info(f'Result Text: {result.text}')
    logger.info(f'Result Usage: {result.usage}')
    


asyncio.run(run_test())    

