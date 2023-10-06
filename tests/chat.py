import asyncio
from client import OpenAI
from async_openai.utils import logger

async def run_test():

    model = "gpt-3.5-turbo-16k"

    result = await OpenAI.chat.async_create(
        model = model,
        messages = [
            {"role": "user", "content": "Translate the following English text to French: “Multiple models, each with different capabilities and price points. Prices are per 1,000 tokens. You can think of tokens as pieces of words, where 1,000 tokens is about 750 words. This paragraph is 35 tokens”"}
        ],
    )
    logger.info(f'Result Model: {result}')
    logger.info(f'Result Type: {type(result)}')

    logger.info(f'Result Text: {result.text}')
    logger.info(f'Result Chat Message: {result.messages}')
    
    logger.info(f'Result Usage: {result.usage}')
    logger.info(f'Result Consumption: {result.consumption}')
    
    

    result = OpenAI.chat.create(
        messages = [
            {"role": "user", "content": "Translate the following English text to French: “Multiple models, each with different capabilities and price points. Prices are per 1,000 tokens. You can think of tokens as pieces of words, where 1,000 tokens is about 750 words. This paragraph is 35 tokens”"}
        ],
    )

    logger.info(f'Result Model: {result}')
    logger.info(f'Result Type: {type(result)}')

    logger.info(f'Result Text: {result.text}')
    logger.info(f'Result Chat Message: {result.messages}')
    
    logger.info(f'Result Usage: {result.usage}')
    


asyncio.run(run_test())    

