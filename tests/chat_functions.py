import asyncio
from enum import Enum
from client_rotate import OpenAI
from async_openai.utils import logger
from pydantic import BaseModel, Field


class Unit(str, Enum):
    celsius = "celsius"
    fahrenheit = "fahrenheit"

class Weather(BaseModel):
    location: str = Field(..., description="The city and state, e.g. San Francisco, CA.")
    unit: Unit = Field(Unit.fahrenheit)

functions = [ 
  {
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": Weather,
  }
]

async def run_test():

    model = "gpt-3.5-turbo-instruct"

    result = await OpenAI.chat.async_create(
        model = model,
        messages = [
            {"role": "user", "content": "What's the weather like in Boston today?"}
        ],
        functions = functions,
    )
    logger.info(f'Result Model: {result}')
    logger.info(f'Result Type: {type(result)}')

    logger.info(f'Result Text: {result.text}')
    logger.info(f'Result Chat Message: {result.messages}')
    logger.info(f'Result Chat Function: {result.function_results}')
    
    logger.info(f'Result Usage: {result.usage}')
    logger.info(f'Result Consumption: {result.consumption}')
    
    

    result = OpenAI.chat.create(
        model = model,
        messages = [
            {"role": "user", "content": "What's the weather like in Boston today?"}
        ],
        functions = functions,
    )

    logger.info(f'Result Model: {result}')
    logger.info(f'Result Type: {type(result)}')

    logger.info(f'Result Text: {result.text}')
    logger.info(f'Result Chat Message: {result.messages}')
    logger.info(f'Result Chat Function: {result.function_results}')
    
    logger.info(f'Result Usage: {result.usage}')
    


asyncio.run(run_test())    

