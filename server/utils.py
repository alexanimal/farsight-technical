import logging
import os
from typing import Dict, List, Optional
from openai import AsyncOpenAI, APIError, RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import json
from json import JSONDecodeError
from dotenv import load_dotenv
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#####################################################
#####################################################
################# Pinecone Utils ####################
#####################################################
#####################################################


# Link to Pinecone Documentation:
# https://docs.pinecone.io/guides/get-started/overview


#####################################################
#####################################################
#################### RDS Utils ######################
#####################################################
#####################################################


# Setting up an RDS connection is the same as any other postgres connection.


#####################################################
#####################################################
#################### GPT Utils ######################
#####################################################
#####################################################


# Initialize the AsyncOpenAI client with the API key
load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((RateLimitError, APIError)),
)
async def gpt_chat(
    model: str = "gpt-4.1", messages: List[Dict[str, str]] = [], json_mode: bool = False
) -> Optional[Dict | str]:
    """
    Makes an asynchronous request to OpenAI's Public API.

    @author: Noah Faro

    @params:
        model (str): The GPT model to use for completion. Options are "gpt-4.1" and "gpt-4.1-mini"
        messages (List[Dict[str, str]]): The messages to send to the API. Defaults to an empty list.
                example of messages: [
                                        {"role": "system", "content": "You are a helpful assistant."},
                                        {"role": "user", "content": "Say hello world!"}
                                      ]
        json_mode (bool): Whether to return a JSON object or a string. Defaults to False.

    @returns:
        str | Dict: response from the API

    @onError:
        If no response from OpenAI after 3 attempts, throws RateLimitError, APIError, or an Exception.
                If an unknown error occurs, throws an Exception and skips retries.
    """
    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"} if json_mode else None,
        )

        content = completion.choices[0].message.content

        # If json_mode is true, try to parse the content as a JSON object.
        # If it fails, we should not retry as this should not be possible via OpenAI documentation.
        if json_mode:
            try:
                content = json.loads(content)
            except JSONDecodeError as e:
                raise e

        return content

    # Exception options - RateLimitError, APIError will be retried, other exceptions will be raised.
    # After third retry, if any error persists, it gets raised.
    except RateLimitError as e:
        logger.warning(f"GPT - Rate Limit Exceeded: {str(e)}")
        raise e
    except APIError as e:
        logger.error(f"GPT - API Error Occurred: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"GPT - Unexpected Error Occurred: {str(e)}")
        raise Exception(f"Unexpected GPT Error: {str(e)}")


# Example of running the above util
asyncio.run(
    gpt_chat(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": 'Say hello world! Please respond with valid JSON in with "message": your response.',
            },
        ],
        json_mode=True,
    )
)
