import logging
import os
import time

import openai
import tiktoken
from pydantic import BaseModel


class OpenAIChatModelConfig(BaseModel):
    # OpenAI's model name
    chat_model: str = "gpt-3.5-turbo"

    max_tokens: int = 4096

    generation_length_tokens: int = 256


class OpenAIChatModel:
    """A wrapper around OpenAI's Chat completion API.

    You need to set your own OPENAI_API_KEY token and OPENAI_ORGANIZATION in your
    environment variables.

    We use the gpt-3.5-turbo model by default.
    https://platform.openai.com/docs/guides/gpt/chat-completions-api
    """

    def __init__(self, config: OpenAIChatModelConfig) -> None:

        self.openai_client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            organization=os.getenv("OPENAI_ORGANIZATION"),
        )
        self.chat_model_name = config.chat_model
        self.text_encoder = tiktoken.encoding_for_model(self.chat_model_name)
        self.max_tokens = config.max_tokens

    def generate(self, text: str) -> str:
        """Simple wrapper around the ChatCompletion API.

        Note: This restricts the answer to 256 tokes only. We need to evaluate
        each setting once we have training/eval datasets.

        Args:
            query_prompt: The prompt to use for generating an answer.

        Returns:
            The generated answer.
        """

        # Encode the text
        tokens = self.text_encoder.encode(text)
        # max tokens also include generation length ->
        # truncate prompt to have enough tokens for generation
        # extra 10 to account for some more tokens added as we create messages
        logging.info("OpenAI LLM: generate called with %s tokens", len(tokens))
        tokens = tokens[: self.max_tokens - self.generation_length_tokens - 10]
        prompt = self.text_encoder.decode(tokens)
        messages = [{"role": "user", "content": prompt}]
        # TODO: Make the model parameters (e.g. temperature) configurable.
        generate_begin = time.time()
        chat_completion = self.openai_client.chat.completions.create(
            model=self.chat_model_name,
            messages=messages,
            temperature=0,
            max_tokens=self.generation_length_tokens,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        output_num_tokens = self.text_encoder.encode(
            chat_completion.choices[0].message.content.strip()
        )
        generate_end = time.time()

        num_tokens_generated = len(output_num_tokens)

        logging.info(
            "OpenAI LLM generate output token count %s", len(output_num_tokens)
        )
        logging.info(
            "Total time taken is: %s", int((generate_end - generate_begin) / 60)
        )

        return chat_completion.choices[0].message.content.strip()
