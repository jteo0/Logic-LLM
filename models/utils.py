import os
import asyncio
from typing import Any
from openai import OpenAI

# Stub these out — not used by Groq class but imported elsewhere
def completions_with_backoff(**kwargs):
    raise NotImplementedError("Use OpenAIModel.generate() instead")

def chat_completions_with_backoff(**kwargs):
    raise NotImplementedError("Use OpenAIModel.generate() instead")

async def dispatch_openai_chat_requests(
    messages_list: list[list[dict[str, Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: list[str]
) -> list[str]:
    raise NotImplementedError("Async dispatch not supported for Groq")

async def dispatch_openai_prompt_requests(
    messages_list: list[list[dict[str, Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: list[str]
) -> list[str]:
    raise NotImplementedError("Async dispatch not supported for Groq")


class OpenAIModel:
    """
    Drop-in replacement routing all generation through Groq's API.

    Groq is OpenAI-compatible so only the base_url and api_key differ.
    Recommended models:
        llama-3.3-70b-versatile   — best quality, generous free tier
        llama-3.1-8b-instant      — faster, lower limits
        mixtral-8x7b-32768        — good for longer context prompts

    Get a free key at: https://console.groq.com
    """

    def __init__(self, API_KEY, model_name, stop_words, max_new_tokens):
        self.client = OpenAI(
            api_key=API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words

    def generate(self, input_string, temperature=0.0):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": input_string}],
            max_tokens=self.max_new_tokens,
            temperature=temperature,
            stop=self.stop_words if self.stop_words else None
        )
        return response.choices[0].message.content.strip()

    def batch_generate(self, messages_list, temperature=0.0):
        return [self.generate(m, temperature) for m in messages_list]

    def chat_generate(self, input_string, temperature=0.0):
        return self.generate(input_string, temperature)

    def prompt_generate(self, input_string, temperature=0.0):
        return self.generate(input_string, temperature)

    def batch_chat_generate(self, messages_list, temperature=0.0):
        return self.batch_generate(messages_list, temperature)

    def batch_prompt_generate(self, prompt_list, temperature=0.0):
        return self.batch_generate(prompt_list, temperature)