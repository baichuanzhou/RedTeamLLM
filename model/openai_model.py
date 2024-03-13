from dataclasses import dataclass, field, asdict
import os
import random
from typing import List, Dict
import logging

import openai
from .base_model import LanguageModel, BaseGenerationConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


MODEL_NAMES = [
    "gpt-4-0125-preview",
    "gpt-4-turbo-preview",
    "gpt-4-1106-preview",
    "gpt-4-vision-preview",
    "gpt-4-1106-vision-preview",
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0613",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-instruct",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k-0613"
]


@dataclass
class OpenaiGenerationConfig(BaseGenerationConfig):
    tool_choice: str = field(
        default=None, metadata={"help": "controls the model's function calls (none/auto/function)"}
    )
    response_format: str = field(
        default=None, metadata={"help": "specifies the output format, e.g., JSON mode"}
    )
    n: int = field(
        default=1, metadata={"help": "generates a specified number of chat completion choices for each input"}
    )

    top_k: int = field(
        default=None, repr=False,
    )
    top_logprobs: int = field(default=None, )

    max_new_tokens: int = field(
        default=None, repr=False,
    )
    max_tokens: int = field(default=None, )

    def __post_init__(self):
        if self.top_k is not None and self.top_logprobs is None:
            self.top_logprobs = self.top_k
        delattr(self, "top_k")
        if self.max_new_tokens is not None and self.max_tokens is None:
            self.max_tokens = self.max_new_tokens
        delattr(self, "max_new_tokens")

    def to_dict(self) -> Dict:
        return self.__dict__


class OpenaiGPT(LanguageModel):
    """
    TODO: add docs
    """
    model_type = "openai"

    def __init__(self, model_name: str, api_key: str = None):
        super().__init__(model_name, self.model_type)
        if model_name not in MODEL_NAMES:
            raise ValueError(f"Model name {model_name} is not a valid GPT version."
                             f"Available models: {MODEL_NAMES}")
        self.client = openai.OpenAI(api_key=os.environ.get(
            "OPENAI_API_KEY", api_key)
        )
        self.total_generated_tokens = 0

    def chat(
            self,
            system_prompt: str = None,
            messages: List[Dict] = None,
            generation_config: OpenaiGenerationConfig = None,
    ) -> List[Dict]:
        """
        chat method should receive multiple messages, and it should be combined with history
        TODO: add docs
        Input:
            system_prompt: str, system message for GPT model
            messages: A list of messages. Example format for one message: {"role": EXAMPLE_ROLE,
                                                                           "content": ACTUAL_MESSAGE}
            generation_config:
        """
        if system_prompt is not None and len(self.history) == 0:
            messages = [{"role": "system", "content": system_prompt}] + messages

        elif system_prompt is not None and len(self.history):
            logger.warning(f"Chat history is not None, ignore system message: {system_prompt}")

        if generation_config is None:
            generation_config = OpenaiGenerationConfig()

        self.history.extend(messages)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.history,
            **generation_config.to_dict()
        )
        # get the generated content
        content = [
            {"role": choice.message.role, "content": choice.message.content}
            for choice in response.choices
        ]
        # add content to history
        # FIXME: if n is set to more than 1, randomly choose one response
        self.history.extend(content)
        # compute total tokens
        self.total_generated_tokens += response.usage.total_tokens
        return content
