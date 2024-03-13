from typing import List, Dict
from dataclasses import dataclass, field
from abc import abstractmethod


@dataclass
class BaseGenerationConfig:
    top_p: float = field(default=1, metadata={"help": "probability for nucleus sampling"})
    top_k: int = field(default=None, metadata={"help": "number of tokens to return at each position during sampling"})
    temperature: float = field(default=0, metadata={"help": "temperature for generation"})
    max_new_tokens: int = field(default=200, metadata={"help": "sets the maximum number of generated tokens"})
    seed: int = field(default=42, metadata={"help": "deterministic sampling with a specified seed"})
    logprobs: bool = field(default=None, metadata={"help": "returns log probabilities of output tokens if true"})


class LanguageModel:
    """
    Base Model for language model
    """
    def __init__(self, model_name: str, model_type: str):
        self.model_name = model_name
        self.model_type = model_type
        self.history = []

    def chat(
            self,
            system_prompt: str = None,
            messages: List[Dict[str, str]] = None,
            generation_config: BaseGenerationConfig = None,
    ):
        raise NotImplementedError("Language model needs to implement its own chat method")


