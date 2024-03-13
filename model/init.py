from typing import Dict, Tuple

from .openai_model import OpenaiGPT, OpenaiGenerationConfig
from .base_model import LanguageModel, BaseGenerationConfig


def model_init(
        model_name: str,
        api_key: str = None,
        generation_kwargs: Dict = None
) -> Tuple[LanguageModel, BaseGenerationConfig]:
    """
    Takes the model name as input and return the corresponding language model and configuration
    Input:
        model_name: str, the name of desired model
    Return:
        model: an instantiation of LanguageModel
        generation_config: an instantiation of BaseGenerationConfig
    """
    if generation_kwargs is None:
        generation_kwargs = dict()
    if "gpt" in model_name:
        return OpenaiGPT(model_name, api_key), OpenaiGenerationConfig(**generation_kwargs)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented yet")