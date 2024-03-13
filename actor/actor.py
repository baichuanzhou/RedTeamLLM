import os
from typing import Dict, List
import logging

from model import model_init

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseActorModel:
    """
    TODO: add docs
    BaseActorModel
    """

    def __init__(
            self,
            model_name: str,
            api_key: str = None,
            generation_kwargs: Dict = None,
            system_message: str = None
    ):
        self.model, self.generation_config = model_init(model_name, api_key, generation_kwargs)
        self.system_message = system_message

    def reset_history(self):
        self.model.history = []

    def get_history(self):
        return self.model.history
