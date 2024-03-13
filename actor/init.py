from typing import Dict, Tuple

from .actor import BaseActorModel
from .blueteam import BlueTeamModel
from .redteam import RedTeamModel
from .judge import JudgeModel

ACTOR_CHOICES = {"red": RedTeamModel, "blue": BlueTeamModel, "judge": JudgeModel}


def actor_init(
        actor_type: str,
        model_name: str,
        api_key: str = None,
        generation_kwargs: Dict = None,
        system_message: str = None
) -> BaseActorModel:
    if actor_type.lower() in ACTOR_CHOICES.keys():
        return ACTOR_CHOICES[actor_type.lower()](
            model_name=model_name,
            api_key=api_key,
            generation_kwargs=generation_kwargs,
            system_message=system_message
        )
    else:
        raise ValueError(f"Actor type {actor_type} not supported. Available team types {ACTOR_CHOICES}")