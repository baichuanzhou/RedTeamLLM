from typing import Dict, List
import logging

from .actor import BaseActorModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RedTeamModel(BaseActorModel):
    """
    TODO: add docs
    RedTeamModel
    """
    team_type = "redteam"

    def attack(
            self,
            blueteam_response: str
    ) -> List[Dict[str, str]]:
        blueteam_message = [{"role": "user", "content": blueteam_response}]
        system_prompt = self.system_message if len(self.get_history()) == 0 else None

        response = self.model.chat(
            system_prompt=system_prompt,
            messages=blueteam_message,
            generation_config=self.generation_config
        )

        logger.info(f"Red Team: {response[0]['content']}")
        return response[0]['content']
