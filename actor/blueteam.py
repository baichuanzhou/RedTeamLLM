from typing import Dict, List
import logging

from .actor import BaseActorModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BlueTeamModel(BaseActorModel):
    """
    TODO: add docs
    BlueTeamModel
    """
    team_type = "blueteam"

    def defend(
            self,
            redteam_prompt: str
    ) -> List[Dict[str, str]]:
        defender_message = [{"role": "user", "content": redteam_prompt}]
        system_prompt = self.system_message if len(self.get_history()) == 0 else None
        response = self.model.chat(
            system_prompt=system_prompt,
            messages=defender_message,
            generation_config=self.generation_config
        )
        logger.info(f"Blue Team: {response[0]['content']}")
        return response[0]['content']
