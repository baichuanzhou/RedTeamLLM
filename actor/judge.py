from typing import Dict, List
import logging

from .actor import BaseActorModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class JudgeModel(BaseActorModel):
    """
    TODO: add docs
    RedTeamModel
    """
    team_type = "judge"

    @staticmethod
    def create_judge_prompt(
            redteam_response: str,
            blueteam_response: str
    ) -> str:
        return f"[PROMPT]:{redteam_response}\n[ASSISTANT'S RESPONSE]: {blueteam_response}"

    def eval(
            self,
            redteam_response: str,
            blueteam_response: str
    ) -> List[Dict[str, str]]:
        actor_message = [{"role": "user", "content": self.create_judge_prompt(redteam_response, blueteam_response)}]
        system_prompt = self.system_message if len(self.get_history()) == 0 else None
        response = self.model.chat(
            system_prompt=system_prompt,
            messages=actor_message,
            generation_config=self.generation_config
        )
        logger.info(f"Judge: {response[0]['content']}")
        return response[0]['content']
