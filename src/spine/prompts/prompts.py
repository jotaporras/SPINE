from typing import List

from spine.prompts.base import BASE_SYSTEM_INSTRUCTIONS
from spine.prompts.examples import (
    EXAMPLE_1,
    EXAMPLE_2,
    EXAMPLE_3,
    EXAMPLE_4,
    EXAMPLE_5,
)

SYS_PROMPT = {"role": "system", "content": BASE_SYSTEM_INSTRUCTIONS}


def get_base_prompt_update_graph(request: str, scene_graph: str, use_icl: bool = True) -> List[str]:
    header = [SYS_PROMPT] + EXAMPLE_1 + EXAMPLE_2 + EXAMPLE_3 + EXAMPLE_4 + EXAMPLE_5 if use_icl else []
    prompt = (
        header
        + [
            {
                "role": "user",
                "content": f"{request}\nAdvice: \n- Recall the scene may be incomplete. \n- Carefully explain your reasoning in a step-by-step manner.\n- Reason over   connections, coordinates, and semantic relationships between objects and regions in the scene.\n\n"
                f"Scene graph:{scene_graph}",
            }
        ]
    )

    return prompt


INVALID_JSON = (
    "Feedback: You did not return your message in a valid json format. "
    "Received following error when parsing: {}. Update your response accordingly."
)
