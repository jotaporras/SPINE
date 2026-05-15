import time
from typing import Dict, List, Tuple

import tiktoken
from openai import OpenAI

from spine.prompts.prompts import get_base_prompt_update_graph
from spine.spine_util import from_huggingface


from openai import OpenAI
import time
import tiktoken
from typing import Tuple

class GPT5:
    def __init__(self, temperature: float = 0.05) -> None:
        """Wrapper for OpenAI (GPT-5.1, no reasoning)"""
        self.client = OpenAI()
        self.model = "gpt-5.1"
        self.temperature = temperature
        self.token_encoder = tiktoken.get_encoding("cl100k_base")
        self.token_history = []
        self.time_history = []

    def query_llm(self, msg: str) -> Tuple[str, bool]:
        self.token_history.append(len(self.token_encoder.encode(str(msg))))
        self.most_recent_query = msg

        try:
            t1 = time.time()

            response = self.client.responses.create(
                model=self.model,
                input=msg,
                temperature=self.temperature,
                max_output_tokens=2048,
                top_p=1,
                # No reasoning param → defaults to non-reasoning
                # response_format={"type": "json_object"},
            )

            top_msg = response.output_text

            self.time_history.append(time.time() - t1)
            return top_msg, True

        except Exception as ex:
            return f"Error: network dropout: {ex}", False

    def format_prompt(self, base_request: str, graph_as_json: str) -> str:
        return get_base_prompt_update_graph(
            request=base_request, scene_graph=graph_as_json
        )




class OpenAILLM:
    def __init__(self, temperature: float = 0.05) -> None:
        """Wrapper for OpenAI"""
        self.client = OpenAI()
        self.model = "gpt-4o"
        self.temperature = temperature
        self.token_encoder = tiktoken.get_encoding("cl100k_base")
        self.token_history = []
        self.time_history = []

    def query_llm(self, msg: str) -> Tuple[str, bool]:
        self.token_history.append(len(self.token_encoder.encode(str(msg))))
        self.most_recent_query = msg
        try:
            t1 = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=msg,
                temperature=self.temperature,
                max_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                response_format={"type": "json_object"},
            )
            top_msg = response.choices[0].message.content
            self.time_history.append(time.time() - t1)
            return top_msg, True
        except Exception as ex:
            return "Error: network dropout", False

    def format_prompt(self, base_request: str, graph_as_json: str) -> str:
        return get_base_prompt_update_graph(
            request=base_request, scene_graph=graph_as_json
        )


class HuggingFaceLLM:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model, self.tokenizer = from_huggingface(model_path)
        self.model = self.model.to(self.device)

    def format_prompt(self, base_request: str, graph_as_json: str) -> str:
        return [
            {
                "role": "user",
                "content": f"task: {base_request}. scene graph {graph_as_json}",
            }
        ]

    def query_llm(self, msg: List[Dict[str, str]]):
        inputs = self.tokenizer.apply_chat_template(
            msg,
            tokenize=True,
            add_generation_prompt=True,  # Must add for generation
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=4048,
            use_cache=True,
            temperature=0.01,
            min_p=0.1,
        )
        out = self.tokenizer.batch_decode(outputs)

        planner_response = out[0].split("end_header_id|>")[-1].split("<|eot_id|>")[0]

        return planner_response, True


class InMemoryLLM:
    """LLM wrapper that uses a pre-loaded model and tokenizer instead of loading from disk."""

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def format_prompt(self, base_request: str, graph_as_json: str) -> str:
        return [
            {
                "role": "user",
                "content": f"task: {base_request}. scene graph {graph_as_json}",
            }
        ]

    def query_llm(self, msg: List[Dict[str, str]]):
        inputs = self.tokenizer.apply_chat_template(
            msg,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=4048,
            use_cache=True,
            temperature=0.01,
            min_p=0.1,
        )
        out = self.tokenizer.batch_decode(outputs)

        planner_response = out[0].split("end_header_id|>")[-1].split("<|eot_id|>")[0]

        return planner_response, True


