import os
from types import SimpleNamespace
from .base import BaseAgent, AsyncBaseAgent
from transformers import AutoTokenizer
from .llama_405b_client import Llama3_Client

class AsyncLlama3Agent(AsyncBaseAgent):
    def __init__(self, kwargs: dict):
        super().__init__()
        self.args = SimpleNamespace(**kwargs)
        self._set_default_args()
        self.client = Llama3_Client(model_endpoint=os.environ["LLAMA_VLLM_ENDPOINT"], model_name=self.args.model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model)

    def generate(self, prompt, temperature=None, max_tokens=None, json_mode=False):
        response_obj = self.client.chat_completion(
            messages=prompt,
            model_name=self.args.model,
            max_tokens = self.args.max_tokens if max_tokens is None else max_tokens,
            temperature = self.args.temperature if temperature is None else temperature
        )
        assert response_obj["status_ok"]
        response_obj.pop("status_ok")
        return response_obj

    def preprocess_input(self, text):
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": text},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        return prompt

    def postprocess_output(self, output):
        response_text = output["content"].strip()
        return response_text