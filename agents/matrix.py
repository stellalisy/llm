import os
from types import SimpleNamespace

from matrix import Cli

from .base import AsyncBaseAgent
from pydantic import BaseModel
import yaml

import asyncio


class Recipe(BaseModel):
  recipe_name: str
  ingredients: list[str]


# Models that support structured output
STRUCTURED_OUTPUT_MODELS = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp", "gemini-2.0-flash", "gemini-2.0-pro"]


class AsyncMatrixAgent(AsyncBaseAgent):
    def __init__(self, kwargs: dict):
        super().__init__()
        self.args = SimpleNamespace(**kwargs)
        self._set_default_args()
        # genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        # self.model = genai.GenerativeModel(self.args.model)
        if not os.path.exists(self.args.api_info):
            raise ValueError(f"API info file {self.args.api_info} not found")
        with open(self.args.api_info, 'r') as f:
            self.api_info = yaml.safe_load(f).get(self.args.api_account, {})

        from matrix.client import query_llm
        self.client = query_llm
        self.metadata = Cli().get_app_metadata(app_name=self.api_info.get("app_name", "scout"))
    
    def interact(self, prompt, temperature=0, max_tokens=256, history=None, json_mode=False, response_format=None, **kwargs):
        if json_mode:
            if isinstance(prompt, str):
                outputs = self._generate(prompt, json_mode=True, temperature=temperature, max_tokens=max_tokens)
            elif isinstance(prompt, list):
                outputs = self._generate_from_messages(prompt, json_mode=True, temperature=temperature, max_tokens=max_tokens)
            else:
                raise ValueError("Prompt must be a string or a list of dictionaries")
            
        elif response_format and self.args.model in STRUCTURED_OUTPUT_MODELS:
            if isinstance(prompt, str):
                outputs = self._generate(prompt, response_format=response_format, temperature=temperature, max_tokens=max_tokens)
            elif isinstance(prompt, list):
                outputs = self._generate_from_messages(prompt, response_format=response_format, temperature=temperature, max_tokens=max_tokens)
            else:
                raise ValueError("Prompt must be a string or a list of dictionaries")
            
        else:
            if isinstance(prompt, str):
                outputs = self._generate(prompt, temperature=temperature, max_tokens=max_tokens, history=history)
            elif isinstance(prompt, list):
                outputs = self._generate_from_messages(prompt, temperature=temperature, max_tokens=max_tokens, history=history)
            else:
                raise ValueError("Prompt must be a string or a list of dictionaries")
        
        
        responses = self._postprocess_output(outputs, response_format=response_format)

        return responses

    def _generate(self, prompt, json_mode=False, temperature=None, max_tokens=None, response_format=None, history=None):
        # Convert messages to the format expected by Matrix
        if response_format == "json" or json_mode:
            if not hasattr(self.args, "guided_decoding"): self.args.guided_decoding = {"json": {"type": "object"}}
            else: self.args.guided_decoding["json"] = {"type": "object"}
        elif response_format and isinstance(response_format, dict):
            # Handle complex JSON schema for guided decoding
            if not hasattr(self.args, "guided_decoding"): self.args.guided_decoding = {"json": response_format}
            else: self.args.guided_decoding["json"] = response_format
        if temperature:
            self.args.temperature = temperature
        if max_tokens:
            self.args.max_tokens = max_tokens
        if history:
            if isinstance(history, str):
                prompt = history + prompt
            else:
                print("[llm.agents.matrix.py(78): WARNING: this should probably not happen!")
                messages = history + [{'role': 'user', 'content': prompt}]
                return self._generate_from_messages(messages, json_mode=json_mode, temperature=temperature, max_tokens=max_tokens, response_format=response_format)

        kwargs = {k:v for k,v in vars(self.args).items() if k not in ["url", "model", "app_name", "requests", "api_info", "api_account", "frequency_penalty", "presence_penalty"]}
        output = asyncio.run(self.client.make_request(
            url=self.metadata["endpoints"]["head"],
            model=self.metadata["model_name"],
            app_name=self.metadata["name"],
            data={"prompt": prompt},
            **kwargs
        ))
        return output

    def _generate_from_messages(self, messages, json_mode=False, temperature=None, max_tokens=None, response_format=None, history=None):
        # Convert messages to the format expected by Matrix
        if response_format == "json" or json_mode:
            if not hasattr(self.args, "guided_decoding"): self.args.guided_decoding = {"json": {"type": "object"}}
            else: self.args.guided_decoding["json"] = {"type": "object"}
        elif response_format and isinstance(response_format, dict):
            # Handle complex JSON schema for guided decoding
            if not hasattr(self.args, "guided_decoding"): self.args.guided_decoding = {"json": response_format}
            else: self.args.guided_decoding["json"] = response_format
        if temperature:
            self.args.temperature = temperature
        if max_tokens:
            self.args.max_tokens = max_tokens
        if history:
            if isinstance(history, list):
                messages = history + messages
            else:
                print("[llm.agents.matrix.py(78): WARNING: this should probably not happen!")
                messages = [{'role': 'user', 'content': history}] + messages
        
        kwargs = {k:v for k,v in vars(self.args).items() if k not in ["url", "model", "app_name", "requests", "api_info", "api_account", "frequency_penalty", "presence_penalty"]}
        output = asyncio.run(self.client.make_request(
            url=self.metadata["endpoints"]["head"],
            model=self.metadata["model_name"],
            app_name=self.metadata["name"],
            data={"messages": messages},
            **kwargs
        ))
        return output

    def preprocess_input(self, text):
        return text

    def _postprocess_output(self, output, response_format=None):
        request_timestamp = output['request']['metadata']['request_timestamp'] if 'request_timestamp' in output['request']['metadata'] else None
        response_timestamp = output['response']['response_timestamp'] if 'response_timestamp' in output['response'] else None
        response_time = response_timestamp - request_timestamp if request_timestamp and response_timestamp else None

        return {
            "response_text": output['response']['text'][0],
            "input_tokens": output['response']['usage']['prompt_tokens'] if 'prompt_tokens' in output['response']['usage'] else 0,
            "output_tokens": output['response']['usage']['completion_tokens'] if 'completion_tokens' in output['response']['usage'] else 0,
            "cached_tokens": output['response']['usage']['cached_tokens'] if 'cached_tokens' in output['response']['usage'] else 0,
            "reasoning_tokens": output['response']['usage']['reasoning_tokens'] if 'reasoning_tokens' in output['response']['usage'] else 0,
            "finish_reason": output['response']['finish_reason'] if 'finish_reason' in output['response'] else None,
            "response_time": response_time,
        }