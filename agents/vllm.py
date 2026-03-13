from vllm import LLM, SamplingParams
from .base import BaseAgent

class VllmAgent(BaseAgent):
    def __init__(self, model_name, num_gpus=2, max_tokens=265, **kwargs):
        self.model_name = model_name
        self.model = LLM(model=model_name, tensor_parallel_size=num_gpus, gpu_memory_utilization=0.5)
        self.tokenizer = self.model.get_tokenizer()
        self.max_tokens = max_tokens
        self.temperature = 1
        self.stop_tokens = kwargs.get("stop_tokens", None)
        
    def preprocess_input(self, text):
        messages = [
        # {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": text},
        ]
        chat_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return chat_prompt

    def postprocess_output(self, output):
        return output.outputs[0].text.strip()

    def interact(self, text, temperature=None, max_tokens=None):
        return self.batch_interact([text], temperature=temperature, max_tokens=max_tokens)[0]

    def batch_interact(self, texts, temperature=None, max_tokens=None, stop_tokens=None, prompt_logprobs=None):
        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature
        if stop_tokens is None:
            stop_tokens = self.stop_tokens

        sampling_params = SamplingParams(temperature=temperature, top_p=1, max_tokens=max_tokens, stop=stop_tokens, prompt_logprobs=prompt_logprobs)
        prompts = [self.preprocess_input(text) for text in texts]
        outputs = self.model.generate(prompts, sampling_params=sampling_params)
        responses = [self.postprocess_output(output) for output in outputs]

        return responses

    def batch_compute_likelihood(self, texts, temperature=None, max_tokens=None, stop_tokens=None, prompt_logprobs=None):
        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature
        if stop_tokens is None:
            stop_tokens = self.stop_tokens

        sampling_params = SamplingParams(temperature=temperature, top_p=1, max_tokens=max_tokens, stop=self.stop_tokens, prompt_logprobs=prompt_logprobs)
        prompts = [self.preprocess_input(text) for text in texts]
        outputs = self.model.generate(prompts, sampling_params=sampling_params)

        return outputs

class VllmBaseAgent(VllmAgent):
    def preprocess_input(self, text):
        return text