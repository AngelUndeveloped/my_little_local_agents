from llama_index.core.llms import LLM
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.base.llms.types import CompletionResponse, LLMMetadata
import requests
import json

class LMStudioLLM(LLM):
    def __init__(self, model_name: str = "deepseek-r1-distill-qwen-14b", api_base: str = "http://127.0.0.1:1234/v1"):
        super().__init__()
        self.model_name = model_name
        self.api_base = api_base
        self._metadata = LLMMetadata(
            context_window=8192,
            num_output=4096,
            model_name=model_name,
        )

    @property
    def metadata(self) -> LLMMetadata:
        return self._metadata

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.7),
        }

        response = requests.post(f"{self.api_base}/chat/completions", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            return CompletionResponse(text=text)
        else:
            raise Exception(f"Error from LMStudio API: {response.text}")

    @llm_completion_callback()
    def chat(self, messages: list, **kwargs) -> CompletionResponse:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.7),
        }

        response = requests.post(f"{self.api_base}/chat/completions", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            return CompletionResponse(text=text)
        else:
            raise Exception(f"Error from LMStudio API: {response.text}") 