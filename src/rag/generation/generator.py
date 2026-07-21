"""LLM call wrapper. Supports two free providers:
- Groq: free-tier hosted API, very fast inference
- Ollama: fully local, zero cost, zero API key

Switch via config.yaml -> generation.provider
"""
import os
import time
from typing import Dict, List, Tuple


class Generator:
    def __init__(self, config: dict):
        self.config = config["generation"]
        self.provider = self.config["provider"]

        if self.provider == "groq":
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError(
                    "GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys "
                    "or switch generation.provider to 'ollama' in configs/config.yaml"
                )
            self.client = Groq(api_key=api_key)
        elif self.provider == "ollama":
            import ollama
            self.client = ollama
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def generate(self, messages: List[Dict[str, str]]) -> Tuple[str, Dict]:
        """Call the LLM and return (answer_text, usage_metadata)."""
        start = time.time()

        if self.provider == "groq":
            response = self.client.chat.completions.create(
                model=self.config["model_name"],
                messages=messages,
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"],
            )
            answer = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        else:  # ollama
            response = self.client.chat(
                model=self.config["ollama_model_name"],
                messages=messages,
                options={"temperature": self.config["temperature"]},
            )
            answer = response["message"]["content"]
            usage = {
                "prompt_tokens": response.get("prompt_eval_count", 0),
                "completion_tokens": response.get("eval_count", 0),
                "total_tokens": response.get("prompt_eval_count", 0) + response.get("eval_count", 0),
            }

        usage["latency_seconds"] = round(time.time() - start, 3)
        return answer, usage
