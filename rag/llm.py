"""
LLM Client Wrapper for RAG Generation
Supports Groq API via direct REST calls (Qwen-27b / GPT-OSS-120b) with local deterministic fallback.
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

class LLMClient:
    def __init__(self, model_name: str = "qwen/qwen3.8-27b"):
        self.model_name = model_name
        self.api_key = self._load_api_key()
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self._backend = "groq" if bool(self.api_key) else "local_template"

    def _load_api_key(self) -> str:
        # Check environment variable
        key = os.environ.get("GROQ_API_KEY", "").strip()
        if key:
            return key

        # Check .env file in project root
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.is_file():
            try:
                with env_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GROQ_API_KEY="):
                            return line.split("=", 1)[1].strip().strip("'\"")
            except Exception:
                pass
        return ""

    @property
    def backend(self) -> str:
        return self._backend

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024
    ) -> str:
        """
        Sends prompt to Groq API if configured; otherwise returns structured fallback.
        """
        if self.is_available():
            try:
                import requests
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                data = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }

                resp = requests.post(self.api_url, json=data, headers=headers, timeout=12)
                if resp.status_code == 200:
                    result = resp.json()
                    choices = result.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", prompt)
            except Exception as e:
                # Graceful fallback on network or API error
                pass

        # Local template mode: prompt already contains formatted context
        return prompt
