"""OpenAI-compatible chat client using the Python standard library."""

import json
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class ChatClient:
    def __init__(self):
        self.base_url = os.environ.get("MADREC_BASE_URL", "").rstrip("/")
        self.api_key = os.environ.get("MADREC_API_KEY", "")
        self.model = os.environ.get("MADREC_MODEL", "qwen-flash-2025-07-28")
        if not self.base_url or not self.api_key:
            raise ValueError("set MADREC_BASE_URL and MADREC_API_KEY before calling the LLM")

    def __call__(self, instruction: str, context: dict) -> dict:
        payload = {
            "model": self.model,
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 8192,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ],
        }
        request = Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                text = json.load(response)["choices"][0]["message"]["content"].strip()
        except HTTPError as exc:
            raise RuntimeError(f"chat completion failed with HTTP {exc.code}") from exc

        if text.startswith("```") and text.endswith("```"):
            text = text.split("\n", 1)[-1][:-3].strip()
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("LLM response must be a JSON object")
        return parsed
