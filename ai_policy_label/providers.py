# -*- coding: utf-8 -*-

import requests


class LLMProvider(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.runtime = cfg["runtime"]

    def chat(
        self,
        system_prompt,
        user_prompt
    ):
        raise NotImplementedError


class OllamaProvider(LLMProvider):

    def __init__(self, cfg):

        super(OllamaProvider, self).__init__(
            cfg
        )

        c = cfg["ollama"]

        self.url = (
            c["base_url"].rstrip("/")
            + c["endpoint"]
        )

        self.model = c["model"]


    def chat(
        self,
        system_prompt,
        user_prompt
    ):

        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": float(
                    self.runtime.get(
                        "temperature",
                        0.0
                    )
                ),
                "num_predict": int(
                    self.runtime.get(
                        "max_tokens",
                        300
                    )
                ),
            },
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                },
            ],
        }

        r = requests.post(
            self.url,
            json=payload,
            timeout=int(
                self.runtime.get(
                    "timeout",
                    300
                )
            ),
        )

        r.raise_for_status()

        obj = r.json()

        return (
            obj["message"]["content"]
        )


class VLLMProvider(LLMProvider):

    def __init__(self, cfg):

        super(VLLMProvider, self).__init__(
            cfg
        )

        c = cfg["vllm"]

        self.url = (
            c["base_url"].rstrip("/")
            + c["endpoint"]
        )

        self.model = c["model"]

        self.api_key = c.get(
            "api_key",
            "EMPTY"
        )


    def chat(
        self,
        system_prompt,
        user_prompt
    ):

        payload = {
            "model": self.model,
            "temperature": float(
                self.runtime.get(
                    "temperature",
                    0.0
                )
            ),
            "max_tokens": int(
                self.runtime.get(
                    "max_tokens",
                    300
                )
            ),
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                },
            ],
        }

        headers = {
            "Content-Type":
            "application/json",

            "Authorization":
            "Bearer %s"
            % self.api_key,
        }

        r = requests.post(
            self.url,
            headers=headers,
            json=payload,
            timeout=int(
                self.runtime.get(
                    "timeout",
                    300
                )
            ),
        )

        r.raise_for_status()

        obj = r.json()

        return (
            obj["choices"][0]
            ["message"]
            ["content"]
        )


class MLXProvider(LLMProvider):

    def __init__(self, cfg):

        super(MLXProvider, self).__init__(
            cfg
        )

        c = cfg["mlx"]

        self.url = (
            c["base_url"].rstrip("/")
            + c["endpoint"]
        )

        self.model = c["model"]

        self.api_key = c.get(
            "api_key",
            "EMPTY"
        )


    def chat(
        self,
        system_prompt,
        user_prompt
    ):

        payload = {
            "model": self.model,

            "temperature": float(
                self.runtime.get(
                    "temperature",
                    0.0
                )
            ),

            "max_tokens": int(
                self.runtime.get(
                    "max_tokens",
                    300
                )
            ),

            # 关键：关闭 Qwen3 thinking mode
            "chat_template_kwargs": {
                "enable_thinking": False
            },

            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                },
            ],
        }

        headers = {
            "Content-Type":
            "application/json",
        }

        r = requests.post(
            self.url,
            headers=headers,
            json=payload,
            timeout=int(
                self.runtime.get(
                    "timeout",
                    300
                )
            ),
        )

        r.raise_for_status()

        obj = r.json()

        return (
            obj["choices"][0]
            ["message"]
            ["content"]
        )

def build_provider(cfg):

    name = str(
        cfg["provider"]
    ).strip().lower()

    if name == "ollama":

        return OllamaProvider(
            cfg
        )

    if name == "vllm":

        return VLLMProvider(
            cfg
        )

    if name == "mlx":

        return MLXProvider(
            cfg
        )

    raise ValueError(
        "provider must be "
        "'ollama', 'vllm' or 'mlx'"
    )