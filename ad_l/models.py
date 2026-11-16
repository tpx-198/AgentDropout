"""LangChain model adapters. Offline execution is the default."""

import math
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langsmith import tracing_context


@dataclass(frozen=True)
class ModelSettings:
    backend: str = "mock"
    model: str = "local-fixed-response"
    temperature: float | None = None
    max_tokens: int | None = None
    timeout: float = 60.0
    base_url: str | None = None

    def __post_init__(self):
        if self.backend not in {"mock", "deepseek", "openai"}:
            raise ValueError("backend must be mock, deepseek or openai")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be nonempty")
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.max_tokens is not None and (type(self.max_tokens) is not int or self.max_tokens < 1):
            raise ValueError("max_tokens must be a positive integer")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if self.base_url is not None:
            if not isinstance(self.base_url, str):
                raise ValueError("base_url must be a URL string")
            url = urlsplit(self.base_url)
            if (url.scheme not in {"http", "https"} or not url.hostname
                    or url.username is not None or url.password is not None or url.query or url.fragment):
                raise ValueError("base_url must be an HTTP(S) URL without credentials, query or fragment")


class ModelAdapter:
    def __init__(self, model: BaseChatModel, settings: ModelSettings):
        self.model = model
        self.settings = settings

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        with tracing_context(enabled=False):
            return self.model.invoke(messages)

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        with tracing_context(enabled=False):
            return await self.model.ainvoke(messages)


def create_model(
    settings: ModelSettings | None = None,
    *,
    allow_real_api: bool = False,
    env_file: Path | None = None,
    mock_response: str = "[MOCK fixed response]\nThe answer is 5",
    api_key_env: str | None = None,
    api_key: str | None = None,
) -> ModelAdapter:
    settings = settings or ModelSettings()
    if settings.backend == "mock":
        return ModelAdapter(FakeListChatModel(responses=[mock_response]), settings)
    if not allow_real_api:
        raise ValueError("Real model requests require allow_real_api=True")
    if env_file is not None:
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
    prefix = "DEEPSEEK" if settings.backend == "deepseek" else "OPENAI"
    key_name = api_key_env if api_key_env is not None else f"{prefix}_API_KEY"
    key = api_key.strip() if api_key is not None else os.environ.get(key_name, "").strip()
    if not key:
        raise ValueError(f"Set {key_name} in the environment or local .env")
    default_url = "https://api.deepseek.com" if settings.backend == "deepseek" else "https://api.openai.com/v1"
    base_url = settings.base_url or os.environ.get(f"{prefix}_BASE_URL") or default_url
    # Validate inherited environment URLs before passing them to a client.
    from dataclasses import replace

    settings = replace(settings, base_url=base_url)
    if settings.backend == "deepseek":
        from langchain_deepseek import ChatDeepSeek

        client_class = ChatDeepSeek
    else:
        from langchain_openai import ChatOpenAI

        client_class = ChatOpenAI

    model = client_class(
        model=settings.model, api_key=key,
        base_url=settings.base_url,
        temperature=settings.temperature, max_tokens=settings.max_tokens,
        timeout=settings.timeout, max_retries=0,
    )
    return ModelAdapter(model, settings)
