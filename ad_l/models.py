"""LangChain model adapters. Offline execution is the default."""

import math
import os
from dataclasses import dataclass
from pathlib import Path

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

    def __post_init__(self):
        if self.backend not in {"mock", "deepseek"}:
            raise ValueError("backend must be mock or deepseek")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be nonempty")
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.max_tokens is not None and (type(self.max_tokens) is not int or self.max_tokens < 1):
            raise ValueError("max_tokens must be a positive integer")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be finite and positive")


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
) -> ModelAdapter:
    settings = settings or ModelSettings()
    if settings.backend == "mock":
        return ModelAdapter(FakeListChatModel(responses=[mock_response]), settings)
    if not allow_real_api:
        raise ValueError("DeepSeek requests require allow_real_api=True")
    if env_file is not None:
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise ValueError("Set DEEPSEEK_API_KEY in the environment or local .env")
    from langchain_deepseek import ChatDeepSeek

    model = ChatDeepSeek(
        model=settings.model, api_key=key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
        temperature=settings.temperature, max_tokens=settings.max_tokens,
        timeout=settings.timeout, max_retries=0,
    )
    return ModelAdapter(model, settings)
