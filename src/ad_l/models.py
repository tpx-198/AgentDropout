"""LangChain 模型入口；本地模拟为默认后端。"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langsmith import tracing_context


@dataclass(frozen=True)
class ModelSettings:
    backend: str = "mock"
    model: str = "local-fixed-response"
    # 原请求未传这两个参数；None 表示沿用提供方默认值。
    temperature: float | None = None
    max_tokens: int | None = None
    timeout: float = 60.0
    max_retries: int = 0

    def __post_init__(self):
        if self.backend not in {"mock", "deepseek"}:
            raise ValueError("backend 必须为 mock 或 deepseek")
        if not self.model.strip():
            raise ValueError("model 不能为空")
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise ValueError("temperature 必须在 0 到 2 之间")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens 必须大于 0")
        if self.timeout <= 0 or self.max_retries < 0:
            raise ValueError("timeout 必须大于 0，max_retries 不能为负")


class ModelAdapter:
    def __init__(self, model: BaseChatModel, settings: ModelSettings):
        self._model = model
        self.settings = settings

    def invoke(self, messages: Sequence[BaseMessage]) -> AIMessage: # 同步调用
        with tracing_context(enabled=False):
            return self._model.invoke(list(messages))

    async def ainvoke(self, messages: Sequence[BaseMessage]) -> AIMessage: # 异步调用
        with tracing_context(enabled=False):
            return await self._model.ainvoke(list(messages))


def create_model_adapter( 
    settings: ModelSettings | None = None,
    *,
    mock_response: str = "[MOCK: fixed response, not model reasoning]\nThe answer is 5",
    allow_real_api: bool = False,
    env_file: Path | None = None,
) -> ModelAdapter:
    settings = settings or ModelSettings()
    if settings.backend == "mock":
        return ModelAdapter(FakeListChatModel(responses=[mock_response]), settings)
    if not allow_real_api:
        raise ValueError("真实模型请求需显式设置 allow_real_api=True")

    # 仅真实后端显式加载指定文件，不搜索父目录，也不覆盖已有环境变量。
    if env_file is not None:
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise ValueError("请通过环境变量或本地 .env 设置 DEEPSEEK_API_KEY")

    from langchain_deepseek import ChatDeepSeek

    model = ChatDeepSeek(
        model=settings.model,
        api_key=api_key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout,
        max_retries=settings.max_retries,
    )
    return ModelAdapter(model, settings)
