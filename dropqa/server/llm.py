"""LLM 服务模块"""

import logging
from typing import AsyncGenerator, Optional

from openai import AsyncOpenAI

from dropqa.common.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMService:
    """LLM 服务

    封装 OpenAI API 格式的 LLM 调用。
    """

    def __init__(self, config: LLMConfig):
        """初始化 LLM 服务

        Args:
            config: LLM 配置
        """
        self.config = config
        self.client = AsyncOpenAI(
            base_url=config.api_base,
            api_key=config.api_key,
        )

    async def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """非流式对话

        Args:
            messages: 消息列表，格式 [{"role": "user", "content": "..."}]
            temperature: 温度参数，None 则使用配置值
            max_tokens: 最大 token 数，None 则使用配置值

        Returns:
            LLM 响应文本
        """
        full_messages = self._build_messages(messages)

        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=full_messages,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
        )

        if not response.choices:
            return ""

        return response.choices[0].message.content or ""

    async def chat_stream(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """流式对话

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大 token 数

        Yields:
            LLM 响应文本块
        """
        full_messages = self._build_messages(messages)

        stream = await self.client.chat.completions.create(
            model=self.config.model,
            messages=full_messages,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _build_messages(self, messages: list[dict]) -> list[dict]:
        """构建完整消息列表（包含系统提示）

        Args:
            messages: 用户消息列表

        Returns:
            包含系统提示的完整消息列表
        """
        full_messages = []

        # 添加系统提示
        if self.config.system_prompt:
            full_messages.append({
                "role": "system",
                "content": self.config.system_prompt,
            })

        # 添加用户消息
        full_messages.extend(messages)

        return full_messages
