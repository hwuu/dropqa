"""Embedding 服务模块

提供文本向量化功能，支持批量异步调用。
"""

import logging
from typing import Optional

from openai import AsyncOpenAI

from dropqa.common.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Embedding 服务

    使用 OpenAI API 格式的 Embedding 接口生成文本向量。
    支持批量异步调用以提高效率。
    """

    def __init__(self, config: EmbeddingConfig):
        """初始化 Embedding 服务

        Args:
            config: Embedding 配置
        """
        self._config = config
        self._client = AsyncOpenAI(
            base_url=config.api_base,
            api_key=config.api_key,
        )
        self._model = config.model
        self._dimension = config.dimension

    @property
    def model(self) -> str:
        """获取模型名称"""
        return self._model

    @property
    def dimension(self) -> int:
        """获取向量维度"""
        return self._dimension

    async def embed(self, text: str) -> list[float]:
        """生成单个文本的 embedding

        Args:
            text: 输入文本

        Returns:
            向量列表
        """
        if not text or not text.strip():
            # 返回零向量
            return [0.0] * self._dimension

        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量生成 embedding

        Args:
            texts: 输入文本列表

        Returns:
            向量列表的列表，顺序与输入一致
        """
        if not texts:
            return []

        # 过滤空文本，记录位置
        non_empty_indices = []
        non_empty_texts = []
        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(text.strip())

        # 如果全是空文本，返回零向量
        if not non_empty_texts:
            return [[0.0] * self._dimension for _ in texts]

        logger.debug(f"[Embedding] 批量生成 {len(non_empty_texts)} 个文本的向量")

        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=non_empty_texts,
            )

            # 提取向量
            non_empty_embeddings = [data.embedding for data in response.data]

            # 还原到原始顺序，空文本位置填充零向量
            result = [[0.0] * self._dimension for _ in texts]
            for idx, embedding in zip(non_empty_indices, non_empty_embeddings):
                result[idx] = embedding

            logger.debug(f"[Embedding] 成功生成 {len(non_empty_embeddings)} 个向量，维度 {len(non_empty_embeddings[0]) if non_empty_embeddings else 0}")

            return result

        except Exception as e:
            logger.error(f"[Embedding] 生成向量失败: {e}")
            raise
