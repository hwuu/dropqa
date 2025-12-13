"""Embedding 服务测试"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dropqa.common.config import EmbeddingConfig
from dropqa.common.embedding import EmbeddingService


class TestEmbeddingService:
    """EmbeddingService 测试"""

    @pytest.fixture
    def config(self):
        """创建测试配置"""
        return EmbeddingConfig(
            api_base="http://localhost:11435/v1",
            api_key="test-key",
            model="test-model",
            dimension=128,
        )

    @pytest.fixture
    def service(self, config):
        """创建测试服务"""
        return EmbeddingService(config)

    def test_init(self, service, config):
        """测试初始化"""
        assert service.model == config.model
        assert service.dimension == config.dimension

    @pytest.mark.asyncio
    async def test_embed_single_text(self, service):
        """测试单个文本 embedding"""
        mock_embedding = [0.1] * 128

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=mock_embedding)]

        with patch.object(
            service._client.embeddings,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await service.embed("Hello world")

        assert len(result) == 128
        assert result == mock_embedding

    @pytest.mark.asyncio
    async def test_embed_empty_text(self, service):
        """测试空文本返回零向量"""
        result = await service.embed("")
        assert len(result) == 128
        assert all(v == 0.0 for v in result)

    @pytest.mark.asyncio
    async def test_embed_whitespace_only(self, service):
        """测试只有空白字符的文本返回零向量"""
        result = await service.embed("   ")
        assert len(result) == 128
        assert all(v == 0.0 for v in result)

    @pytest.mark.asyncio
    async def test_embed_batch_returns_vectors(self, service):
        """测试批量生成返回正确数量的向量"""
        mock_embeddings = [[0.1 * i] * 128 for i in range(3)]

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=e) for e in mock_embeddings]

        with patch.object(
            service._client.embeddings,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await service.embed_batch(["text1", "text2", "text3"])

        assert len(result) == 3
        for i, vec in enumerate(result):
            assert len(vec) == 128

    @pytest.mark.asyncio
    async def test_embed_batch_empty_input(self, service):
        """测试空输入返回空列表"""
        result = await service.embed_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_batch_with_empty_texts(self, service):
        """测试批量中包含空文本时正确处理"""
        mock_embedding = [0.5] * 128

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=mock_embedding)]

        with patch.object(
            service._client.embeddings,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_create:
            result = await service.embed_batch(["", "valid text", "  "])

        # 只有一个非空文本被发送
        mock_create.assert_called_once()
        call_args = mock_create.call_args
        assert call_args[1]["input"] == ["valid text"]

        # 结果保持原始顺序
        assert len(result) == 3
        assert all(v == 0.0 for v in result[0])  # 空文本 -> 零向量
        assert result[1] == mock_embedding  # 有效文本 -> 真实向量
        assert all(v == 0.0 for v in result[2])  # 空白文本 -> 零向量

    @pytest.mark.asyncio
    async def test_embed_batch_all_empty(self, service):
        """测试全是空文本时返回零向量列表"""
        result = await service.embed_batch(["", "  ", ""])

        assert len(result) == 3
        for vec in result:
            assert len(vec) == 128
            assert all(v == 0.0 for v in vec)

    @pytest.mark.asyncio
    async def test_embed_batch_preserves_order(self, service):
        """测试批量结果保持输入顺序"""
        # 模拟 API 返回的向量有不同值
        mock_embeddings = [
            [1.0] * 128,
            [2.0] * 128,
        ]

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=e) for e in mock_embeddings]

        with patch.object(
            service._client.embeddings,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await service.embed_batch(["first", "second"])

        assert result[0][0] == 1.0
        assert result[1][0] == 2.0

    @pytest.mark.asyncio
    async def test_embed_batch_api_error(self, service):
        """测试 API 错误时抛出异常"""
        with patch.object(
            service._client.embeddings,
            "create",
            new_callable=AsyncMock,
            side_effect=Exception("API Error"),
        ):
            with pytest.raises(Exception, match="API Error"):
                await service.embed_batch(["text"])
