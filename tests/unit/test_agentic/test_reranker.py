"""Reranker 测试"""

import json
import uuid
from unittest.mock import AsyncMock

import pytest

from dropqa.server.agentic.reranker import Reranker, RankedResult
from dropqa.server.search import BreadcrumbItem, NodeContext


class TestReranker:
    """Reranker 测试"""

    @pytest.fixture
    def sample_contexts(self):
        """创建示例上下文"""
        return [
            NodeContext(
                node_id=uuid.uuid4(),
                content="DropQA 是一个文档问答系统",
                breadcrumb=[BreadcrumbItem(title="第1章", summary=None, depth=1)],
                document_name="intro.md",
            ),
            NodeContext(
                node_id=uuid.uuid4(),
                content="系统架构包括索引服务和问答服务",
                breadcrumb=[BreadcrumbItem(title="第2章", summary=None, depth=1)],
                document_name="arch.md",
            ),
            NodeContext(
                node_id=uuid.uuid4(),
                content="Python 是一种编程语言",
                breadcrumb=[BreadcrumbItem(title="附录", summary=None, depth=1)],
                document_name="appendix.md",
            ),
        ]

    @pytest.mark.asyncio
    async def test_rerank_empty_contexts(self):
        """测试空上下文"""
        mock_llm = AsyncMock()
        reranker = Reranker(mock_llm)

        result = await reranker.rerank("测试问题", [])

        assert result == []

    @pytest.mark.asyncio
    async def test_rerank_single_context(self, sample_contexts):
        """测试单个上下文（直接返回）"""
        mock_llm = AsyncMock()
        reranker = Reranker(mock_llm)

        result = await reranker.rerank("测试问题", [sample_contexts[0]])

        assert len(result) == 1
        assert result[0].context == sample_contexts[0]
        mock_llm.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_rerank_success(self, sample_contexts):
        """测试成功重排序"""
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value=json.dumps({
            "scores": [
                {"id": 0, "score": 9, "reason": "直接相关"},
                {"id": 1, "score": 7, "reason": "部分相关"},
                {"id": 2, "score": 2, "reason": "不相关"},
            ]
        }))
        reranker = Reranker(mock_llm)

        result = await reranker.rerank("什么是 DropQA", sample_contexts, top_k=2)

        assert len(result) == 2
        # 应该按分数降序排列
        assert result[0].score >= result[1].score
        assert result[0].score == 9

    @pytest.mark.asyncio
    async def test_rerank_llm_error_fallback(self, sample_contexts):
        """测试 LLM 错误时的回退"""
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(side_effect=Exception("LLM 错误"))
        reranker = Reranker(mock_llm)

        result = await reranker.rerank("测试问题", sample_contexts, top_k=2)

        # 应该返回原始顺序
        assert len(result) == 2
        assert result[0].context == sample_contexts[0]


class TestRankedResult:
    """RankedResult 数据类测试"""

    def test_creation(self):
        """测试创建"""
        ctx = NodeContext(
            node_id=uuid.uuid4(),
            content="test",
            breadcrumb=[],
            document_name="test.md",
        )
        result = RankedResult(context=ctx, score=8.5, reason="高度相关")

        assert result.context == ctx
        assert result.score == 8.5
        assert result.reason == "高度相关"
