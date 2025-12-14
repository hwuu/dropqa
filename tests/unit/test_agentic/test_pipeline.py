"""Pipeline 测试"""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from dropqa.server.agentic.config import (
    AgenticConfig,
    HybridSearchConfig,
    MultiRoundConfig,
    QueryRewriteConfig,
    RerankConfig,
)
from dropqa.server.agentic.pipeline import AgenticRAGPipeline, AgenticRAGResult
from dropqa.server.search import BreadcrumbItem, NodeContext, SearchResult


class TestAgenticRAGPipeline:
    """AgenticRAGPipeline 测试"""

    @pytest.fixture
    def mock_search_service(self):
        """创建模拟搜索服务"""
        service = AsyncMock()
        return service

    @pytest.fixture
    def mock_llm_service(self):
        """创建模拟 LLM 服务"""
        service = AsyncMock()
        return service

    @pytest.fixture
    def sample_search_results(self):
        """创建示例搜索结果"""
        return [
            SearchResult(
                node_id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                title="测试标题",
                content="测试内容",
                rank=0.9,
            )
        ]

    @pytest.fixture
    def sample_context(self):
        """创建示例上下文"""
        return NodeContext(
            node_id=uuid.uuid4(),
            content="DropQA 是一个文档问答系统",
            breadcrumb=[BreadcrumbItem(title="第1章", summary=None, depth=1)],
            document_name="intro.md",
        )

    @pytest.mark.asyncio
    async def test_pipeline_disabled(self, mock_search_service, mock_llm_service, sample_context):
        """测试所有功能禁用时的基本流程"""
        config = AgenticConfig(
            enabled=True,
            query_rewrite=QueryRewriteConfig(enabled=False),
            hybrid_search=HybridSearchConfig(enabled=False),
            multi_round=MultiRoundConfig(enabled=False),
            rerank=RerankConfig(enabled=False),
        )

        mock_search_service.fulltext_search = AsyncMock(return_value=[
            SearchResult(
                node_id=sample_context.node_id,
                document_id=uuid.uuid4(),
                title="测试",
                content="内容",
                rank=0.9,
            )
        ])
        mock_search_service.get_node_context = AsyncMock(return_value=sample_context)

        pipeline = AgenticRAGPipeline(
            config=config,
            search_service=mock_search_service,
            llm_service=mock_llm_service,
        )

        result = await pipeline.run("什么是 DropQA？")

        assert isinstance(result, AgenticRAGResult)
        assert result.question == "什么是 DropQA？"
        assert result.rewritten_query is None  # 查询改写已禁用
        assert len(result.ranked_results) > 0

    @pytest.mark.asyncio
    async def test_pipeline_with_query_rewrite(self, mock_search_service, mock_llm_service, sample_context):
        """测试启用查询改写"""
        config = AgenticConfig(
            enabled=True,
            query_rewrite=QueryRewriteConfig(enabled=True),
            hybrid_search=HybridSearchConfig(enabled=False),
            multi_round=MultiRoundConfig(enabled=False),
            rerank=RerankConfig(enabled=False),
        )

        # Mock LLM 返回查询改写结果
        mock_llm_service.chat = AsyncMock(return_value='{"keywords": ["DropQA"], "fulltext_query": "DropQA 文档问答", "semantic_query": "什么是 DropQA 系统"}')

        mock_search_service.fulltext_search = AsyncMock(return_value=[
            SearchResult(
                node_id=sample_context.node_id,
                document_id=uuid.uuid4(),
                title="测试",
                content="内容",
                rank=0.9,
            )
        ])
        mock_search_service.get_node_context = AsyncMock(return_value=sample_context)

        pipeline = AgenticRAGPipeline(
            config=config,
            search_service=mock_search_service,
            llm_service=mock_llm_service,
        )

        result = await pipeline.run("什么是 DropQA？")

        assert result.rewritten_query is not None
        assert "DropQA" in result.rewritten_query.keywords

    @pytest.mark.asyncio
    async def test_pipeline_empty_results(self, mock_search_service, mock_llm_service):
        """测试空搜索结果"""
        config = AgenticConfig(
            enabled=True,
            query_rewrite=QueryRewriteConfig(enabled=False),
            hybrid_search=HybridSearchConfig(enabled=False),
            multi_round=MultiRoundConfig(enabled=False),
            rerank=RerankConfig(enabled=False),
        )

        mock_search_service.fulltext_search = AsyncMock(return_value=[])

        pipeline = AgenticRAGPipeline(
            config=config,
            search_service=mock_search_service,
            llm_service=mock_llm_service,
        )

        result = await pipeline.run("不存在的问题")

        assert len(result.contexts) == 0
        assert len(result.ranked_results) == 0


class TestAgenticRAGResult:
    """AgenticRAGResult 数据类测试"""

    def test_creation(self):
        """测试创建"""
        result = AgenticRAGResult(
            question="测试问题",
            rewritten_query=None,
            contexts=[],
            ranked_results=[],
        )

        assert result.question == "测试问题"
        assert result.rewritten_query is None
        assert result.contexts == []
        assert result.ranked_results == []
