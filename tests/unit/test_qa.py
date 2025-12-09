"""QA 服务测试"""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from dropqa.server.qa import QAService, QAResponse, SourceReference
from dropqa.server.search import SearchResult, NodeContext, BreadcrumbItem


class TestSourceReference:
    """SourceReference 测试"""

    def test_source_reference_creation(self):
        """测试创建 SourceReference"""
        ref = SourceReference(
            document_name="report.md",
            path="第1章 > 1.1 背景",
            content_snippet="这是相关内容片段...",
        )
        assert ref.document_name == "report.md"
        assert "第1章" in ref.path


class TestQAResponse:
    """QAResponse 测试"""

    def test_qa_response_creation(self):
        """测试创建 QAResponse"""
        response = QAResponse(
            answer="这是回答内容",
            sources=[
                SourceReference(
                    document_name="doc.md",
                    path="Chapter 1",
                    content_snippet="snippet",
                )
            ],
        )
        assert response.answer == "这是回答内容"
        assert len(response.sources) == 1


class TestQAService:
    """QAService 测试"""

    @pytest.fixture
    def mock_search_service(self):
        """创建模拟搜索服务"""
        service = MagicMock()
        service.fulltext_search = AsyncMock()
        service.get_node_context = AsyncMock()
        return service

    @pytest.fixture
    def mock_llm_service(self):
        """创建模拟 LLM 服务"""
        service = MagicMock()
        service.chat = AsyncMock()
        return service

    @pytest.fixture
    def qa_service(self, mock_search_service, mock_llm_service):
        """创建 QAService 实例"""
        return QAService(mock_search_service, mock_llm_service)

    def test_qa_service_init(self, qa_service, mock_search_service, mock_llm_service):
        """测试 QAService 初始化"""
        assert qa_service.search_service == mock_search_service
        assert qa_service.llm_service == mock_llm_service

    @pytest.mark.asyncio
    async def test_ask_returns_response(self, qa_service, mock_search_service, mock_llm_service):
        """测试 ask 方法返回响应"""
        # 模拟搜索结果
        node_id = uuid.uuid4()
        mock_search_service.fulltext_search.return_value = [
            SearchResult(
                node_id=node_id,
                document_id=uuid.uuid4(),
                title="背景介绍",
                content="这是关于项目背景的内容",
                rank=0.9,
            )
        ]

        # 模拟节点上下文
        mock_search_service.get_node_context.return_value = NodeContext(
            node_id=node_id,
            content="这是关于项目背景的内容",
            breadcrumb=[
                BreadcrumbItem(title="文档", summary=None, depth=0),
                BreadcrumbItem(title="第1章", summary="概述", depth=1),
            ],
            document_name="report.md",
        )

        # 模拟 LLM 响应
        mock_llm_service.chat.return_value = "根据文档，项目背景是..."

        # 执行
        response = await qa_service.ask("项目背景是什么？")

        # 验证
        assert response.answer == "根据文档，项目背景是..."
        assert len(response.sources) == 1
        assert response.sources[0].document_name == "report.md"

    @pytest.mark.asyncio
    async def test_ask_no_search_results(self, qa_service, mock_search_service, mock_llm_service):
        """测试无搜索结果时的处理"""
        mock_search_service.fulltext_search.return_value = []
        mock_llm_service.chat.return_value = "抱歉，没有找到相关文档。"

        response = await qa_service.ask("不存在的内容")

        assert "没有找到" in response.answer or response.answer != ""
        assert len(response.sources) == 0

    @pytest.mark.asyncio
    async def test_ask_empty_question(self, qa_service):
        """测试空问题"""
        response = await qa_service.ask("")

        assert response.answer != ""
        assert len(response.sources) == 0

    def test_build_context_prompt(self, qa_service):
        """测试构建上下文 Prompt"""
        contexts = [
            NodeContext(
                node_id=uuid.uuid4(),
                content="内容1",
                breadcrumb=[BreadcrumbItem(title="章节1", summary=None, depth=1)],
                document_name="doc1.md",
            ),
            NodeContext(
                node_id=uuid.uuid4(),
                content="内容2",
                breadcrumb=[BreadcrumbItem(title="章节2", summary=None, depth=1)],
                document_name="doc2.md",
            ),
        ]

        prompt = qa_service._build_context_prompt(contexts, "测试问题")

        assert "内容1" in prompt
        assert "内容2" in prompt
        assert "测试问题" in prompt

    def test_truncate_content(self, qa_service):
        """测试内容截断"""
        long_content = "a" * 1000
        truncated = qa_service._truncate_content(long_content, max_length=100)

        assert len(truncated) <= 103  # 100 + "..."
        assert truncated.endswith("...")
