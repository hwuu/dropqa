"""全文搜索服务测试"""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from dropqa.server.search import (
    SearchService,
    SearchResult,
    BreadcrumbItem,
    NodeContext,
)


class TestSearchResult:
    """SearchResult 测试"""

    def test_search_result_creation(self):
        """测试创建 SearchResult"""
        result = SearchResult(
            node_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            title="Test Title",
            content="Test content",
            rank=0.85,
        )
        assert result.title == "Test Title"
        assert result.rank == 0.85

    def test_search_result_optional_fields(self):
        """测试可选字段"""
        result = SearchResult(
            node_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            title=None,
            content=None,
            rank=0.5,
        )
        assert result.title is None
        assert result.content is None


class TestSearchService:
    """SearchService 测试"""

    @pytest.fixture
    def mock_db(self):
        """创建模拟数据库"""
        db = MagicMock()
        return db

    @pytest.fixture
    def search_service(self, mock_db):
        """创建 SearchService 实例"""
        return SearchService(mock_db)

    def test_search_service_init(self, search_service, mock_db):
        """测试 SearchService 初始化"""
        assert search_service.db == mock_db

    @pytest.mark.asyncio
    async def test_fulltext_search_returns_results(self, search_service, mock_db):
        """测试全文搜索返回结果"""
        # 模拟数据库返回
        mock_rows = [
            MagicMock(
                id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                title="Chapter 1",
                content="This is about Python programming",
                rank=0.9,
            ),
            MagicMock(
                id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                title="Chapter 2",
                content="More Python content",
                rank=0.7,
            ),
        ]

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall = MagicMock(return_value=mock_rows)
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_db.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_db.session.return_value.__aexit__ = AsyncMock(return_value=None)

        results = await search_service.fulltext_search("Python", top_k=10)

        assert len(results) == 2
        assert results[0].rank >= results[1].rank  # 按相关度排序

    @pytest.mark.asyncio
    async def test_fulltext_search_empty_query(self, search_service):
        """测试空查询返回空结果"""
        results = await search_service.fulltext_search("", top_k=10)
        assert results == []

    @pytest.mark.asyncio
    async def test_fulltext_search_respects_top_k(self, search_service, mock_db):
        """测试 top_k 参数"""
        # 模拟返回 5 个结果
        mock_rows = [
            MagicMock(
                id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                title=f"Title {i}",
                content=f"Content {i}",
                rank=0.9 - i * 0.1,
            )
            for i in range(5)
        ]

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall = MagicMock(return_value=mock_rows[:3])  # 只返回 3 个
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_db.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_db.session.return_value.__aexit__ = AsyncMock(return_value=None)

        results = await search_service.fulltext_search("test", top_k=3)

        assert len(results) <= 3

    def test_build_tsquery_simple(self, search_service):
        """测试构建简单查询"""
        query = search_service._build_tsquery("hello world")
        assert "hello" in query
        assert "world" in query

    def test_build_tsquery_chinese(self, search_service):
        """测试构建中文查询"""
        query = search_service._build_tsquery("项目 预算")
        assert "项目" in query
        assert "预算" in query

    def test_build_tsquery_special_chars(self, search_service):
        """测试特殊字符处理"""
        query = search_service._build_tsquery("test's & query")
        # 应该安全处理特殊字符
        assert query is not None


class TestBreadcrumb:
    """Breadcrumb 测试"""

    def test_breadcrumb_item_creation(self):
        """测试创建 BreadcrumbItem"""
        item = BreadcrumbItem(
            title="Chapter 1",
            summary="Introduction to the topic",
            depth=1,
        )
        assert item.title == "Chapter 1"
        assert item.depth == 1

    def test_node_context_creation(self):
        """测试创建 NodeContext"""
        context = NodeContext(
            node_id=uuid.uuid4(),
            content="This is the content",
            breadcrumb=[
                BreadcrumbItem(title="Doc", summary=None, depth=0),
                BreadcrumbItem(title="Chapter 1", summary="Intro", depth=1),
            ],
            document_name="test.md",
        )
        assert len(context.breadcrumb) == 2
        assert context.document_name == "test.md"

    def test_node_context_path_string(self):
        """测试路径字符串生成"""
        context = NodeContext(
            node_id=uuid.uuid4(),
            content="Content",
            breadcrumb=[
                BreadcrumbItem(title="文档", summary=None, depth=0),
                BreadcrumbItem(title="第1章", summary="概述", depth=1),
                BreadcrumbItem(title="1.1 背景", summary="背景介绍", depth=2),
            ],
            document_name="report.md",
        )
        path = context.get_path_string()
        assert "第1章" in path
        assert "1.1 背景" in path


class TestGetNodeContext:
    """get_node_context 测试"""

    @pytest.fixture
    def mock_db(self):
        """创建模拟数据库"""
        db = MagicMock()
        return db

    @pytest.fixture
    def search_service(self, mock_db):
        """创建 SearchService 实例"""
        return SearchService(mock_db)

    @pytest.mark.asyncio
    async def test_get_node_context_not_found(self, search_service, mock_db):
        """测试节点不存在"""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchone = MagicMock(return_value=None)
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_db.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_db.session.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await search_service.get_node_context(uuid.uuid4())
        assert result is None
