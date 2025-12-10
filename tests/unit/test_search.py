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
from dropqa.common.repository import SearchResult as RepoSearchResult
from dropqa.common.repository import NodeWithAncestors, AncestorInfo


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
    def mock_search_repo(self):
        """创建模拟搜索仓库"""
        repo = MagicMock()
        return repo

    @pytest.fixture
    def mock_node_repo(self):
        """创建模拟节点仓库"""
        repo = MagicMock()
        return repo

    @pytest.fixture
    def search_service(self, mock_search_repo, mock_node_repo):
        """创建 SearchService 实例"""
        return SearchService(mock_search_repo, mock_node_repo)

    def test_search_service_init(self, search_service, mock_search_repo, mock_node_repo):
        """测试 SearchService 初始化"""
        assert search_service._search_repo == mock_search_repo
        assert search_service._node_repo == mock_node_repo

    @pytest.mark.asyncio
    async def test_fulltext_search_returns_results(self, search_service, mock_search_repo):
        """测试全文搜索返回结果"""
        # 模拟仓库返回
        mock_results = [
            RepoSearchResult(
                node_id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                title="Chapter 1",
                content="This is about Python programming",
                rank=0.9,
            ),
            RepoSearchResult(
                node_id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                title="Chapter 2",
                content="More Python content",
                rank=0.7,
            ),
        ]
        mock_search_repo.fulltext_search = AsyncMock(return_value=mock_results)

        results = await search_service.fulltext_search("Python", top_k=10)

        assert len(results) == 2
        assert results[0].rank >= results[1].rank  # 按相关度排序
        mock_search_repo.fulltext_search.assert_called_once_with("Python", 10)

    @pytest.mark.asyncio
    async def test_fulltext_search_empty_results(self, search_service, mock_search_repo):
        """测试空结果"""
        mock_search_repo.fulltext_search = AsyncMock(return_value=[])

        results = await search_service.fulltext_search("nonexistent", top_k=10)

        assert results == []

    @pytest.mark.asyncio
    async def test_fulltext_search_respects_top_k(self, search_service, mock_search_repo):
        """测试 top_k 参数"""
        mock_results = [
            RepoSearchResult(
                node_id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                title=f"Title {i}",
                content=f"Content {i}",
                rank=0.9 - i * 0.1,
            )
            for i in range(3)
        ]
        mock_search_repo.fulltext_search = AsyncMock(return_value=mock_results)

        results = await search_service.fulltext_search("test", top_k=3)

        assert len(results) == 3
        mock_search_repo.fulltext_search.assert_called_once_with("test", 3)


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
    def mock_search_repo(self):
        """创建模拟搜索仓库"""
        repo = MagicMock()
        return repo

    @pytest.fixture
    def mock_node_repo(self):
        """创建模拟节点仓库"""
        repo = MagicMock()
        return repo

    @pytest.fixture
    def search_service(self, mock_search_repo, mock_node_repo):
        """创建 SearchService 实例"""
        return SearchService(mock_search_repo, mock_node_repo)

    @pytest.mark.asyncio
    async def test_get_node_context_not_found(self, search_service, mock_node_repo):
        """测试节点不存在"""
        mock_node_repo.get_with_ancestors = AsyncMock(return_value=None)

        result = await search_service.get_node_context(uuid.uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_get_node_context_found(self, search_service, mock_node_repo):
        """测试找到节点"""
        node_id = uuid.uuid4()
        mock_node_repo.get_with_ancestors = AsyncMock(return_value=NodeWithAncestors(
            node_id=node_id,
            content="This is the content",
            ancestors=[
                AncestorInfo(title="文档", summary=None, depth=0),
                AncestorInfo(title="第1章", summary="概述", depth=1),
                AncestorInfo(title="1.1 背景", summary="背景介绍", depth=2),
            ],
            document_name="report.md",
        ))

        result = await search_service.get_node_context(node_id)

        assert result is not None
        assert result.node_id == node_id
        assert result.content == "This is the content"
        assert len(result.breadcrumb) == 3
        assert result.document_name == "report.md"
        assert "第1章" in result.get_path_string()
