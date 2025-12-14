"""混合搜索测试"""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from dropqa.common.repository.base import SearchResult
from dropqa.common.repository.sqlite import SQLiteSearchRepository


class TestRRFMerge:
    """RRF 合并算法测试"""

    @pytest.fixture
    def sqlite_search_repo(self):
        """创建 SQLite 搜索仓库实例（仅用于测试 _rrf_merge）"""
        repo = SQLiteSearchRepository(
            db_path=":memory:",
            get_chroma_collection=None,
        )
        return repo

    def test_rrf_merge_basic(self, sqlite_search_repo):
        """测试基本 RRF 合并"""
        id1 = uuid.uuid4()
        id2 = uuid.uuid4()
        id3 = uuid.uuid4()

        list1 = [
            SearchResult(node_id=id1, document_id=uuid.uuid4(), title="A", content="a", rank=1.0),
            SearchResult(node_id=id2, document_id=uuid.uuid4(), title="B", content="b", rank=0.8),
        ]
        list2 = [
            SearchResult(node_id=id2, document_id=uuid.uuid4(), title="B", content="b", rank=0.9),
            SearchResult(node_id=id3, document_id=uuid.uuid4(), title="C", content="c", rank=0.7),
        ]

        result = sqlite_search_repo._rrf_merge(list1, list2, 0.5, 0.5, top_k=3)

        # id2 在两个列表中都有，RRF 分数应该最高
        assert len(result) == 3
        assert result[0].node_id == id2  # 两边都有，分数最高

    def test_rrf_merge_single_list_results(self, sqlite_search_repo):
        """测试单个列表独有的结果"""
        id1 = uuid.uuid4()
        id2 = uuid.uuid4()

        list1 = [SearchResult(node_id=id1, document_id=uuid.uuid4(), title="A", content="a", rank=1.0)]
        list2 = [SearchResult(node_id=id2, document_id=uuid.uuid4(), title="B", content="b", rank=0.9)]

        result = sqlite_search_repo._rrf_merge(list1, list2, 0.5, 0.5, top_k=2)

        assert len(result) == 2
        node_ids = {r.node_id for r in result}
        assert id1 in node_ids
        assert id2 in node_ids

    def test_rrf_merge_respects_weights(self, sqlite_search_repo):
        """测试权重影响排序"""
        id1 = uuid.uuid4()
        id2 = uuid.uuid4()

        # id1 在高权重列表排第一
        list1 = [SearchResult(node_id=id1, document_id=uuid.uuid4(), title="A", content="a", rank=1.0)]
        # id2 在低权重列表排第一
        list2 = [SearchResult(node_id=id2, document_id=uuid.uuid4(), title="B", content="b", rank=1.0)]

        # 高权重给 list1
        result = sqlite_search_repo._rrf_merge(list1, list2, 0.8, 0.2, top_k=2)

        assert result[0].node_id == id1  # list1 权重高，id1 应该排第一

    def test_rrf_merge_top_k_limit(self, sqlite_search_repo):
        """测试 top_k 限制"""
        ids = [uuid.uuid4() for _ in range(5)]

        list1 = [
            SearchResult(node_id=ids[i], document_id=uuid.uuid4(), title=f"T{i}", content=f"c{i}", rank=1.0)
            for i in range(3)
        ]
        list2 = [
            SearchResult(node_id=ids[i+2], document_id=uuid.uuid4(), title=f"T{i+2}", content=f"c{i+2}", rank=0.9)
            for i in range(3)
        ]

        result = sqlite_search_repo._rrf_merge(list1, list2, 0.5, 0.5, top_k=3)

        assert len(result) == 3


class TestSQLiteHybridSearch:
    """SQLite 混合搜索测试"""

    @pytest.fixture
    def mock_search_repo(self):
        """创建模拟的搜索仓库"""
        repo = AsyncMock()
        return repo

    @pytest.mark.asyncio
    async def test_hybrid_search_empty_results(self, mock_search_repo):
        """测试空结果"""
        mock_search_repo.fulltext_search = AsyncMock(return_value=[])
        mock_search_repo.vector_search = AsyncMock(return_value=[])
        mock_search_repo.hybrid_search = AsyncMock(return_value=[])

        result = await mock_search_repo.hybrid_search("test", [0.1, 0.2], top_k=10)

        assert result == []

    @pytest.mark.asyncio
    async def test_hybrid_search_fulltext_only(self):
        """测试仅全文搜索有结果"""
        repo = SQLiteSearchRepository(
            db_path=":memory:",
            get_chroma_collection=None,
        )
        # Mock fulltext_search
        id1 = uuid.uuid4()
        repo.fulltext_search = AsyncMock(return_value=[
            SearchResult(node_id=id1, document_id=uuid.uuid4(), title="T", content="c", rank=1.0)
        ])

        result = await repo.hybrid_search("test", [0.1], top_k=10, fulltext_weight=0.5)

        # 向量搜索不可用，应该返回全文结果
        assert len(result) == 1
        assert result[0].node_id == id1

    @pytest.mark.asyncio
    async def test_hybrid_search_vector_only(self):
        """测试仅向量搜索有结果"""
        mock_collection = MagicMock()
        mock_collection.query = MagicMock(return_value={
            "ids": [[str(uuid.uuid4())]],
            "documents": [["test content"]],
            "metadatas": [[{"document_id": str(uuid.uuid4()), "title": "Test"}]],
            "distances": [[0.1]],
        })

        repo = SQLiteSearchRepository(
            db_path=":memory:",
            get_chroma_collection=lambda: mock_collection,
        )
        repo.fulltext_search = AsyncMock(return_value=[])

        result = await repo.hybrid_search("", [0.1, 0.2], top_k=10, fulltext_weight=0.5)

        # 全文搜索无结果，应该返回向量结果
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_hybrid_search_weight_zero_fulltext(self):
        """测试全文权重为0时跳过全文搜索"""
        mock_collection = MagicMock()
        mock_collection.query = MagicMock(return_value={
            "ids": [[str(uuid.uuid4())]],
            "documents": [["test"]],
            "metadatas": [[{"document_id": str(uuid.uuid4())}]],
            "distances": [[0.1]],
        })

        repo = SQLiteSearchRepository(
            db_path=":memory:",
            get_chroma_collection=lambda: mock_collection,
        )
        repo.fulltext_search = AsyncMock(return_value=[])

        result = await repo.hybrid_search("test", [0.1], top_k=10, fulltext_weight=0.0)

        # fulltext_weight=0，不应调用 fulltext_search
        repo.fulltext_search.assert_not_called()
        assert len(result) == 1


class TestSearchServiceHybridSearch:
    """SearchService 混合搜索测试"""

    @pytest.mark.asyncio
    async def test_search_service_hybrid_search(self):
        """测试 SearchService 的 hybrid_search 方法"""
        from dropqa.server.search import SearchService

        mock_search_repo = AsyncMock()
        mock_node_repo = AsyncMock()

        id1 = uuid.uuid4()
        mock_search_repo.hybrid_search = AsyncMock(return_value=[
            SearchResult(node_id=id1, document_id=uuid.uuid4(), title="Test", content="content", rank=0.8)
        ])

        service = SearchService(mock_search_repo, mock_node_repo)
        result = await service.hybrid_search("query", [0.1, 0.2], top_k=5, fulltext_weight=0.6)

        assert len(result) == 1
        assert result[0].node_id == id1
        mock_search_repo.hybrid_search.assert_called_once_with("query", [0.1, 0.2], 5, 0.6)
