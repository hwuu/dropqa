"""Repository 模块测试"""

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dropqa.common.config import (
    PostgresConfig,
    SQLiteConfig,
    StorageConfig,
    StorageBackend,
    create_repository_factory,
)
from dropqa.common.repository import (
    DocumentData,
    NodeData,
    SearchResult,
    NodeWithAncestors,
    AncestorInfo,
    PostgresRepositoryFactory,
    SQLiteRepositoryFactory,
)


class TestDataClasses:
    """数据类测试"""

    def test_document_data_creation(self):
        """测试创建 DocumentData"""
        doc = DocumentData(
            id=uuid.uuid4(),
            filename="test.md",
            file_type="md",
            file_hash="abc123",
            file_size=1024,
            storage_path="/path/to/test.md",
        )
        assert doc.filename == "test.md"
        assert doc.current_version == 1

    def test_node_data_creation(self):
        """测试创建 NodeData"""
        node = NodeData(
            id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            parent_id=None,
            node_type="heading",
            depth=1,
            title="Chapter 1",
            content="Content here",
        )
        assert node.title == "Chapter 1"
        assert node.depth == 1

    def test_search_result_creation(self):
        """测试创建 SearchResult"""
        result = SearchResult(
            node_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            title="Title",
            content="Content",
            rank=0.85,
        )
        assert result.rank == 0.85

    def test_node_with_ancestors_creation(self):
        """测试创建 NodeWithAncestors"""
        ancestors = [
            AncestorInfo(title="Doc", summary=None, depth=0),
            AncestorInfo(title="Chapter 1", summary="Intro", depth=1),
        ]
        node = NodeWithAncestors(
            node_id=uuid.uuid4(),
            content="Content",
            ancestors=ancestors,
            document_name="test.md",
        )
        assert len(node.ancestors) == 2
        assert node.document_name == "test.md"


class TestStorageConfig:
    """存储配置测试"""

    def test_default_backend_is_postgres(self):
        """测试默认后端是 PostgreSQL"""
        config = StorageConfig()
        assert config.backend == StorageBackend.POSTGRES

    def test_postgres_config(self):
        """测试 PostgreSQL 配置"""
        config = StorageConfig(
            backend=StorageBackend.POSTGRES,
            postgres=PostgresConfig(
                host="db.example.com",
                port=5433,
                name="testdb",
            )
        )
        assert config.postgres.host == "db.example.com"
        assert config.postgres.port == 5433

    def test_sqlite_backend(self):
        """测试 SQLite 后端配置"""
        config = StorageConfig(backend=StorageBackend.SQLITE)
        assert config.backend == StorageBackend.SQLITE


class TestCreateRepositoryFactory:
    """create_repository_factory 测试"""

    def test_create_postgres_factory(self):
        """测试创建 PostgreSQL 工厂"""
        config = StorageConfig(
            backend=StorageBackend.POSTGRES,
            postgres=PostgresConfig(
                host="localhost",
                port=5432,
                name="testdb",
                user="testuser",
                password="testpass",
            )
        )
        factory = create_repository_factory(config)
        assert isinstance(factory, PostgresRepositoryFactory)

    def test_create_sqlite_factory(self, tmp_path: Path):
        """测试创建 SQLite 工厂"""
        config = StorageConfig(
            backend=StorageBackend.SQLITE,
            sqlite=SQLiteConfig(
                db_path=str(tmp_path / "test.db"),
                chroma_path=str(tmp_path / "chroma"),
            )
        )
        factory = create_repository_factory(config)
        assert isinstance(factory, SQLiteRepositoryFactory)


class TestPostgresRepositoryFactory:
    """PostgreSQL 仓库工厂测试"""

    @pytest.fixture
    def factory(self):
        """创建测试工厂"""
        config = PostgresConfig(
            host="localhost",
            port=5432,
            name="testdb",
            user="testuser",
            password="testpass",
        )
        return PostgresRepositoryFactory(config)

    def test_get_document_repository(self, factory):
        """测试获取文档仓库"""
        repo = factory.get_document_repository()
        assert repo is not None

    def test_get_node_repository(self, factory):
        """测试获取节点仓库"""
        repo = factory.get_node_repository()
        assert repo is not None

    def test_get_search_repository(self, factory):
        """测试获取搜索仓库"""
        repo = factory.get_search_repository()
        assert repo is not None

    def test_repositories_are_singletons(self, factory):
        """测试仓库是单例"""
        doc_repo1 = factory.get_document_repository()
        doc_repo2 = factory.get_document_repository()
        assert doc_repo1 is doc_repo2


class TestPostgresSearchRepository:
    """PostgreSQL 搜索仓库测试"""

    @pytest.fixture
    def factory(self):
        """创建测试工厂"""
        config = PostgresConfig(
            host="localhost",
            port=5432,
            name="testdb",
            user="testuser",
            password="testpass",
        )
        return PostgresRepositoryFactory(config)

    def test_build_tsquery_simple(self, factory):
        """测试构建简单查询"""
        search_repo = factory.get_search_repository()
        query = search_repo._build_tsquery("hello world")
        assert "hello" in query
        assert "world" in query
        assert "|" in query  # OR 查询，提高召回率

    def test_build_tsquery_chinese(self, factory):
        """测试构建中文查询"""
        search_repo = factory.get_search_repository()
        query = search_repo._build_tsquery("项目 预算")
        assert "项目" in query
        assert "预算" in query

    def test_build_tsquery_special_chars(self, factory):
        """测试特殊字符处理"""
        search_repo = factory.get_search_repository()
        query = search_repo._build_tsquery("test's & query")
        # 应该安全处理特殊字符
        assert query is not None
        assert "'" not in query

    def test_build_tsquery_empty(self, factory):
        """测试空查询"""
        search_repo = factory.get_search_repository()
        query = search_repo._build_tsquery("")
        assert query == ""

    def test_build_tsquery_whitespace_only(self, factory):
        """测试只有空白字符的查询"""
        search_repo = factory.get_search_repository()
        query = search_repo._build_tsquery("   ")
        assert query == ""


class TestSQLiteRepositoryFactory:
    """SQLite 仓库工厂测试"""

    @pytest.fixture
    def factory(self, tmp_path: Path):
        """创建测试工厂"""
        config = SQLiteConfig(
            db_path=str(tmp_path / "test.db"),
            chroma_path=str(tmp_path / "chroma"),
        )
        return SQLiteRepositoryFactory(config)

    def test_get_document_repository(self, factory):
        """测试获取文档仓库"""
        repo = factory.get_document_repository()
        assert repo is not None

    def test_get_node_repository(self, factory):
        """测试获取节点仓库"""
        repo = factory.get_node_repository()
        assert repo is not None

    def test_get_search_repository(self, factory):
        """测试获取搜索仓库"""
        repo = factory.get_search_repository()
        assert repo is not None

    def test_repositories_are_singletons(self, factory):
        """测试仓库是单例"""
        doc_repo1 = factory.get_document_repository()
        doc_repo2 = factory.get_document_repository()
        assert doc_repo1 is doc_repo2

    @pytest.mark.asyncio
    async def test_initialize_creates_tables(self, factory, tmp_path: Path):
        """测试初始化创建表"""
        await factory.initialize()
        db_path = tmp_path / "test.db"
        assert db_path.exists()


class TestSQLiteSearchRepository:
    """SQLite 搜索仓库测试"""

    @pytest.fixture
    def factory(self, tmp_path: Path):
        """创建测试工厂"""
        config = SQLiteConfig(
            db_path=str(tmp_path / "test.db"),
            chroma_path=str(tmp_path / "chroma"),
        )
        return SQLiteRepositoryFactory(config)

    def test_build_fts_query_simple(self, factory):
        """测试构建简单查询"""
        search_repo = factory.get_search_repository()
        query = search_repo._build_fts_query("hello world")
        assert '"hello"' in query
        assert '"world"' in query

    def test_build_fts_query_chinese(self, factory):
        """测试构建中文查询"""
        search_repo = factory.get_search_repository()
        query = search_repo._build_fts_query("项目 预算")
        assert '"项目"' in query
        assert '"预算"' in query

    def test_build_fts_query_special_chars(self, factory):
        """测试特殊字符处理"""
        search_repo = factory.get_search_repository()
        query = search_repo._build_fts_query("test's & query")
        # 应该安全处理特殊字符
        assert query is not None
        assert "'" not in query

    def test_build_fts_query_empty(self, factory):
        """测试空查询"""
        search_repo = factory.get_search_repository()
        query = search_repo._build_fts_query("")
        assert query == ""

    def test_build_fts_query_whitespace_only(self, factory):
        """测试只有空白字符的查询"""
        search_repo = factory.get_search_repository()
        query = search_repo._build_fts_query("   ")
        assert query == ""


class TestSQLiteDocumentRepository:
    """SQLite 文档仓库集成测试"""

    @pytest.fixture
    async def factory(self, tmp_path: Path):
        """创建并初始化测试工厂"""
        config = SQLiteConfig(
            db_path=str(tmp_path / "test.db"),
            chroma_path=str(tmp_path / "chroma"),
        )
        factory = SQLiteRepositoryFactory(config)
        await factory.initialize()
        return factory

    @pytest.mark.asyncio
    async def test_save_and_get_document(self, factory):
        """测试保存和获取文档"""
        doc_repo = factory.get_document_repository()
        doc_id = uuid.uuid4()
        doc = DocumentData(
            id=doc_id,
            filename="test.md",
            file_type="md",
            file_hash="abc123",
            file_size=1024,
            storage_path="/path/to/test.md",
        )
        await doc_repo.save(doc)

        # 通过 ID 获取
        result = await doc_repo.get_by_id(doc_id)
        assert result is not None
        assert result.filename == "test.md"
        assert result.file_hash == "abc123"

    @pytest.mark.asyncio
    async def test_get_by_path(self, factory):
        """测试通过路径获取文档"""
        doc_repo = factory.get_document_repository()
        doc = DocumentData(
            id=uuid.uuid4(),
            filename="test.md",
            file_type="md",
            file_hash="abc123",
            file_size=1024,
            storage_path="/path/to/test.md",
        )
        await doc_repo.save(doc)

        result = await doc_repo.get_by_path("/path/to/test.md")
        assert result is not None
        assert result.filename == "test.md"

    @pytest.mark.asyncio
    async def test_get_by_hash(self, factory):
        """测试通过哈希获取文档"""
        doc_repo = factory.get_document_repository()
        doc = DocumentData(
            id=uuid.uuid4(),
            filename="test.md",
            file_type="md",
            file_hash="unique_hash_123",
            file_size=1024,
            storage_path="/path/to/test.md",
        )
        await doc_repo.save(doc)

        result = await doc_repo.get_by_hash("unique_hash_123")
        assert result is not None
        assert result.filename == "test.md"

    @pytest.mark.asyncio
    async def test_update_document(self, factory):
        """测试更新文档"""
        doc_repo = factory.get_document_repository()
        doc_id = uuid.uuid4()
        doc = DocumentData(
            id=doc_id,
            filename="test.md",
            file_type="md",
            file_hash="abc123",
            file_size=1024,
            storage_path="/path/to/test.md",
        )
        await doc_repo.save(doc)

        # 更新
        doc.file_hash = "new_hash"
        doc.file_size = 2048
        await doc_repo.update(doc)

        result = await doc_repo.get_by_id(doc_id)
        assert result is not None
        assert result.file_hash == "new_hash"
        assert result.file_size == 2048

    @pytest.mark.asyncio
    async def test_delete_document(self, factory):
        """测试删除文档"""
        doc_repo = factory.get_document_repository()
        doc_id = uuid.uuid4()
        doc = DocumentData(
            id=doc_id,
            filename="test.md",
            file_type="md",
            file_hash="abc123",
            file_size=1024,
            storage_path="/path/to/test.md",
        )
        await doc_repo.save(doc)

        deleted = await doc_repo.delete(doc_id)
        assert deleted is True

        result = await doc_repo.get_by_id(doc_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_by_path(self, factory):
        """测试通过路径删除文档"""
        doc_repo = factory.get_document_repository()
        doc = DocumentData(
            id=uuid.uuid4(),
            filename="test.md",
            file_type="md",
            file_hash="abc123",
            file_size=1024,
            storage_path="/path/to/delete.md",
        )
        await doc_repo.save(doc)

        deleted = await doc_repo.delete_by_path("/path/to/delete.md")
        assert deleted is True

        result = await doc_repo.get_by_path("/path/to/delete.md")
        assert result is None


class TestSQLiteNodeRepository:
    """SQLite 节点仓库集成测试"""

    @pytest.fixture
    async def factory(self, tmp_path: Path):
        """创建并初始化测试工厂"""
        config = SQLiteConfig(
            db_path=str(tmp_path / "test.db"),
            chroma_path=str(tmp_path / "chroma"),
        )
        factory = SQLiteRepositoryFactory(config)
        await factory.initialize()
        return factory

    @pytest.mark.asyncio
    async def test_save_batch_nodes(self, factory):
        """测试批量保存节点"""
        doc_repo = factory.get_document_repository()
        node_repo = factory.get_node_repository()

        # 先创建文档
        doc_id = uuid.uuid4()
        doc = DocumentData(
            id=doc_id,
            filename="test.md",
            file_type="md",
            file_hash="abc123",
            file_size=1024,
            storage_path="/path/to/test.md",
        )
        await doc_repo.save(doc)

        # 创建节点
        root_id = uuid.uuid4()
        child_id = uuid.uuid4()
        nodes = [
            NodeData(
                id=root_id,
                document_id=doc_id,
                parent_id=None,
                node_type="document",
                depth=0,
                title="Document",
                content=None,
            ),
            NodeData(
                id=child_id,
                document_id=doc_id,
                parent_id=root_id,
                node_type="heading",
                depth=1,
                title="Chapter 1",
                content="Chapter content",
            ),
        ]
        await node_repo.save_batch(nodes)

        # 验证节点存在（通过获取上下文）
        result = await node_repo.get_with_ancestors(child_id)
        assert result is not None
        assert result.content == "Chapter content"

    @pytest.mark.asyncio
    async def test_get_with_ancestors(self, factory):
        """测试获取节点及祖先"""
        doc_repo = factory.get_document_repository()
        node_repo = factory.get_node_repository()

        # 创建文档
        doc_id = uuid.uuid4()
        doc = DocumentData(
            id=doc_id,
            filename="test.md",
            file_type="md",
            file_hash="abc123",
            file_size=1024,
            storage_path="/path/to/test.md",
        )
        await doc_repo.save(doc)

        # 创建三层节点
        root_id = uuid.uuid4()
        chapter_id = uuid.uuid4()
        section_id = uuid.uuid4()
        nodes = [
            NodeData(
                id=root_id,
                document_id=doc_id,
                parent_id=None,
                node_type="document",
                depth=0,
                title="Document",
                content=None,
            ),
            NodeData(
                id=chapter_id,
                document_id=doc_id,
                parent_id=root_id,
                node_type="heading",
                depth=1,
                title="Chapter 1",
                content="Chapter content",
            ),
            NodeData(
                id=section_id,
                document_id=doc_id,
                parent_id=chapter_id,
                node_type="heading",
                depth=2,
                title="Section 1.1",
                content="Section content",
            ),
        ]
        await node_repo.save_batch(nodes)

        # 获取最深节点的祖先
        result = await node_repo.get_with_ancestors(section_id)
        assert result is not None
        assert result.content == "Section content"
        assert len(result.ancestors) == 3
        assert result.document_name == "test.md"

    @pytest.mark.asyncio
    async def test_delete_by_document(self, factory):
        """测试删除文档的所有节点"""
        doc_repo = factory.get_document_repository()
        node_repo = factory.get_node_repository()

        # 创建文档
        doc_id = uuid.uuid4()
        doc = DocumentData(
            id=doc_id,
            filename="test.md",
            file_type="md",
            file_hash="abc123",
            file_size=1024,
            storage_path="/path/to/test.md",
        )
        await doc_repo.save(doc)

        # 创建节点
        node_id = uuid.uuid4()
        nodes = [
            NodeData(
                id=node_id,
                document_id=doc_id,
                parent_id=None,
                node_type="document",
                depth=0,
                title="Document",
                content=None,
            ),
        ]
        await node_repo.save_batch(nodes)

        # 删除文档的节点
        await node_repo.delete_by_document(doc_id)

        # 验证节点已删除
        result = await node_repo.get_with_ancestors(node_id)
        assert result is None


class TestSQLiteFulltextSearch:
    """SQLite 全文搜索集成测试"""

    @pytest.fixture
    async def factory(self, tmp_path: Path):
        """创建并初始化测试工厂"""
        config = SQLiteConfig(
            db_path=str(tmp_path / "test.db"),
            chroma_path=str(tmp_path / "chroma"),
        )
        factory = SQLiteRepositoryFactory(config)
        await factory.initialize()
        return factory

    @pytest.mark.asyncio
    async def test_fulltext_search_returns_results(self, factory):
        """测试全文搜索返回结果"""
        doc_repo = factory.get_document_repository()
        node_repo = factory.get_node_repository()
        search_repo = factory.get_search_repository()

        # 创建文档
        doc_id = uuid.uuid4()
        doc = DocumentData(
            id=doc_id,
            filename="test.md",
            file_type="md",
            file_hash="abc123",
            file_size=1024,
            storage_path="/path/to/test.md",
        )
        await doc_repo.save(doc)

        # 创建节点
        nodes = [
            NodeData(
                id=uuid.uuid4(),
                document_id=doc_id,
                parent_id=None,
                node_type="heading",
                depth=1,
                title="Python Programming",
                content="Learn Python programming language basics",
            ),
            NodeData(
                id=uuid.uuid4(),
                document_id=doc_id,
                parent_id=None,
                node_type="heading",
                depth=1,
                title="Java Programming",
                content="Learn Java programming language basics",
            ),
        ]
        await node_repo.save_batch(nodes)

        # 搜索
        results = await search_repo.fulltext_search("Python", top_k=10)
        assert len(results) >= 1
        assert any("Python" in (r.title or "") for r in results)

    @pytest.mark.asyncio
    async def test_fulltext_search_empty_query(self, factory):
        """测试空查询返回空结果"""
        search_repo = factory.get_search_repository()
        results = await search_repo.fulltext_search("", top_k=10)
        assert results == []

    @pytest.mark.asyncio
    async def test_fulltext_search_no_results(self, factory):
        """测试无匹配结果"""
        search_repo = factory.get_search_repository()
        results = await search_repo.fulltext_search("nonexistent_term_xyz", top_k=10)
        assert results == []


class TestSearchStrategy:
    """搜索策略测试"""

    @pytest.fixture
    async def factory(self, tmp_path: Path):
        """创建并初始化测试工厂"""
        config = SQLiteConfig(
            db_path=str(tmp_path / "test.db"),
            chroma_path=str(tmp_path / "chroma"),
        )
        factory = SQLiteRepositoryFactory(config)
        await factory.initialize()
        return factory

    @pytest.fixture
    async def populated_factory(self, factory):
        """创建已有数据的工厂"""
        doc_repo = factory.get_document_repository()
        node_repo = factory.get_node_repository()

        # 创建文档
        doc_id = uuid.uuid4()
        doc = DocumentData(
            id=doc_id,
            filename="test.md",
            file_type="md",
            file_hash="abc123",
            file_size=1024,
            storage_path="/path/to/test.md",
        )
        await doc_repo.save(doc)

        # 创建节点
        nodes = [
            NodeData(
                id=uuid.uuid4(),
                document_id=doc_id,
                parent_id=None,
                node_type="heading",
                depth=1,
                title="DropQA Introduction",
                content="DropQA is a document question answering system.",
            ),
            NodeData(
                id=uuid.uuid4(),
                document_id=doc_id,
                parent_id=None,
                node_type="heading",
                depth=1,
                title="Configuration Guide",
                content="This section covers PostgreSQL and SQLite configuration.",
            ),
        ]
        await node_repo.save_batch(nodes)

        return factory

    @pytest.mark.asyncio
    async def test_search_with_auto_strategy(self, populated_factory):
        """测试 auto 策略（默认使用 fulltext）"""
        search_repo = populated_factory.get_search_repository()
        results = await search_repo.search("DropQA", strategy="auto", top_k=10)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_search_with_fulltext_strategy(self, populated_factory):
        """测试 fulltext 策略"""
        search_repo = populated_factory.get_search_repository()
        results = await search_repo.search("DropQA", strategy="fulltext", top_k=10)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_search_with_keyword_strategy(self, populated_factory):
        """测试 keyword 策略"""
        search_repo = populated_factory.get_search_repository()
        results = await search_repo.search("DropQA", strategy="keyword", top_k=10)
        assert len(results) >= 1
        # 关键词搜索匹配的结果 rank 应该是 1.0
        assert all(r.rank == 1.0 for r in results)

    @pytest.mark.asyncio
    async def test_keyword_search_case_insensitive(self, populated_factory):
        """测试关键词搜索不区分大小写"""
        search_repo = populated_factory.get_search_repository()

        # 小写搜索
        results_lower = await search_repo.keyword_search("dropqa", top_k=10)
        # 大写搜索
        results_upper = await search_repo.keyword_search("DROPQA", top_k=10)
        # 混合大小写
        results_mixed = await search_repo.keyword_search("DropQA", top_k=10)

        # 都应该返回相同数量的结果
        assert len(results_lower) == len(results_upper) == len(results_mixed)
        assert len(results_lower) >= 1

    @pytest.mark.asyncio
    async def test_keyword_search_empty_query(self, populated_factory):
        """测试空关键词搜索"""
        search_repo = populated_factory.get_search_repository()
        results = await search_repo.keyword_search("", top_k=10)
        assert results == []

    @pytest.mark.asyncio
    async def test_keyword_search_partial_match(self, populated_factory):
        """测试关键词部分匹配"""
        search_repo = populated_factory.get_search_repository()
        # 搜索部分词
        results = await search_repo.keyword_search("SQL", top_k=10)
        # 应该匹配到包含 "SQLite" 或 "PostgreSQL" 的内容
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_search_regex_fallback_to_keyword(self, populated_factory):
        """测试 SQLite 的 regex 策略回退到 keyword"""
        search_repo = populated_factory.get_search_repository()
        # SQLite 不支持正则，应该回退到 keyword 搜索
        results = await search_repo.search("DropQA", strategy="regex", top_k=10)
        assert len(results) >= 1


class TestSQLiteVectorSearch:
    """SQLite + ChromaDB 向量搜索测试"""

    @pytest.fixture
    async def factory(self, tmp_path: Path):
        """创建并初始化测试工厂"""
        config = SQLiteConfig(
            db_path=str(tmp_path / "test.db"),
            chroma_path=str(tmp_path / "chroma"),
        )
        factory = SQLiteRepositoryFactory(config)
        await factory.initialize()
        return factory

    @pytest.fixture
    async def populated_factory(self, factory):
        """创建已有数据的工厂"""
        doc_repo = factory.get_document_repository()
        node_repo = factory.get_node_repository()

        # 创建文档
        doc_id = uuid.uuid4()
        doc = DocumentData(
            id=doc_id,
            filename="test.md",
            file_type="md",
            file_hash="abc123",
            file_size=1024,
            storage_path="/path/to/test.md",
        )
        await doc_repo.save(doc)

        # 创建节点
        self.node_ids = [uuid.uuid4(), uuid.uuid4()]
        nodes = [
            NodeData(
                id=self.node_ids[0],
                document_id=doc_id,
                parent_id=None,
                node_type="heading",
                depth=1,
                title="Python Programming",
                content="Learn Python programming language basics",
            ),
            NodeData(
                id=self.node_ids[1],
                document_id=doc_id,
                parent_id=None,
                node_type="heading",
                depth=1,
                title="Java Programming",
                content="Learn Java programming language basics",
            ),
        ]
        await node_repo.save_batch(nodes)

        self.doc_id = doc_id
        self.nodes_data = [
            NodeData(
                id=self.node_ids[0],
                document_id=doc_id,
                parent_id=None,
                node_type="heading",
                depth=1,
                title="Python Programming",
                content="Learn Python programming language basics",
            ),
            NodeData(
                id=self.node_ids[1],
                document_id=doc_id,
                parent_id=None,
                node_type="heading",
                depth=1,
                title="Java Programming",
                content="Learn Java programming language basics",
            ),
        ]

        return factory

    @pytest.mark.asyncio
    async def test_save_embeddings(self, populated_factory):
        """测试保存 embeddings 到 ChromaDB"""
        search_repo = populated_factory.get_search_repository()

        # 生成测试向量（384 维）
        embeddings = [
            [0.1] * 384,  # Python 节点的向量
            [0.2] * 384,  # Java 节点的向量
        ]

        # 保存 embeddings
        await search_repo.save_embeddings(
            self.nodes_data,
            embeddings,
            "test-model",
        )

        # 验证通过向量搜索能找到结果
        results = await search_repo.vector_search(
            embedding=[0.1] * 384,
            top_k=2,
        )
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_vector_search_returns_similar_results(self, populated_factory):
        """测试向量搜索返回相似结果"""
        search_repo = populated_factory.get_search_repository()

        # 保存不同的向量
        embeddings = [
            [1.0, 0.0, 0.0] + [0.0] * 381,  # Python 节点 - 更接近查询
            [0.0, 1.0, 0.0] + [0.0] * 381,  # Java 节点 - 较远
        ]

        await search_repo.save_embeddings(
            self.nodes_data,
            embeddings,
            "test-model",
        )

        # 搜索与 Python 节点相似的向量
        query_embedding = [1.0, 0.0, 0.0] + [0.0] * 381
        results = await search_repo.vector_search(
            embedding=query_embedding,
            top_k=2,
        )

        assert len(results) == 2
        # 第一个结果应该是 Python（最相似）
        assert results[0].node_id == self.node_ids[0]
        # rank 应该是相似度（1 - distance）
        assert results[0].rank > results[1].rank

    @pytest.mark.asyncio
    async def test_vector_search_empty_embedding(self, populated_factory):
        """测试空向量搜索"""
        search_repo = populated_factory.get_search_repository()
        results = await search_repo.vector_search(embedding=[], top_k=10)
        assert results == []

    @pytest.mark.asyncio
    async def test_delete_embeddings_by_document(self, populated_factory):
        """测试删除文档的所有 embeddings"""
        search_repo = populated_factory.get_search_repository()

        # 先保存 embeddings
        embeddings = [
            [0.1] * 384,
            [0.2] * 384,
        ]
        await search_repo.save_embeddings(
            self.nodes_data,
            embeddings,
            "test-model",
        )

        # 删除文档的 embeddings
        await search_repo.delete_embeddings_by_document(self.doc_id)

        # 验证 embeddings 已删除（搜索应该返回空）
        results = await search_repo.vector_search(
            embedding=[0.1] * 384,
            top_k=2,
        )
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_save_embeddings_upsert(self, populated_factory):
        """测试 upsert 更新已有的 embeddings"""
        search_repo = populated_factory.get_search_repository()

        # 第一次保存
        embeddings_v1 = [
            [0.1] * 384,
            [0.2] * 384,
        ]
        await search_repo.save_embeddings(
            self.nodes_data,
            embeddings_v1,
            "test-model",
        )

        # 第二次保存（upsert 更新）
        embeddings_v2 = [
            [0.9] * 384,  # 更新为不同的值
            [0.8] * 384,
        ]
        await search_repo.save_embeddings(
            self.nodes_data,
            embeddings_v2,
            "test-model-v2",
        )

        # 搜索应该找到新的向量
        results = await search_repo.vector_search(
            embedding=[0.9] * 384,
            top_k=2,
        )
        assert len(results) > 0
        # 第一个结果的 rank 应该很高（接近 1.0）
        assert results[0].rank > 0.9

    @pytest.mark.asyncio
    async def test_save_embeddings_mismatched_count(self, populated_factory):
        """测试节点数量与向量数量不匹配时报错"""
        search_repo = populated_factory.get_search_repository()

        # 只提供一个向量，但有两个节点
        embeddings = [
            [0.1] * 384,
        ]

        with pytest.raises(ValueError, match="不匹配"):
            await search_repo.save_embeddings(
                self.nodes_data,
                embeddings,
                "test-model",
            )

    @pytest.mark.asyncio
    async def test_save_embeddings_empty_input(self, populated_factory):
        """测试空输入不报错"""
        search_repo = populated_factory.get_search_repository()

        # 空节点列表
        await search_repo.save_embeddings([], [], "test-model")

        # 空向量列表
        await search_repo.save_embeddings([], [], "test-model")
