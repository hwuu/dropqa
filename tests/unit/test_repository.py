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
