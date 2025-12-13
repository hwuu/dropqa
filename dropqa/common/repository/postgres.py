"""PostgreSQL 后端实现"""

import logging
import re
from typing import Optional
from uuid import UUID

from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from dropqa.common.config import PostgresConfig
from dropqa.common.repository.base import (
    AncestorInfo,
    DocumentData,
    DocumentRepository,
    NodeData,
    NodeRepository,
    NodeWithAncestors,
    RepositoryFactory,
    SearchRepository,
    SearchResult,
    SearchStrategy,
)

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy 基类"""
    pass


# 导入模型以注册到 Base
from dropqa.common.models import Document, Node


class PostgresDocumentRepository(DocumentRepository):
    """PostgreSQL 文档仓库"""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory

    async def save(self, doc: DocumentData) -> None:
        async with self._session_factory() as session:
            db_doc = Document(
                id=doc.id,
                filename=doc.filename,
                file_type=doc.file_type,
                file_hash=doc.file_hash,
                file_size=doc.file_size,
                storage_path=doc.storage_path,
                current_version=doc.current_version,
            )
            session.add(db_doc)
            await session.commit()

    async def get_by_id(self, doc_id: UUID) -> Optional[DocumentData]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(Document).where(Document.id == doc_id)
            )
            doc = result.scalar_one_or_none()
            if doc is None:
                return None
            return self._to_data(doc)

    async def get_by_path(self, storage_path: str) -> Optional[DocumentData]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(Document).where(Document.storage_path == storage_path)
            )
            doc = result.scalar_one_or_none()
            if doc is None:
                return None
            return self._to_data(doc)

    async def get_by_hash(self, file_hash: str) -> Optional[DocumentData]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(Document).where(Document.file_hash == file_hash)
            )
            doc = result.scalar_one_or_none()
            if doc is None:
                return None
            return self._to_data(doc)

    async def update(self, doc: DocumentData) -> None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(Document).where(Document.id == doc.id)
            )
            db_doc = result.scalar_one_or_none()
            if db_doc:
                db_doc.filename = doc.filename
                db_doc.file_type = doc.file_type
                db_doc.file_hash = doc.file_hash
                db_doc.file_size = doc.file_size
                db_doc.storage_path = doc.storage_path
                db_doc.current_version = doc.current_version
                await session.commit()

    async def delete(self, doc_id: UUID) -> bool:
        async with self._session_factory() as session:
            result = await session.execute(
                delete(Document).where(Document.id == doc_id)
            )
            await session.commit()
            return result.rowcount > 0

    async def delete_by_path(self, storage_path: str) -> bool:
        async with self._session_factory() as session:
            result = await session.execute(
                delete(Document).where(Document.storage_path == storage_path)
            )
            await session.commit()
            return result.rowcount > 0

    def _to_data(self, doc: Document) -> DocumentData:
        return DocumentData(
            id=doc.id,
            filename=doc.filename,
            file_type=doc.file_type,
            file_hash=doc.file_hash,
            file_size=doc.file_size,
            storage_path=doc.storage_path,
            current_version=doc.current_version,
        )


class PostgresNodeRepository(NodeRepository):
    """PostgreSQL 节点仓库"""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory

    async def save_batch(self, nodes: list[NodeData]) -> None:
        async with self._session_factory() as session:
            for node_data in nodes:
                db_node = Node(
                    id=node_data.id,
                    document_id=node_data.document_id,
                    parent_id=node_data.parent_id,
                    node_type=node_data.node_type,
                    depth=node_data.depth,
                    title=node_data.title,
                    content=node_data.content,
                    summary=node_data.summary,
                    position=node_data.position,
                    version=node_data.version,
                )
                session.add(db_node)
            await session.commit()

    async def delete_by_document(self, document_id: UUID) -> None:
        async with self._session_factory() as session:
            await session.execute(
                delete(Node).where(Node.document_id == document_id)
            )
            await session.commit()

    async def get_with_ancestors(self, node_id: UUID) -> Optional[NodeWithAncestors]:
        sql = text("""
            WITH RECURSIVE ancestors AS (
                SELECT
                    n.id,
                    n.parent_id,
                    n.document_id,
                    n.title,
                    n.content,
                    n.summary,
                    n.depth,
                    n.node_type,
                    0 AS level
                FROM nodes n
                WHERE n.id = :node_id

                UNION ALL

                SELECT
                    p.id,
                    p.parent_id,
                    p.document_id,
                    p.title,
                    p.content,
                    p.summary,
                    p.depth,
                    p.node_type,
                    a.level + 1
                FROM nodes p
                INNER JOIN ancestors a ON p.id = a.parent_id
            )
            SELECT
                a.*,
                d.filename AS document_name
            FROM ancestors a
            LEFT JOIN documents d ON a.document_id = d.id
            ORDER BY a.depth ASC
        """)

        async with self._session_factory() as session:
            result = await session.execute(sql, {"node_id": str(node_id)})
            rows = result.fetchall()

        if not rows:
            return None

        target_node = None
        ancestors = []
        document_name = ""

        for row in rows:
            if row.level == 0:
                target_node = row
                document_name = row.document_name or "unknown"

            ancestors.append(AncestorInfo(
                title=row.title or "",
                summary=row.summary,
                depth=row.depth,
            ))

        if target_node is None:
            return None

        return NodeWithAncestors(
            node_id=node_id,
            content=target_node.content,
            ancestors=ancestors,
            document_name=document_name,
        )


class PostgresSearchRepository(SearchRepository):
    """PostgreSQL 搜索仓库"""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory

    async def search(
        self,
        query: str,
        strategy: SearchStrategy = "auto",
        top_k: int = 10,
    ) -> list[SearchResult]:
        """统一搜索入口"""
        query = query.strip()
        if not query:
            return []

        if strategy == "auto" or strategy == "fulltext":
            return await self.fulltext_search(query, top_k)
        elif strategy == "keyword":
            return await self.keyword_search(query, top_k)
        elif strategy == "regex":
            return await self.regex_search(query, top_k)
        else:
            return await self.fulltext_search(query, top_k)

    async def keyword_search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """关键词搜索（精确匹配，不区分大小写）"""
        query = query.strip()
        if not query:
            return []

        logger.debug(f"[PostgresKeyword] 关键词搜索: '{query}'")

        # 使用 ILIKE 进行不区分大小写的匹配
        # 搜索 title 和 content 字段
        sql = text("""
            SELECT
                n.id,
                n.document_id,
                n.title,
                n.content,
                1.0 AS rank
            FROM nodes n
            WHERE n.title ILIKE :pattern OR n.content ILIKE :pattern
            LIMIT :top_k
        """)

        pattern = f"%{query}%"

        async with self._session_factory() as session:
            result = await session.execute(
                sql,
                {"pattern": pattern, "top_k": top_k},
            )
            rows = result.fetchall()

        logger.debug(f"[PostgresKeyword] 匹配到 {len(rows)} 条结果")

        return [
            SearchResult(
                node_id=row.id,
                document_id=row.document_id,
                title=row.title,
                content=row.content,
                rank=float(row.rank),
            )
            for row in rows
        ]

    async def regex_search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """正则表达式搜索（仅 PostgreSQL 支持）"""
        query = query.strip()
        if not query:
            return []

        logger.debug(f"[PostgresRegex] 正则搜索: '{query}'")

        # 使用 PostgreSQL 的 ~* 操作符（不区分大小写的正则匹配）
        sql = text("""
            SELECT
                n.id,
                n.document_id,
                n.title,
                n.content,
                1.0 AS rank
            FROM nodes n
            WHERE n.title ~* :pattern OR n.content ~* :pattern
            LIMIT :top_k
        """)

        async with self._session_factory() as session:
            result = await session.execute(
                sql,
                {"pattern": query, "top_k": top_k},
            )
            rows = result.fetchall()

        logger.debug(f"[PostgresRegex] 匹配到 {len(rows)} 条结果")

        return [
            SearchResult(
                node_id=row.id,
                document_id=row.document_id,
                title=row.title,
                content=row.content,
                rank=float(row.rank),
            )
            for row in rows
        ]

    async def fulltext_search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        query = query.strip()
        if not query:
            return []

        tsquery = self._build_tsquery(query)
        if not tsquery:
            return []

        logger.debug(f"[PostgresSearch] 原始查询: '{query}' -> tsquery: '{tsquery}'")

        sql = text("""
            SELECT
                n.id,
                n.document_id,
                n.title,
                n.content,
                ts_rank(n.search_vector, to_tsquery('simple', :tsquery)) AS rank
            FROM nodes n
            WHERE n.search_vector @@ to_tsquery('simple', :tsquery)
            ORDER BY rank DESC
            LIMIT :top_k
        """)

        async with self._session_factory() as session:
            result = await session.execute(
                sql,
                {"tsquery": tsquery, "top_k": top_k},
            )
            rows = result.fetchall()

        logger.debug(f"[PostgresSearch] 匹配到 {len(rows)} 条结果")

        return [
            SearchResult(
                node_id=row.id,
                document_id=row.document_id,
                title=row.title,
                content=row.content,
                rank=float(row.rank),
            )
            for row in rows
        ]

    async def vector_search(
        self,
        embedding: list[float],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """向量搜索（使用 pgvector 余弦相似度）"""
        if not embedding:
            return []

        logger.debug(f"[PostgresVector] 向量搜索，维度: {len(embedding)}, top_k: {top_k}")

        # 将向量转换为 PostgreSQL 格式
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

        sql = text("""
            SELECT
                n.id,
                n.document_id,
                n.title,
                n.content,
                1 - (e.embedding <=> :embedding::vector) AS rank
            FROM embeddings e
            JOIN nodes n ON e.node_id = n.id
            ORDER BY e.embedding <=> :embedding::vector
            LIMIT :top_k
        """)

        async with self._session_factory() as session:
            result = await session.execute(
                sql,
                {"embedding": embedding_str, "top_k": top_k},
            )
            rows = result.fetchall()

        logger.debug(f"[PostgresVector] 匹配到 {len(rows)} 条结果")

        return [
            SearchResult(
                node_id=row.id,
                document_id=row.document_id,
                title=row.title,
                content=row.content,
                rank=float(row.rank) if row.rank else 0.0,
            )
            for row in rows
        ]

    async def hybrid_search(
        self,
        query: str,
        embedding: list[float],
        top_k: int = 10,
        fulltext_weight: float = 0.5,
    ) -> list[SearchResult]:
        # TODO: 实现混合搜索
        raise NotImplementedError("PostgreSQL 混合搜索尚未实现")

    async def save_embeddings(
        self,
        nodes: list["NodeData"],
        embeddings: list[list[float]],
        model_name: str,
    ) -> None:
        """保存节点的 embedding"""
        if not nodes or not embeddings:
            return

        if len(nodes) != len(embeddings):
            raise ValueError(f"节点数量 ({len(nodes)}) 与向量数量 ({len(embeddings)}) 不匹配")

        logger.debug(f"[PostgresEmbedding] 保存 {len(nodes)} 个节点的向量")

        async with self._session_factory() as session:
            for node, embedding in zip(nodes, embeddings):
                # 将向量转换为 PostgreSQL 格式
                embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

                # 使用 INSERT ... ON CONFLICT 实现 upsert
                sql = text("""
                    INSERT INTO embeddings (id, node_id, embedding, model_name)
                    VALUES (gen_random_uuid(), :node_id, :embedding::vector, :model_name)
                    ON CONFLICT (node_id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        model_name = EXCLUDED.model_name
                """)

                await session.execute(
                    sql,
                    {
                        "node_id": str(node.id),
                        "embedding": embedding_str,
                        "model_name": model_name,
                    },
                )

            await session.commit()

        logger.debug(f"[PostgresEmbedding] 保存完成")

    async def delete_embeddings_by_document(self, document_id: "UUID") -> None:
        """删除文档的所有 embedding"""
        logger.debug(f"[PostgresEmbedding] 删除文档 {document_id} 的所有向量")

        sql = text("""
            DELETE FROM embeddings
            WHERE node_id IN (
                SELECT id FROM nodes WHERE document_id = :document_id
            )
        """)

        async with self._session_factory() as session:
            await session.execute(sql, {"document_id": str(document_id)})
            await session.commit()

    def _build_tsquery(self, query: str) -> str:
        """构建 tsquery 查询字符串

        使用 OR 操作符，只要匹配任意一个词就返回结果。
        这样用户问 "DropQA 是什么？" 时，即使 "是什么" 不在文档中，
        也能匹配到包含 "DropQA" 的文档。
        """
        cleaned = re.sub(r"[^\w\u4e00-\u9fff\s]", " ", query)
        words = cleaned.split()
        words = [w.strip() for w in words if w.strip()]
        if not words:
            return ""
        return " | ".join(words)


class PostgresRepositoryFactory(RepositoryFactory):
    """PostgreSQL 仓库工厂"""

    def __init__(self, config: PostgresConfig):
        self._config = config
        self._engine = create_async_engine(
            config.url,
            echo=False,
            pool_pre_ping=True,
        )
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        self._document_repo = PostgresDocumentRepository(self._session_factory)
        self._node_repo = PostgresNodeRepository(self._session_factory)
        self._search_repo = PostgresSearchRepository(self._session_factory)

    def get_document_repository(self) -> DocumentRepository:
        return self._document_repo

    def get_node_repository(self) -> NodeRepository:
        return self._node_repo

    def get_search_repository(self) -> SearchRepository:
        return self._search_repo

    async def initialize(self) -> None:
        """初始化数据库（建表、索引）"""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # 创建全文搜索索引 - 拆分成单独的语句执行
        fulltext_statements = [
            # 1. 添加全文搜索向量列（如果不存在）
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'nodes' AND column_name = 'search_vector'
                ) THEN
                    ALTER TABLE nodes ADD COLUMN search_vector tsvector;
                END IF;
            END $$
            """,
            # 2. 创建或替换更新搜索向量的函数
            """
            CREATE OR REPLACE FUNCTION update_nodes_search_vector()
            RETURNS trigger AS $$
            BEGIN
                NEW.search_vector :=
                    setweight(to_tsvector('simple', COALESCE(NEW.title, '')), 'A') ||
                    setweight(to_tsvector('simple', COALESCE(NEW.content, '')), 'B') ||
                    setweight(to_tsvector('simple', COALESCE(NEW.summary, '')), 'C');
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql
            """,
            # 3. 删除旧触发器（如果存在）
            "DROP TRIGGER IF EXISTS nodes_search_vector_update ON nodes",
            # 4. 创建触发器
            """
            CREATE TRIGGER nodes_search_vector_update
                BEFORE INSERT OR UPDATE ON nodes
                FOR EACH ROW EXECUTE FUNCTION update_nodes_search_vector()
            """,
            # 5. 创建 GIN 索引（如果不存在）
            "CREATE INDEX IF NOT EXISTS nodes_search_idx ON nodes USING gin(search_vector)",
            # 6. 更新现有数据的搜索向量
            """
            UPDATE nodes SET search_vector =
                setweight(to_tsvector('simple', COALESCE(title, '')), 'A') ||
                setweight(to_tsvector('simple', COALESCE(content, '')), 'B') ||
                setweight(to_tsvector('simple', COALESCE(summary, '')), 'C')
            WHERE search_vector IS NULL
            """,
        ]

        async with self._engine.begin() as conn:
            for stmt in fulltext_statements:
                await conn.execute(text(stmt))

        # pgvector 初始化语句
        vector_statements = [
            # 1. 启用 pgvector 扩展
            "CREATE EXTENSION IF NOT EXISTS vector",
            # 2. 添加 embedding 列（如果不存在）
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'embeddings' AND column_name = 'embedding'
                ) THEN
                    ALTER TABLE embeddings ADD COLUMN embedding vector;
                END IF;
            END $$
            """,
            # 3. 添加 node_id 唯一约束（用于 upsert）
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname = 'embeddings_node_id_key'
                ) THEN
                    ALTER TABLE embeddings ADD CONSTRAINT embeddings_node_id_key UNIQUE (node_id);
                END IF;
            END $$
            """,
            # 4. 创建 HNSW 索引（如果不存在）
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_indexes
                    WHERE indexname = 'embeddings_embedding_idx'
                ) THEN
                    CREATE INDEX embeddings_embedding_idx ON embeddings USING hnsw (embedding vector_cosine_ops);
                END IF;
            END $$
            """,
        ]

        async with self._engine.begin() as conn:
            for stmt in vector_statements:
                await conn.execute(text(stmt))

    async def close(self) -> None:
        """关闭连接"""
        await self._engine.dispose()
