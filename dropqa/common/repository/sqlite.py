"""SQLite 后端实现

使用 SQLite + FTS5 实现轻量级存储方案。
"""

import re
from pathlib import Path
from typing import Optional
from uuid import UUID

import aiosqlite

from dropqa.common.config import SQLiteConfig
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
)


class SQLiteDocumentRepository(DocumentRepository):
    """SQLite 文档仓库"""

    def __init__(self, db_path: Path):
        self._db_path = db_path

    async def save(self, doc: DocumentData) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO documents (id, filename, file_type, file_hash, file_size, storage_path, current_version)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(doc.id),
                    doc.filename,
                    doc.file_type,
                    doc.file_hash,
                    doc.file_size,
                    doc.storage_path,
                    doc.current_version,
                ),
            )
            await db.commit()

    async def get_by_id(self, doc_id: UUID) -> Optional[DocumentData]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM documents WHERE id = ?", (str(doc_id),)
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                return self._row_to_data(row)

    async def get_by_path(self, storage_path: str) -> Optional[DocumentData]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM documents WHERE storage_path = ?", (storage_path,)
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                return self._row_to_data(row)

    async def get_by_hash(self, file_hash: str) -> Optional[DocumentData]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM documents WHERE file_hash = ?", (file_hash,)
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                return self._row_to_data(row)

    async def update(self, doc: DocumentData) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                UPDATE documents
                SET filename = ?, file_type = ?, file_hash = ?, file_size = ?,
                    storage_path = ?, current_version = ?
                WHERE id = ?
                """,
                (
                    doc.filename,
                    doc.file_type,
                    doc.file_hash,
                    doc.file_size,
                    doc.storage_path,
                    doc.current_version,
                    str(doc.id),
                ),
            )
            await db.commit()

    async def delete(self, doc_id: UUID) -> bool:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "DELETE FROM documents WHERE id = ?", (str(doc_id),)
            )
            await db.commit()
            return cursor.rowcount > 0

    async def delete_by_path(self, storage_path: str) -> bool:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "DELETE FROM documents WHERE storage_path = ?", (storage_path,)
            )
            await db.commit()
            return cursor.rowcount > 0

    def _row_to_data(self, row: aiosqlite.Row) -> DocumentData:
        return DocumentData(
            id=UUID(row["id"]),
            filename=row["filename"],
            file_type=row["file_type"],
            file_hash=row["file_hash"],
            file_size=row["file_size"],
            storage_path=row["storage_path"],
            current_version=row["current_version"],
        )


class SQLiteNodeRepository(NodeRepository):
    """SQLite 节点仓库"""

    def __init__(self, db_path: Path):
        self._db_path = db_path

    async def save_batch(self, nodes: list[NodeData]) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            for node in nodes:
                # 插入节点
                await db.execute(
                    """
                    INSERT INTO nodes (id, document_id, parent_id, node_type, depth, title, content, summary, position, version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(node.id),
                        str(node.document_id),
                        str(node.parent_id) if node.parent_id else None,
                        node.node_type,
                        node.depth,
                        node.title,
                        node.content,
                        node.summary,
                        node.position,
                        node.version,
                    ),
                )
                # 更新 FTS 索引
                await db.execute(
                    """
                    INSERT INTO nodes_fts (rowid, title, content, summary)
                    SELECT rowid, title, content, summary FROM nodes WHERE id = ?
                    """,
                    (str(node.id),),
                )
            await db.commit()

    async def delete_by_document(self, document_id: UUID) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            # 先删除 FTS 索引
            await db.execute(
                """
                DELETE FROM nodes_fts WHERE rowid IN (
                    SELECT rowid FROM nodes WHERE document_id = ?
                )
                """,
                (str(document_id),),
            )
            # 再删除节点
            await db.execute(
                "DELETE FROM nodes WHERE document_id = ?", (str(document_id),)
            )
            await db.commit()

    async def get_with_ancestors(self, node_id: UUID) -> Optional[NodeWithAncestors]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row

            # SQLite 递归 CTE 获取节点及其祖先
            sql = """
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
                    WHERE n.id = ?

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
            """

            async with db.execute(sql, (str(node_id),)) as cursor:
                rows = await cursor.fetchall()

            if not rows:
                return None

            target_node = None
            ancestors = []
            document_name = ""

            for row in rows:
                if row["level"] == 0:
                    target_node = row
                    document_name = row["document_name"] or "unknown"

                ancestors.append(
                    AncestorInfo(
                        title=row["title"] or "",
                        summary=row["summary"],
                        depth=row["depth"],
                    )
                )

            if target_node is None:
                return None

            return NodeWithAncestors(
                node_id=node_id,
                content=target_node["content"],
                ancestors=ancestors,
                document_name=document_name,
            )


class SQLiteSearchRepository(SearchRepository):
    """SQLite 搜索仓库（使用 FTS5）"""

    def __init__(self, db_path: Path):
        self._db_path = db_path

    async def fulltext_search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        query = query.strip()
        if not query:
            return []

        fts_query = self._build_fts_query(query)
        if not fts_query:
            return []

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row

            # 使用 FTS5 的 bm25 排名函数
            sql = """
                SELECT
                    n.id,
                    n.document_id,
                    n.title,
                    n.content,
                    bm25(nodes_fts, 1.0, 0.75, 0.5) AS rank
                FROM nodes_fts
                JOIN nodes n ON nodes_fts.rowid = n.rowid
                WHERE nodes_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """

            async with db.execute(sql, (fts_query, top_k)) as cursor:
                rows = await cursor.fetchall()

            return [
                SearchResult(
                    node_id=UUID(row["id"]),
                    document_id=UUID(row["document_id"]),
                    title=row["title"],
                    content=row["content"],
                    rank=abs(float(row["rank"])),  # bm25 返回负值，取绝对值
                )
                for row in rows
            ]

    async def vector_search(
        self,
        embedding: list[float],
        top_k: int = 10,
    ) -> list[SearchResult]:
        # TODO: 集成 ChromaDB 实现向量搜索
        raise NotImplementedError("SQLite 向量搜索需要集成 ChromaDB")

    async def hybrid_search(
        self,
        query: str,
        embedding: list[float],
        top_k: int = 10,
        fulltext_weight: float = 0.5,
    ) -> list[SearchResult]:
        # TODO: 实现混合搜索
        raise NotImplementedError("SQLite 混合搜索尚未实现")

    def _build_fts_query(self, query: str) -> str:
        """构建 FTS5 查询字符串

        将用户输入转换为 FTS5 查询格式。
        """
        # 移除特殊字符，保留字母、数字、中文
        cleaned = re.sub(r"[^\w\u4e00-\u9fff\s]", " ", query)

        # 分词
        words = cleaned.split()
        words = [w.strip() for w in words if w.strip()]

        if not words:
            return ""

        # FTS5 使用 AND 连接（默认行为），用双引号包裹每个词
        return " ".join(f'"{w}"' for w in words)


class SQLiteRepositoryFactory(RepositoryFactory):
    """SQLite 仓库工厂"""

    def __init__(self, config: SQLiteConfig):
        self._config = config
        self._db_path = config.get_db_path()
        self._document_repo = SQLiteDocumentRepository(self._db_path)
        self._node_repo = SQLiteNodeRepository(self._db_path)
        self._search_repo = SQLiteSearchRepository(self._db_path)

    def get_document_repository(self) -> DocumentRepository:
        return self._document_repo

    def get_node_repository(self) -> NodeRepository:
        return self._node_repo

    def get_search_repository(self) -> SearchRepository:
        return self._search_repo

    async def initialize(self) -> None:
        """初始化数据库（建表、索引）"""
        # 确保目录存在
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self._db_path) as db:
            # 创建文档表
            await db.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    storage_path TEXT NOT NULL UNIQUE,
                    current_version INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 创建节点表
            await db.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    parent_id TEXT,
                    node_type TEXT NOT NULL,
                    depth INTEGER NOT NULL,
                    title TEXT,
                    content TEXT,
                    summary TEXT,
                    position INTEGER DEFAULT 0,
                    version INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                    FOREIGN KEY (parent_id) REFERENCES nodes(id) ON DELETE CASCADE
                )
            """)

            # 创建索引
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_nodes_document_id ON nodes(document_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_nodes_parent_id ON nodes(parent_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_storage_path ON documents(storage_path)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash)"
            )

            # 创建 FTS5 虚拟表（全文搜索）
            await db.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
                    title,
                    content,
                    summary,
                    content='nodes',
                    content_rowid='rowid',
                    tokenize='unicode61'
                )
            """)

            # 启用外键约束
            await db.execute("PRAGMA foreign_keys = ON")

            await db.commit()

    async def close(self) -> None:
        """关闭连接（SQLite 不需要显式关闭）"""
        pass
