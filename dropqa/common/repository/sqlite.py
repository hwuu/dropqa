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
    SearchStrategy,
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

            # 获取节点内容，如果为空则聚合子节点内容
            content = target_node["content"]
            if not content:
                content = await self._get_children_content(db, node_id)

            return NodeWithAncestors(
                node_id=node_id,
                content=content,
                ancestors=ancestors,
                document_name=document_name,
            )

    async def _get_children_content(
        self,
        db: aiosqlite.Connection,
        parent_id: UUID,
        max_depth: int = 3,
    ) -> str:
        """递归获取所有子节点的内容并聚合

        Args:
            db: 数据库连接
            parent_id: 父节点 ID
            max_depth: 最大递归深度

        Returns:
            聚合后的子节点内容
        """
        # 使用递归 CTE 获取所有子节点，按 position 排序
        sql = """
            WITH RECURSIVE descendants AS (
                SELECT
                    n.id,
                    n.parent_id,
                    n.title,
                    n.content,
                    n.position,
                    n.depth,
                    1 AS level
                FROM nodes n
                WHERE n.parent_id = ?

                UNION ALL

                SELECT
                    c.id,
                    c.parent_id,
                    c.title,
                    c.content,
                    c.position,
                    c.depth,
                    d.level + 1
                FROM nodes c
                INNER JOIN descendants d ON c.parent_id = d.id
                WHERE d.level < ?
            )
            SELECT title, content, depth, position
            FROM descendants
            ORDER BY depth, position
        """

        async with db.execute(sql, (str(parent_id), max_depth)) as cursor:
            rows = await cursor.fetchall()

        if not rows:
            return ""

        # 聚合子节点内容
        parts = []
        for row in rows:
            if row[1]:  # content
                parts.append(row[1])
            elif row[0]:  # title (如果没有 content 但有 title)
                parts.append(row[0])

        return "\n\n".join(parts)


class SQLiteSearchRepository(SearchRepository):
    """SQLite 搜索仓库（使用 FTS5 + ChromaDB）"""

    def __init__(self, db_path: Path, get_chroma_collection=None):
        self._db_path = db_path
        self._get_chroma_collection = get_chroma_collection

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
            # SQLite 不支持正则搜索，回退到关键词搜索
            return await self.keyword_search(query, top_k)
        else:
            return await self.fulltext_search(query, top_k)

    async def keyword_search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """关键词搜索（精确匹配，不区分大小写）"""
        query = query.strip()
        if not query:
            return []

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row

            # SQLite 的 LIKE 默认不区分大小写（对 ASCII 字符）
            # 对于中文，LIKE 本身就是精确匹配
            sql = """
                SELECT
                    n.id,
                    n.document_id,
                    n.title,
                    n.content,
                    1.0 AS rank
                FROM nodes n
                WHERE LOWER(n.title) LIKE LOWER(?) OR LOWER(n.content) LIKE LOWER(?)
                LIMIT ?
            """

            pattern = f"%{query}%"

            async with db.execute(sql, (pattern, pattern, top_k)) as cursor:
                rows = await cursor.fetchall()

            return [
                SearchResult(
                    node_id=UUID(row["id"]),
                    document_id=UUID(row["document_id"]),
                    title=row["title"],
                    content=row["content"],
                    rank=float(row["rank"]),
                )
                for row in rows
            ]

    async def fulltext_search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """全文搜索（FTS5 + 标题 LIKE 补充）

        结合 FTS5 全文搜索和标题 LIKE 搜索，确保标题节点能被召回。
        """
        query = query.strip()
        if not query:
            return []

        results: list[SearchResult] = []
        seen_ids: set[UUID] = set()

        # 1. FTS5 全文搜索
        fts_query = self._build_fts_query(query)
        if fts_query:
            fts_results = await self._fts_search(fts_query, top_k)
            for r in fts_results:
                if r.node_id not in seen_ids:
                    results.append(r)
                    seen_ids.add(r.node_id)

        # 2. 标题 LIKE 搜索（补充 FTS 可能遗漏的标题匹配）
        # 提取主要关键词（过滤掉常见疑问词）
        keywords = self._extract_keywords(query)
        if keywords:
            title_results = await self._title_search(keywords, top_k)
            for r in title_results:
                if r.node_id not in seen_ids:
                    results.append(r)
                    seen_ids.add(r.node_id)

        # 按 rank 排序，返回 top_k
        results.sort(key=lambda x: x.rank, reverse=True)
        return results[:top_k]

    async def _fts_search(self, fts_query: str, top_k: int) -> list[SearchResult]:
        """FTS5 全文搜索"""
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

    async def _title_search(self, keywords: list[str], top_k: int) -> list[SearchResult]:
        """标题 LIKE 搜索（大小写不敏感）

        用于补充 FTS 可能遗漏的标题匹配，特别是处理大小写不一致的情况。
        """
        if not keywords:
            return []

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row

            # 构建 OR 条件：任意关键词匹配标题即可
            conditions = " OR ".join("LOWER(n.title) LIKE LOWER(?)" for _ in keywords)
            patterns = [f"%{kw}%" for kw in keywords]

            sql = f"""
                SELECT
                    n.id,
                    n.document_id,
                    n.title,
                    n.content,
                    10.0 AS rank
                FROM nodes n
                WHERE ({conditions}) AND n.title IS NOT NULL
                LIMIT ?
            """

            async with db.execute(sql, (*patterns, top_k)) as cursor:
                rows = await cursor.fetchall()

            return [
                SearchResult(
                    node_id=UUID(row["id"]),
                    document_id=UUID(row["document_id"]),
                    title=row["title"],
                    content=row["content"],
                    rank=float(row["rank"]),
                )
                for row in rows
            ]

    def _extract_keywords(self, query: str) -> list[str]:
        """从查询中提取主要关键词（过滤常见疑问词）"""
        # 常见的中文疑问词/停用词
        stopwords = {"是什么", "什么", "怎么", "如何", "为什么", "哪些", "哪个", "有哪些", "是", "的", "吗", "呢"}

        # 清理并分词
        cleaned = re.sub(r"[^\w\u4e00-\u9fff\s]", " ", query)
        words = cleaned.split()
        words = [w.strip() for w in words if w.strip()]

        # 过滤停用词，保留实质性关键词
        keywords = [w for w in words if w not in stopwords and len(w) > 1]

        return keywords

    async def vector_search(
        self,
        embedding: list[float],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """向量搜索（使用 ChromaDB）"""
        if not embedding:
            return []

        if self._get_chroma_collection is None:
            raise NotImplementedError("SQLite 向量搜索需要配置 ChromaDB")

        collection = self._get_chroma_collection()

        # ChromaDB 查询
        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # 转换结果
        search_results = []
        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            documents = results["documents"][0] if results["documents"] else [None] * len(ids)
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
            distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)

            for i, node_id in enumerate(ids):
                metadata = metadatas[i] if i < len(metadatas) else {}
                # ChromaDB 返回的是距离，转换为相似度（余弦距离: similarity = 1 - distance）
                similarity = 1 - distances[i] if i < len(distances) else 0.0

                search_results.append(SearchResult(
                    node_id=UUID(node_id),
                    document_id=UUID(metadata.get("document_id", node_id)),
                    title=metadata.get("title"),
                    content=documents[i] if i < len(documents) else None,
                    rank=similarity,
                ))

        return search_results

    async def hybrid_search(
        self,
        query: str,
        embedding: list[float],
        top_k: int = 10,
        fulltext_weight: float = 0.5,
    ) -> list[SearchResult]:
        """混合搜索（全文 + 向量）

        使用 RRF (Reciprocal Rank Fusion) 算法合并全文搜索和向量搜索结果。
        RRF 公式: score = w1 * (1 / (k + rank1)) + w2 * (1 / (k + rank2))
        其中 k 是常数（通常为 60），用于减轻排名靠后的影响。

        Args:
            query: 文本查询
            embedding: 查询向量
            top_k: 返回结果数量
            fulltext_weight: 全文搜索权重 (0.0-1.0)

        Returns:
            搜索结果列表
        """
        vector_weight = 1.0 - fulltext_weight

        # 并行执行两种搜索，获取更多候选结果用于合并
        candidate_k = top_k * 2

        # 执行全文搜索
        fulltext_results = []
        if fulltext_weight > 0:
            fulltext_results = await self.fulltext_search(query, candidate_k)

        # 执行向量搜索
        vector_results = []
        if vector_weight > 0 and embedding:
            try:
                vector_results = await self.vector_search(embedding, candidate_k)
            except NotImplementedError:
                # 如果向量搜索不可用，回退到纯全文搜索
                pass

        # 如果只有一种搜索有结果，直接返回
        if not fulltext_results and not vector_results:
            return []
        if not fulltext_results:
            return vector_results[:top_k]
        if not vector_results:
            return fulltext_results[:top_k]

        # 使用 RRF 合并结果
        return self._rrf_merge(
            fulltext_results,
            vector_results,
            fulltext_weight,
            vector_weight,
            top_k,
        )

    def _rrf_merge(
        self,
        list1: list[SearchResult],
        list2: list[SearchResult],
        weight1: float,
        weight2: float,
        top_k: int,
        k: int = 60,
    ) -> list[SearchResult]:
        """RRF (Reciprocal Rank Fusion) 合并两个排序列表

        Args:
            list1: 第一个搜索结果列表
            list2: 第二个搜索结果列表
            weight1: 第一个列表的权重
            weight2: 第二个列表的权重
            top_k: 返回结果数量
            k: RRF 常数，用于平滑排名

        Returns:
            合并后的搜索结果列表
        """
        # 计算 RRF 分数
        scores: dict[UUID, float] = {}
        results_map: dict[UUID, SearchResult] = {}

        # 处理第一个列表
        for rank, result in enumerate(list1):
            node_id = result.node_id
            rrf_score = weight1 * (1.0 / (k + rank + 1))
            scores[node_id] = scores.get(node_id, 0) + rrf_score
            if node_id not in results_map:
                results_map[node_id] = result

        # 处理第二个列表
        for rank, result in enumerate(list2):
            node_id = result.node_id
            rrf_score = weight2 * (1.0 / (k + rank + 1))
            scores[node_id] = scores.get(node_id, 0) + rrf_score
            if node_id not in results_map:
                results_map[node_id] = result

        # 按 RRF 分数排序
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # 构建最终结果
        merged_results = []
        for node_id in sorted_ids[:top_k]:
            result = results_map[node_id]
            # 使用 RRF 分数作为最终 rank
            merged_results.append(SearchResult(
                node_id=result.node_id,
                document_id=result.document_id,
                title=result.title,
                content=result.content,
                rank=scores[node_id],
            ))

        return merged_results

    async def save_embeddings(
        self,
        nodes: list["NodeData"],
        embeddings: list[list[float]],
        model_name: str,
    ) -> None:
        """保存节点的 embedding 到 ChromaDB"""
        if not nodes or not embeddings:
            return

        if len(nodes) != len(embeddings):
            raise ValueError(f"节点数量 ({len(nodes)}) 与向量数量 ({len(embeddings)}) 不匹配")

        if self._get_chroma_collection is None:
            raise NotImplementedError("SQLite embedding 保存需要配置 ChromaDB")

        collection = self._get_chroma_collection()

        # 准备数据
        ids = [str(node.id) for node in nodes]
        documents = [f"{node.title or ''}\n{node.content or ''}".strip() for node in nodes]
        metadatas = [
            {
                "document_id": str(node.document_id),
                "title": node.title or "",
                "node_type": node.node_type,
                "depth": node.depth,
                "model_name": model_name,
            }
            for node in nodes
        ]

        # ChromaDB upsert
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    async def delete_embeddings_by_document(self, document_id: "UUID") -> None:
        """删除文档的所有 embedding"""
        if self._get_chroma_collection is None:
            raise NotImplementedError("SQLite embedding 删除需要配置 ChromaDB")

        collection = self._get_chroma_collection()

        # ChromaDB 按 metadata 条件删除
        collection.delete(
            where={"document_id": str(document_id)}
        )

    def _build_fts_query(self, query: str) -> str:
        """构建 FTS5 查询字符串

        将用户输入转换为 FTS5 查询格式。
        使用 OR 逻辑连接词语，提高召回率。
        """
        # 移除特殊字符，保留字母、数字、中文
        cleaned = re.sub(r"[^\w\u4e00-\u9fff\s]", " ", query)

        # 分词
        words = cleaned.split()
        words = [w.strip() for w in words if w.strip()]

        if not words:
            return ""

        # 使用 OR 逻辑连接，提高召回率
        # 对于问答场景，用户输入 "dbdiag 是什么" 应该能找到 dbdiag 相关内容
        return " OR ".join(f'"{w}"' for w in words)


class SQLiteRepositoryFactory(RepositoryFactory):
    """SQLite 仓库工厂"""

    def __init__(self, config: SQLiteConfig):
        self._config = config
        self._db_path = config.get_db_path()
        self._chroma_path = config.get_chroma_path()

        # 初始化 ChromaDB
        self._chroma_client = None
        self._chroma_collection = None

        self._document_repo = SQLiteDocumentRepository(self._db_path)
        self._node_repo = SQLiteNodeRepository(self._db_path)
        self._search_repo = SQLiteSearchRepository(self._db_path, self._get_chroma_collection)

    def _get_chroma_collection(self):
        """懒加载 ChromaDB collection"""
        if self._chroma_collection is None:
            try:
                import chromadb
                self._chroma_path.parent.mkdir(parents=True, exist_ok=True)
                self._chroma_client = chromadb.PersistentClient(path=str(self._chroma_path))
                self._chroma_collection = self._chroma_client.get_or_create_collection(
                    name="nodes",
                    metadata={"hnsw:space": "cosine"}
                )
            except ImportError:
                raise ImportError("ChromaDB 未安装。请运行: pip install chromadb")
        return self._chroma_collection

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
