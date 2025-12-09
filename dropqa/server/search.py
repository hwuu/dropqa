"""全文搜索服务"""

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy import text

from dropqa.common.db import Database


@dataclass
class SearchResult:
    """搜索结果"""
    node_id: uuid.UUID
    document_id: uuid.UUID
    title: Optional[str]
    content: Optional[str]
    rank: float


@dataclass
class BreadcrumbItem:
    """面包屑项"""
    title: str
    summary: Optional[str]
    depth: int


@dataclass
class NodeContext:
    """节点上下文（包含面包屑）"""
    node_id: uuid.UUID
    content: Optional[str]
    breadcrumb: list[BreadcrumbItem]
    document_name: str

    def get_path_string(self) -> str:
        """获取路径字符串

        Returns:
            格式如 "第1章 > 1.1 背景"
        """
        # 跳过 depth=0 的文档节点
        titles = [item.title for item in self.breadcrumb if item.depth > 0 and item.title]
        return " > ".join(titles)


class SearchService:
    """全文搜索服务

    基于 PostgreSQL tsvector 实现全文搜索。
    """

    def __init__(self, db: Database):
        """初始化搜索服务

        Args:
            db: 数据库实例
        """
        self.db = db

    async def fulltext_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """全文搜索

        Args:
            query: 搜索查询
            top_k: 返回结果数量

        Returns:
            搜索结果列表，按相关度降序排列
        """
        query = query.strip()
        if not query:
            return []

        tsquery = self._build_tsquery(query)

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

        async with self.db.session() as session:
            result = await session.execute(
                sql,
                {"tsquery": tsquery, "top_k": top_k},
            )
            rows = result.fetchall()

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

    def _build_tsquery(self, query: str) -> str:
        """构建 tsquery 查询字符串

        将用户输入转换为 PostgreSQL tsquery 格式。
        多个词之间用 & (AND) 连接。

        Args:
            query: 用户输入的查询

        Returns:
            tsquery 格式的字符串
        """
        # 移除特殊字符，保留字母、数字、中文
        cleaned = re.sub(r"[^\w\u4e00-\u9fff\s]", " ", query)

        # 分词（按空格分割）
        words = cleaned.split()

        # 过滤空词
        words = [w.strip() for w in words if w.strip()]

        if not words:
            return ""

        # 用 & 连接
        return " & ".join(words)

    async def get_node_context(self, node_id: uuid.UUID) -> Optional[NodeContext]:
        """获取节点上下文（包含面包屑）

        Args:
            node_id: 节点 ID

        Returns:
            NodeContext 对象，如果节点不存在返回 None
        """
        # 使用递归 CTE 获取节点及其所有祖先
        sql = text("""
            WITH RECURSIVE ancestors AS (
                -- 基础节点
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

                -- 递归获取祖先
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

        async with self.db.session() as session:
            result = await session.execute(sql, {"node_id": str(node_id)})
            rows = result.fetchall()

        if not rows:
            return None

        # 第一行（depth 最小）到最后一行是从根到目标节点的路径
        # 最后一行（level=0）是目标节点本身
        target_node = None
        breadcrumb = []
        document_name = ""

        for row in rows:
            if row.level == 0:
                # 目标节点
                target_node = row
                document_name = row.document_name or "unknown"

            # 构建面包屑（所有节点都加入）
            breadcrumb.append(BreadcrumbItem(
                title=row.title or "",
                summary=row.summary,
                depth=row.depth,
            ))

        if target_node is None:
            return None

        return NodeContext(
            node_id=node_id,
            content=target_node.content,
            breadcrumb=breadcrumb,
            document_name=document_name,
        )
