"""全文搜索服务"""

import logging
import uuid
from dataclasses import dataclass
from typing import Optional

from dropqa.common.repository import SearchRepository, NodeRepository

logger = logging.getLogger(__name__)


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

    基于 Repository 抽象层实现搜索功能。
    """

    def __init__(
        self,
        search_repo: SearchRepository,
        node_repo: NodeRepository,
    ):
        """初始化搜索服务

        Args:
            search_repo: 搜索仓库
            node_repo: 节点仓库
        """
        self._search_repo = search_repo
        self._node_repo = node_repo

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
        logger.debug(f"[Search] 全文搜索: query='{query}', top_k={top_k}")
        results = await self._search_repo.fulltext_search(query, top_k)
        logger.debug(f"[Search] 返回 {len(results)} 条结果")

        return [
            SearchResult(
                node_id=r.node_id,
                document_id=r.document_id,
                title=r.title,
                content=r.content,
                rank=r.rank,
            )
            for r in results
        ]

    async def get_node_context(self, node_id: uuid.UUID) -> Optional[NodeContext]:
        """获取节点上下文（包含面包屑）

        Args:
            node_id: 节点 ID

        Returns:
            NodeContext 对象，如果节点不存在返回 None
        """
        result = await self._node_repo.get_with_ancestors(node_id)

        if result is None:
            return None

        breadcrumb = [
            BreadcrumbItem(
                title=a.title,
                summary=a.summary,
                depth=a.depth,
            )
            for a in result.ancestors
        ]

        return NodeContext(
            node_id=result.node_id,
            content=result.content,
            breadcrumb=breadcrumb,
            document_name=result.document_name,
        )
