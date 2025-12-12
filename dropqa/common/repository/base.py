"""Repository 抽象接口

定义数据访问层的统一接口，支持多种后端实现。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional
from uuid import UUID


# 搜索策略类型
SearchStrategy = Literal["auto", "fulltext", "keyword", "regex"]


@dataclass
class DocumentData:
    """文档数据"""
    id: UUID
    filename: str
    file_type: str
    file_hash: str
    file_size: int
    storage_path: str
    current_version: int = 1


@dataclass
class NodeData:
    """节点数据"""
    id: UUID
    document_id: UUID
    parent_id: Optional[UUID]
    node_type: str
    depth: int
    title: Optional[str]
    content: Optional[str]
    summary: Optional[str] = None
    position: int = 0
    version: int = 1


@dataclass
class SearchResult:
    """搜索结果"""
    node_id: UUID
    document_id: UUID
    title: Optional[str]
    content: Optional[str]
    rank: float


@dataclass
class NodeWithAncestors:
    """节点及其祖先信息"""
    node_id: UUID
    content: Optional[str]
    ancestors: list["AncestorInfo"]
    document_name: str


@dataclass
class AncestorInfo:
    """祖先节点信息"""
    title: str
    summary: Optional[str]
    depth: int


class DocumentRepository(ABC):
    """文档仓库接口"""

    @abstractmethod
    async def save(self, doc: DocumentData) -> None:
        """保存文档

        Args:
            doc: 文档数据
        """
        pass

    @abstractmethod
    async def get_by_id(self, doc_id: UUID) -> Optional[DocumentData]:
        """根据 ID 获取文档

        Args:
            doc_id: 文档 ID

        Returns:
            文档数据，不存在返回 None
        """
        pass

    @abstractmethod
    async def get_by_path(self, storage_path: str) -> Optional[DocumentData]:
        """根据存储路径获取文档

        Args:
            storage_path: 存储路径

        Returns:
            文档数据，不存在返回 None
        """
        pass

    @abstractmethod
    async def get_by_hash(self, file_hash: str) -> Optional[DocumentData]:
        """根据文件哈希获取文档

        Args:
            file_hash: 文件哈希

        Returns:
            文档数据，不存在返回 None
        """
        pass

    @abstractmethod
    async def update(self, doc: DocumentData) -> None:
        """更新文档

        Args:
            doc: 文档数据
        """
        pass

    @abstractmethod
    async def delete(self, doc_id: UUID) -> bool:
        """删除文档

        Args:
            doc_id: 文档 ID

        Returns:
            是否成功删除
        """
        pass

    @abstractmethod
    async def delete_by_path(self, storage_path: str) -> bool:
        """根据存储路径删除文档

        Args:
            storage_path: 存储路径

        Returns:
            是否成功删除
        """
        pass


class NodeRepository(ABC):
    """节点仓库接口"""

    @abstractmethod
    async def save_batch(self, nodes: list[NodeData]) -> None:
        """批量保存节点

        Args:
            nodes: 节点数据列表
        """
        pass

    @abstractmethod
    async def delete_by_document(self, document_id: UUID) -> None:
        """删除文档的所有节点

        Args:
            document_id: 文档 ID
        """
        pass

    @abstractmethod
    async def get_with_ancestors(self, node_id: UUID) -> Optional[NodeWithAncestors]:
        """获取节点及其所有祖先

        Args:
            node_id: 节点 ID

        Returns:
            节点及祖先信息，不存在返回 None
        """
        pass


class SearchRepository(ABC):
    """搜索仓库接口"""

    @abstractmethod
    async def search(
        self,
        query: str,
        strategy: SearchStrategy = "auto",
        top_k: int = 10,
    ) -> list[SearchResult]:
        """统一搜索入口

        Args:
            query: 搜索查询
            strategy: 搜索策略
                - "auto": 自动选择（默认等同于 fulltext）
                - "fulltext": 全文搜索
                - "keyword": 精确关键词匹配（不区分大小写）
                - "regex": 正则表达式匹配（仅 PostgreSQL 支持）
            top_k: 返回结果数量

        Returns:
            搜索结果列表（关键词匹配优先于全文搜索）
        """
        pass

    @abstractmethod
    async def fulltext_search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """全文搜索

        Args:
            query: 搜索查询
            top_k: 返回结果数量

        Returns:
            搜索结果列表
        """
        pass

    @abstractmethod
    async def keyword_search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """关键词搜索（精确匹配，不区分大小写）

        Args:
            query: 搜索关键词
            top_k: 返回结果数量

        Returns:
            搜索结果列表
        """
        pass

    @abstractmethod
    async def vector_search(
        self,
        embedding: list[float],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """向量搜索

        Args:
            embedding: 查询向量
            top_k: 返回结果数量

        Returns:
            搜索结果列表
        """
        pass

    @abstractmethod
    async def hybrid_search(
        self,
        query: str,
        embedding: list[float],
        top_k: int = 10,
        fulltext_weight: float = 0.5,
    ) -> list[SearchResult]:
        """混合搜索（全文 + 向量）

        Args:
            query: 文本查询
            embedding: 查询向量
            top_k: 返回结果数量
            fulltext_weight: 全文搜索权重

        Returns:
            搜索结果列表
        """
        pass


class RepositoryFactory(ABC):
    """仓库工厂接口"""

    @abstractmethod
    def get_document_repository(self) -> DocumentRepository:
        """获取文档仓库"""
        pass

    @abstractmethod
    def get_node_repository(self) -> NodeRepository:
        """获取节点仓库"""
        pass

    @abstractmethod
    def get_search_repository(self) -> SearchRepository:
        """获取搜索仓库"""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """初始化存储（建表、索引等）"""
        pass

    @abstractmethod
    async def close(self) -> None:
        """关闭连接"""
        pass
