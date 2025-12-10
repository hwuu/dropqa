"""Repository 模块

提供数据访问层抽象，支持多种后端实现。
"""

from dropqa.common.repository.base import (
    DocumentRepository,
    NodeRepository,
    SearchRepository,
    RepositoryFactory,
    DocumentData,
    NodeData,
    SearchResult,
    NodeWithAncestors,
    AncestorInfo,
)
from dropqa.common.repository.postgres import PostgresRepositoryFactory
from dropqa.common.repository.sqlite import SQLiteRepositoryFactory

__all__ = [
    # 抽象接口
    "DocumentRepository",
    "NodeRepository",
    "SearchRepository",
    "RepositoryFactory",
    # 数据类
    "DocumentData",
    "NodeData",
    "SearchResult",
    "NodeWithAncestors",
    "AncestorInfo",
    # 后端实现
    "PostgresRepositoryFactory",
    "SQLiteRepositoryFactory",
]
