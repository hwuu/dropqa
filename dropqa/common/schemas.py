"""Pydantic 模型（API 请求/响应）"""

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ============ Document 相关 ============

class DocumentBase(BaseModel):
    """文档基础模型"""
    filename: str
    file_type: str


class DocumentCreate(DocumentBase):
    """创建文档"""
    file_hash: str
    file_size: int
    storage_path: str


class DocumentResponse(DocumentBase):
    """文档响应"""
    id: uuid.UUID
    file_hash: str
    file_size: int
    current_version: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ============ Node 相关 ============

class NodeBase(BaseModel):
    """节点基础模型"""
    node_type: str
    depth: int
    title: Optional[str] = None
    content: Optional[str] = None
    position: int = 0


class NodeCreate(NodeBase):
    """创建节点"""
    document_id: uuid.UUID
    parent_id: Optional[uuid.UUID] = None
    version: int = 1


class BreadcrumbItem(BaseModel):
    """面包屑项"""
    title: str
    summary: Optional[str] = None
    depth: int


class NodeResponse(NodeBase):
    """节点响应"""
    id: uuid.UUID
    document_id: uuid.UUID
    parent_id: Optional[uuid.UUID] = None
    version: int
    summary: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class NodeWithContext(NodeResponse):
    """带上下文的节点"""
    breadcrumb: list[BreadcrumbItem] = Field(default_factory=list)
    document_name: str = ""


# ============ 搜索相关 ============

class SearchRequest(BaseModel):
    """搜索请求"""
    query: str
    top_k: int = 10


class SearchResult(BaseModel):
    """搜索结果"""
    node: NodeWithContext
    score: float


class SearchResponse(BaseModel):
    """搜索响应"""
    results: list[SearchResult]
    total: int


# ============ 问答相关 ============

class QARequest(BaseModel):
    """问答请求"""
    question: str
    top_k: int = 5
    show_sources: bool = True


class SourceInfo(BaseModel):
    """来源信息"""
    document_id: uuid.UUID
    document_name: str
    section_path: str
    snippet: str
    relevance: float


class QAResponse(BaseModel):
    """问答响应"""
    answer: str
    sources: list[SourceInfo] = Field(default_factory=list)
    mode: str = "direct"  # direct 或 agent
