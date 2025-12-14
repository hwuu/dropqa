"""Agentic RAG 配置模型"""

from pydantic import BaseModel, Field


class QueryRewriteConfig(BaseModel):
    """查询改写配置"""
    enabled: bool = True


class HybridSearchConfig(BaseModel):
    """混合搜索配置

    Attributes:
        enabled: 是否启用混合搜索
        fulltext_weight: 全文搜索权重 (0.0-1.0)，向量搜索权重 = 1 - fulltext_weight
    """
    enabled: bool = True
    fulltext_weight: float = Field(default=0.5, ge=0.0, le=1.0)


class MultiRoundConfig(BaseModel):
    """多轮检索配置

    Attributes:
        enabled: 是否启用多轮检索
        max_rounds: 最大检索轮数
    """
    enabled: bool = True
    max_rounds: int = Field(default=2, ge=1, le=5)


class RerankConfig(BaseModel):
    """结果重排序配置

    Attributes:
        enabled: 是否启用重排序
        top_k: 重排序后返回的结果数量
    """
    enabled: bool = True
    top_k: int = Field(default=5, ge=1)


class AgenticConfig(BaseModel):
    """Agentic RAG 总配置

    Attributes:
        enabled: 总开关，关闭时走原有简单 RAG
        top_k: 检索返回的最大结果数
        query_rewrite: 查询改写配置
        hybrid_search: 混合搜索配置
        multi_round: 多轮检索配置
        rerank: 结果重排序配置
    """
    enabled: bool = False  # 默认关闭，保持向后兼容
    top_k: int = Field(default=10, ge=1)

    query_rewrite: QueryRewriteConfig = Field(default_factory=QueryRewriteConfig)
    hybrid_search: HybridSearchConfig = Field(default_factory=HybridSearchConfig)
    multi_round: MultiRoundConfig = Field(default_factory=MultiRoundConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
