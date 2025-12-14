"""Agentic RAG Pipeline 模块"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from dropqa.server.agentic.config import AgenticConfig
from dropqa.server.agentic.query_rewriter import QueryRewriter, RewrittenQuery
from dropqa.server.agentic.reranker import Reranker, RankedResult
from dropqa.server.llm import LLMService
from dropqa.server.search import NodeContext, SearchService

if TYPE_CHECKING:
    from dropqa.common.embedding import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class AgenticRAGResult:
    """Agentic RAG 结果"""
    question: str
    rewritten_query: Optional[RewrittenQuery]
    contexts: list[NodeContext]
    ranked_results: list[RankedResult]


class AgenticRAGPipeline:
    """Agentic RAG 流程编排

    整合查询改写、混合搜索、多轮检索、结果重排序。
    """

    def __init__(
        self,
        config: AgenticConfig,
        search_service: SearchService,
        llm_service: LLMService,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        """初始化 Pipeline

        Args:
            config: Agentic 配置
            search_service: 搜索服务
            llm_service: LLM 服务
            embedding_service: Embedding 服务（可选，用于混合搜索）
        """
        self._config = config
        self._search_service = search_service
        self._llm_service = llm_service
        self._embedding_service = embedding_service

        # 初始化组件
        self._query_rewriter = QueryRewriter(llm_service) if config.query_rewrite.enabled else None
        self._reranker = Reranker(llm_service) if config.rerank.enabled else None

    async def run(self, question: str) -> AgenticRAGResult:
        """执行 Agentic RAG 流程

        Args:
            question: 用户问题

        Returns:
            AgenticRAGResult 结果
        """
        logger.info(f"[AgenticRAG] 开始处理问题: {question[:50]}...")

        # 1. 查询改写
        rewritten_query = None
        search_query = question
        embedding_query = question

        if self._query_rewriter and self._config.query_rewrite.enabled:
            logger.debug("[AgenticRAG] 执行查询改写")
            rewritten_query = await self._query_rewriter.rewrite(question)
            search_query = rewritten_query.fulltext_query
            embedding_query = rewritten_query.semantic_query
            logger.debug(f"[AgenticRAG] 改写结果: fulltext='{search_query}', semantic='{embedding_query}'")

        # 2. 执行搜索
        contexts = await self._execute_search(search_query, embedding_query)
        logger.debug(f"[AgenticRAG] 搜索返回 {len(contexts)} 条结果")

        # 3. 多轮检索（如果启用且结果不足）
        if self._config.multi_round.enabled and len(contexts) < self._config.top_k // 2:
            logger.debug("[AgenticRAG] 结果不足，执行补充检索")
            additional_contexts = await self._supplementary_search(question, contexts)
            contexts = self._merge_contexts(contexts, additional_contexts)
            logger.debug(f"[AgenticRAG] 补充后共 {len(contexts)} 条结果")

        # 4. 结果重排序
        ranked_results = []
        if self._reranker and self._config.rerank.enabled and contexts:
            logger.debug("[AgenticRAG] 执行结果重排序")
            ranked_results = await self._reranker.rerank(
                question,
                contexts,
                top_k=self._config.rerank.top_k,
            )
            logger.debug(f"[AgenticRAG] 重排序后返回 {len(ranked_results)} 条结果")
        else:
            # 不重排序，直接转换
            ranked_results = [
                RankedResult(context=ctx, score=10.0 - i * 0.1)
                for i, ctx in enumerate(contexts[:self._config.top_k])
            ]

        return AgenticRAGResult(
            question=question,
            rewritten_query=rewritten_query,
            contexts=contexts,
            ranked_results=ranked_results,
        )

    async def _execute_search(
        self,
        fulltext_query: str,
        embedding_query: str,
    ) -> list[NodeContext]:
        """执行搜索

        Args:
            fulltext_query: 全文搜索查询
            embedding_query: 语义搜索查询

        Returns:
            搜索结果上下文列表
        """
        top_k = self._config.top_k
        hybrid_config = self._config.hybrid_search

        # 判断是否使用混合搜索
        if hybrid_config.enabled and self._embedding_service:
            # 生成 embedding
            embedding = await self._embedding_service.embed(embedding_query)

            if embedding:
                # 混合搜索
                results = await self._search_service.hybrid_search(
                    fulltext_query,
                    embedding,
                    top_k=top_k,
                    fulltext_weight=hybrid_config.fulltext_weight,
                )
            else:
                # Embedding 生成失败，回退到全文搜索
                results = await self._search_service.fulltext_search(fulltext_query, top_k)
        else:
            # 纯全文搜索
            results = await self._search_service.fulltext_search(fulltext_query, top_k)

        # 获取完整上下文
        contexts = []
        for result in results:
            ctx = await self._search_service.get_node_context(result.node_id)
            if ctx:
                contexts.append(ctx)

        return contexts

    async def _supplementary_search(
        self,
        question: str,
        existing_contexts: list[NodeContext],
    ) -> list[NodeContext]:
        """补充检索

        Args:
            question: 原始问题
            existing_contexts: 已有的上下文

        Returns:
            补充的上下文列表
        """
        # 简化实现：使用关键词搜索补充
        # 完整实现应该使用 LLM 生成补充查询

        # 提取现有内容的关键词，尝试不同角度的搜索
        supplementary_query = question + " 相关"  # 简单追加词汇

        results = await self._search_service.fulltext_search(
            supplementary_query,
            top_k=self._config.top_k // 2,
        )

        # 获取上下文
        contexts = []
        existing_ids = {ctx.node_id for ctx in existing_contexts}

        for result in results:
            if result.node_id not in existing_ids:
                ctx = await self._search_service.get_node_context(result.node_id)
                if ctx:
                    contexts.append(ctx)

        return contexts

    def _merge_contexts(
        self,
        primary: list[NodeContext],
        secondary: list[NodeContext],
    ) -> list[NodeContext]:
        """合并上下文列表，去重

        Args:
            primary: 主要结果
            secondary: 补充结果

        Returns:
            合并后的列表
        """
        seen_ids = {ctx.node_id for ctx in primary}
        merged = list(primary)

        for ctx in secondary:
            if ctx.node_id not in seen_ids:
                merged.append(ctx)
                seen_ids.add(ctx.node_id)

        return merged[:self._config.top_k]
