"""QA 问答服务"""

import logging
from dataclasses import dataclass
from typing import Optional, AsyncGenerator

from dropqa.common.embedding import EmbeddingService
from dropqa.server.agentic.config import AgenticConfig
from dropqa.server.agentic.pipeline import AgenticRAGPipeline
from dropqa.server.llm import LLMService
from dropqa.server.search import SearchService, NodeContext

logger = logging.getLogger(__name__)


@dataclass
class SourceReference:
    """引用来源"""
    document_name: str
    path: str
    content_snippet: str


@dataclass
class ReasoningStep:
    """推理步骤"""
    step: str
    action: str
    result: str = ""


@dataclass
class QAResponse:
    """问答响应"""
    answer: str
    sources: list[SourceReference]
    mode: str = "simple"  # 'simple' or 'agentic'
    reasoning_trace: list[ReasoningStep] | None = None


@dataclass
class ProgressEvent:
    """进度事件"""
    event: str  # 'progress' or 'complete'
    message: str = ""
    data: Optional[QAResponse] = None


# RAG Prompt 模板
RAG_PROMPT_TEMPLATE = """基于以下文档内容回答用户的问题。

## 文档内容

{context}

## 用户问题

{question}

## 回答要求

1. 仅基于上述文档内容回答，不要编造信息
2. 如果文档中没有相关信息，请明确说明"文档中没有找到相关信息"
3. 回答要简洁明了
4. 如果引用了文档内容，请注明出处（文档名和章节）

请回答："""

NO_CONTEXT_PROMPT = """用户问题：{question}

抱歉，没有找到与您问题相关的文档内容。请尝试：
1. 使用不同的关键词提问
2. 确认相关文档已被索引"""


class QAService:
    """问答服务

    整合搜索和 LLM 实现 RAG 问答。
    支持简单 RAG 和 Agentic RAG 两种模式。
    """

    def __init__(
        self,
        search_service: SearchService,
        llm_service: LLMService,
        top_k: int = 5,
        max_context_length: int = 3000,
        agentic_config: Optional[AgenticConfig] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        """初始化问答服务

        Args:
            search_service: 搜索服务
            llm_service: LLM 服务
            top_k: 检索文档数量
            max_context_length: 单个上下文最大长度
            agentic_config: Agentic RAG 配置（可选）
            embedding_service: Embedding 服务（可选，用于混合搜索）
        """
        self.search_service = search_service
        self.llm_service = llm_service
        self.top_k = top_k
        self.max_context_length = max_context_length
        self.agentic_config = agentic_config
        self.embedding_service = embedding_service

        # 初始化 Agentic Pipeline（如果启用）
        self._agentic_pipeline: Optional[AgenticRAGPipeline] = None
        if agentic_config and agentic_config.enabled:
            self._agentic_pipeline = AgenticRAGPipeline(
                config=agentic_config,
                search_service=search_service,
                llm_service=llm_service,
                embedding_service=embedding_service,
            )
            logger.info("[QA] Agentic RAG 模式已启用")
        else:
            logger.info("[QA] 使用简单 RAG 模式")

    async def ask(self, question: str) -> QAResponse:
        """问答

        Args:
            question: 用户问题

        Returns:
            问答响应
        """
        question = question.strip()
        if not question:
            return QAResponse(
                answer="请输入您的问题。",
                sources=[],
            )

        # 根据配置选择 RAG 模式
        if self._agentic_pipeline:
            return await self._ask_agentic(question)
        else:
            return await self._ask_simple(question)

    async def ask_stream(self, question: str) -> AsyncGenerator[ProgressEvent, None]:
        """流式问答，实时返回进度

        Args:
            question: 用户问题

        Yields:
            进度事件
        """
        question = question.strip()
        if not question:
            yield ProgressEvent(
                event="complete",
                data=QAResponse(answer="请输入您的问题。", sources=[]),
            )
            return

        # 根据配置选择 RAG 模式
        if self._agentic_pipeline:
            async for event in self._ask_agentic_stream(question):
                yield event
        else:
            async for event in self._ask_simple_stream(question):
                yield event

    async def _ask_agentic_stream(self, question: str) -> AsyncGenerator[ProgressEvent, None]:
        """Agentic RAG 流式问答"""
        reasoning_trace = []
        result = None

        # 使用流式 pipeline，实时返回进度
        async for progress in self._agentic_pipeline.run_stream(question):
            if progress.stage == "query_rewrite":
                yield ProgressEvent(event="progress", message=progress.message)
            elif progress.stage == "search":
                yield ProgressEvent(event="progress", message=progress.message)
            elif progress.stage == "rerank":
                yield ProgressEvent(event="progress", message=progress.message)
            elif progress.stage == "complete":
                result = progress.result

        # 检查是否有结果
        if result is None:
            yield ProgressEvent(
                event="complete",
                data=QAResponse(answer="处理过程中出现错误", sources=[], mode="agentic"),
            )
            return

        # 构建推理过程
        if result.rewritten_query:
            rq = result.rewritten_query
            reasoning_trace.append(ReasoningStep(
                step="查询改写",
                action=f"原始问题: {question}",
                result=f"关键词: {', '.join(rq.keywords)} | 全文查询: {rq.fulltext_query}",
            ))

        if result.contexts:
            reasoning_trace.append(ReasoningStep(
                step="文档检索",
                action="混合搜索（全文 + 向量）",
                result=f"找到 {len(result.contexts)} 条相关内容",
            ))

        if result.ranked_results:
            reasoning_trace.append(ReasoningStep(
                step="结果重排序",
                action="基于相关性重新排序",
                result=f"保留前 {len(result.ranked_results)} 条最相关结果",
            ))

        # 检查是否有搜索结果
        if not result.ranked_results:
            answer = await self.llm_service.chat([
                {"role": "user", "content": NO_CONTEXT_PROMPT.format(question=question)}
            ])
            yield ProgressEvent(
                event="complete",
                data=QAResponse(
                    answer=answer,
                    sources=[],
                    mode="agentic",
                    reasoning_trace=reasoning_trace,
                ),
            )
            return

        # 生成回答
        yield ProgressEvent(event="progress", message="正在生成回答...")

        contexts = [r.context for r in result.ranked_results]
        prompt = self._build_context_prompt(contexts, question)
        answer = await self.llm_service.chat([
            {"role": "user", "content": prompt}
        ])

        reasoning_trace.append(ReasoningStep(
            step="答案生成",
            action="基于检索内容生成回答",
            result=f"生成 {len(answer)} 字符的回答",
        ))

        sources = self._build_sources(contexts)

        yield ProgressEvent(
            event="complete",
            data=QAResponse(
                answer=answer,
                sources=sources,
                mode="agentic",
                reasoning_trace=reasoning_trace,
            ),
        )

    async def _ask_simple_stream(self, question: str) -> AsyncGenerator[ProgressEvent, None]:
        """简单 RAG 流式问答"""
        # 1. 搜索
        yield ProgressEvent(event="progress", message="正在搜索文档...")

        search_results = await self.search_service.fulltext_search(
            question,
            top_k=self.top_k,
        )

        if not search_results:
            answer = await self.llm_service.chat([
                {"role": "user", "content": NO_CONTEXT_PROMPT.format(question=question)}
            ])
            yield ProgressEvent(
                event="complete",
                data=QAResponse(answer=answer, sources=[], mode="simple"),
            )
            return

        yield ProgressEvent(
            event="progress",
            message=f"找到 {len(search_results)} 条相关内容"
        )

        # 2. 获取上下文
        contexts: list[NodeContext] = []
        for result in search_results:
            context = await self.search_service.get_node_context(result.node_id)
            if context:
                contexts.append(context)

        if not contexts:
            answer = await self.llm_service.chat([
                {"role": "user", "content": NO_CONTEXT_PROMPT.format(question=question)}
            ])
            yield ProgressEvent(
                event="complete",
                data=QAResponse(answer=answer, sources=[], mode="simple"),
            )
            return

        # 3. 生成回答
        yield ProgressEvent(event="progress", message="正在生成回答...")

        prompt = self._build_context_prompt(contexts, question)
        answer = await self.llm_service.chat([
            {"role": "user", "content": prompt}
        ])

        sources = self._build_sources(contexts)

        yield ProgressEvent(
            event="complete",
            data=QAResponse(answer=answer, sources=sources, mode="simple"),
        )

    async def _ask_agentic(self, question: str) -> QAResponse:
        """Agentic RAG 问答

        使用查询改写、混合搜索、多轮检索、结果重排序。
        """
        logger.debug(f"[AgenticRAG] 开始处理问题: {question}")

        # 执行 Agentic Pipeline
        result = await self._agentic_pipeline.run(question)

        # 构建推理过程
        reasoning_trace = []

        # 添加查询改写步骤
        if result.rewritten_query:
            rq = result.rewritten_query
            reasoning_trace.append(ReasoningStep(
                step="查询改写",
                action=f"原始问题: {question}",
                result=f"关键词: {', '.join(rq.keywords)} | 全文查询: {rq.fulltext_query}",
            ))

        # 添加搜索步骤
        if result.contexts:
            reasoning_trace.append(ReasoningStep(
                step="文档检索",
                action="混合搜索（全文 + 向量）",
                result=f"找到 {len(result.contexts)} 条相关内容",
            ))

        # 添加重排序步骤
        if result.ranked_results:
            reasoning_trace.append(ReasoningStep(
                step="结果重排序",
                action="基于相关性重新排序",
                result=f"保留前 {len(result.ranked_results)} 条最相关结果",
            ))

        # 检查是否有结果
        if not result.ranked_results:
            logger.debug("[AgenticRAG] 无搜索结果，返回默认回答")
            answer = await self.llm_service.chat([
                {"role": "user", "content": NO_CONTEXT_PROMPT.format(question=question)}
            ])
            return QAResponse(
                answer=answer,
                sources=[],
                mode="agentic",
                reasoning_trace=reasoning_trace,
            )

        # 使用重排序后的上下文
        contexts = [r.context for r in result.ranked_results]
        logger.debug(f"[AgenticRAG] 使用 {len(contexts)} 条重排序后的上下文")

        # 打印重排序结果
        for i, ranked in enumerate(result.ranked_results, 1):
            logger.debug(
                f"[AgenticRAG] 结果 {i}: "
                f"score={ranked.score:.1f}, "
                f"doc={ranked.context.document_name}, "
                f"reason={ranked.reason or 'N/A'}"
            )

        # 构建 Prompt 并调用 LLM
        prompt = self._build_context_prompt(contexts, question)
        answer = await self.llm_service.chat([
            {"role": "user", "content": prompt}
        ])

        # 添加生成步骤
        reasoning_trace.append(ReasoningStep(
            step="答案生成",
            action="基于检索内容生成回答",
            result=f"生成 {len(answer)} 字符的回答",
        ))

        # 构建来源引用
        sources = self._build_sources(contexts)

        return QAResponse(
            answer=answer,
            sources=sources,
            mode="agentic",
            reasoning_trace=reasoning_trace,
        )

    async def _ask_simple(self, question: str) -> QAResponse:
        """简单 RAG 问答

        原有的简单 RAG 流程。
        """
        logger.debug(f"[RAG] 开始处理问题: {question}")

        # 1. 搜索相关文档
        logger.debug(f"[RAG] 步骤1: 全文搜索, top_k={self.top_k}")
        search_results = await self.search_service.fulltext_search(
            question,
            top_k=self.top_k,
        )
        logger.debug(f"[RAG] 搜索结果数量: {len(search_results)}")

        if not search_results:
            logger.debug("[RAG] 无搜索结果，返回默认回答")
            # 无搜索结果
            answer = await self.llm_service.chat([
                {"role": "user", "content": NO_CONTEXT_PROMPT.format(question=question)}
            ])
            return QAResponse(answer=answer, sources=[], mode="simple")

        # 打印搜索结果详情
        for i, result in enumerate(search_results, 1):
            logger.debug(
                f"[RAG] 搜索结果 {i}: "
                f"rank={result.rank:.4f}, "
                f"title={result.title}, "
                f"content_preview={result.content[:100] if result.content else 'None'}..."
            )

        # 2. 获取节点上下文
        logger.debug("[RAG] 步骤2: 获取节点上下文")
        contexts: list[NodeContext] = []
        for result in search_results:
            context = await self.search_service.get_node_context(result.node_id)
            if context:
                contexts.append(context)
                logger.debug(
                    f"[RAG] 上下文: doc={context.document_name}, "
                    f"path={context.get_path_string()}"
                )

        if not contexts:
            logger.debug("[RAG] 无有效上下文，返回默认回答")
            answer = await self.llm_service.chat([
                {"role": "user", "content": NO_CONTEXT_PROMPT.format(question=question)}
            ])
            return QAResponse(answer=answer, sources=[], mode="simple")

        # 3. 构建 Prompt
        logger.debug("[RAG] 步骤3: 构建 Prompt")
        prompt = self._build_context_prompt(contexts, question)
        logger.debug(f"[RAG] Prompt 长度: {len(prompt)} 字符")

        # 4. 调用 LLM
        logger.debug("[RAG] 步骤4: 调用 LLM")
        answer = await self.llm_service.chat([
            {"role": "user", "content": prompt}
        ])
        logger.debug(f"[RAG] LLM 回答长度: {len(answer)} 字符")

        # 5. 构建来源引用
        sources = self._build_sources(contexts)
        logger.debug(f"[RAG] 步骤5: 构建来源引用, 数量={len(sources)}")

        return QAResponse(answer=answer, sources=sources, mode="simple")

    def _build_context_prompt(
        self,
        contexts: list[NodeContext],
        question: str,
    ) -> str:
        """构建带上下文的 Prompt

        Args:
            contexts: 节点上下文列表
            question: 用户问题

        Returns:
            完整的 Prompt
        """
        context_parts = []

        for i, ctx in enumerate(contexts, 1):
            path = ctx.get_path_string()
            content = self._truncate_content(
                ctx.content or "",
                self.max_context_length,
            )

            context_parts.append(
                f"### 文档 {i}: {ctx.document_name}\n"
                f"**位置**: {path}\n"
                f"**内容**:\n{content}\n"
            )

        context_text = "\n".join(context_parts)

        return RAG_PROMPT_TEMPLATE.format(
            context=context_text,
            question=question,
        )

    def _build_sources(self, contexts: list[NodeContext]) -> list[SourceReference]:
        """构建来源引用列表

        Args:
            contexts: 节点上下文列表

        Returns:
            来源引用列表
        """
        sources = []

        for ctx in contexts:
            snippet = self._truncate_content(ctx.content or "", max_length=200)
            sources.append(SourceReference(
                document_name=ctx.document_name,
                path=ctx.get_path_string(),
                content_snippet=snippet,
            ))

        return sources

    def _truncate_content(self, content: str, max_length: int) -> str:
        """截断内容

        Args:
            content: 原始内容
            max_length: 最大长度

        Returns:
            截断后的内容
        """
        if len(content) <= max_length:
            return content

        return content[:max_length] + "..."
