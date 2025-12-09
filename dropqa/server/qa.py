"""QA 问答服务"""

import logging
from dataclasses import dataclass
from typing import Optional

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
class QAResponse:
    """问答响应"""
    answer: str
    sources: list[SourceReference]


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
    """

    def __init__(
        self,
        search_service: SearchService,
        llm_service: LLMService,
        top_k: int = 5,
        max_context_length: int = 3000,
    ):
        """初始化问答服务

        Args:
            search_service: 搜索服务
            llm_service: LLM 服务
            top_k: 检索文档数量
            max_context_length: 单个上下文最大长度
        """
        self.search_service = search_service
        self.llm_service = llm_service
        self.top_k = top_k
        self.max_context_length = max_context_length

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

        # 1. 搜索相关文档
        search_results = await self.search_service.fulltext_search(
            question,
            top_k=self.top_k,
        )

        if not search_results:
            # 无搜索结果
            answer = await self.llm_service.chat([
                {"role": "user", "content": NO_CONTEXT_PROMPT.format(question=question)}
            ])
            return QAResponse(answer=answer, sources=[])

        # 2. 获取节点上下文
        contexts: list[NodeContext] = []
        for result in search_results:
            context = await self.search_service.get_node_context(result.node_id)
            if context:
                contexts.append(context)

        if not contexts:
            answer = await self.llm_service.chat([
                {"role": "user", "content": NO_CONTEXT_PROMPT.format(question=question)}
            ])
            return QAResponse(answer=answer, sources=[])

        # 3. 构建 Prompt
        prompt = self._build_context_prompt(contexts, question)

        # 4. 调用 LLM
        answer = await self.llm_service.chat([
            {"role": "user", "content": prompt}
        ])

        # 5. 构建来源引用
        sources = self._build_sources(contexts)

        return QAResponse(answer=answer, sources=sources)

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
