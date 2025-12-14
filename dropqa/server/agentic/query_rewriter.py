"""查询改写模块"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from dropqa.server.llm import LLMService
from dropqa.server.agentic.prompts import QUERY_REWRITE_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class RewrittenQuery:
    """改写后的查询结果"""
    original: str                    # 原始问题
    keywords: list[str]              # 提取的关键词
    fulltext_query: str              # 全文搜索查询
    semantic_query: str              # 语义搜索查询


class QueryRewriter:
    """查询改写器

    使用 LLM 将用户的自然语言问题转换为更适合搜索的形式。
    """

    def __init__(self, llm_service: LLMService):
        """初始化查询改写器

        Args:
            llm_service: LLM 服务实例
        """
        self._llm = llm_service

    async def rewrite(self, question: str) -> RewrittenQuery:
        """改写用户问题

        Args:
            question: 用户的原始问题

        Returns:
            RewrittenQuery 对象
        """
        question = question.strip()
        if not question:
            return RewrittenQuery(
                original=question,
                keywords=[],
                fulltext_query=question,
                semantic_query=question,
            )

        try:
            prompt = QUERY_REWRITE_PROMPT.format(question=question)
            messages = [{"role": "user", "content": prompt}]
            response = await self._llm.chat(messages)

            # 解析 JSON 响应
            result = self._parse_response(response)

            if result:
                return RewrittenQuery(
                    original=question,
                    keywords=result.get("keywords", []),
                    fulltext_query=result.get("fulltext_query", question),
                    semantic_query=result.get("semantic_query", question),
                )

        except Exception as e:
            logger.warning(f"[QueryRewriter] 改写失败: {e}，使用原始查询")

        # 回退：返回原始问题
        return RewrittenQuery(
            original=question,
            keywords=self._extract_keywords_simple(question),
            fulltext_query=question,
            semantic_query=question,
        )

    def _parse_response(self, response: str) -> Optional[dict]:
        """解析 LLM 响应

        Args:
            response: LLM 返回的文本

        Returns:
            解析后的字典，解析失败返回 None
        """
        if not response:
            return None

        # 尝试直接解析 JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # 尝试从 markdown 代码块中提取 JSON
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试提取 { } 之间的内容
        brace_match = re.search(r'\{[\s\S]*\}', response)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning(f"[QueryRewriter] 无法解析响应: {response[:200]}")
        return None

    def _extract_keywords_simple(self, question: str) -> list[str]:
        """简单的关键词提取（回退方案）

        Args:
            question: 用户问题

        Returns:
            关键词列表
        """
        # 移除常见的问句词汇
        stop_words = {
            "什么", "怎么", "如何", "为什么", "哪里", "哪个", "哪些",
            "是什么", "是否", "有没有", "能不能", "可以", "请问",
            "的", "了", "吗", "呢", "啊", "吧", "呀",
            "the", "what", "how", "why", "where", "which", "is", "are",
            "can", "could", "would", "should", "do", "does", "did",
        }

        # 简单分词（按空格和标点）
        words = re.split(r'[\s,，。？！、：；""''（）\(\)\[\]]+', question)

        # 过滤
        keywords = [
            w.strip() for w in words
            if w.strip() and len(w.strip()) > 1 and w.lower() not in stop_words
        ]

        return keywords[:5]  # 最多返回 5 个关键词
