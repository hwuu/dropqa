"""Reranker 结果重排序模块"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from dropqa.server.llm import LLMService
from dropqa.server.search import NodeContext
from dropqa.server.agentic.prompts import RERANK_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class RankedResult:
    """重排序后的结果"""
    context: NodeContext
    score: float
    reason: Optional[str] = None


class Reranker:
    """结果重排序器

    使用 LLM 评估检索结果与问题的相关性，重新排序。
    """

    def __init__(self, llm_service: LLMService):
        """初始化重排序器

        Args:
            llm_service: LLM 服务实例
        """
        self._llm = llm_service

    async def rerank(
        self,
        question: str,
        contexts: list[NodeContext],
        top_k: int = 5,
    ) -> list[RankedResult]:
        """重排序检索结果

        Args:
            question: 用户问题
            contexts: 待重排序的上下文列表
            top_k: 返回的结果数量

        Returns:
            重排序后的结果列表
        """
        if not contexts:
            return []

        if len(contexts) <= 1:
            # 只有一个结果，直接返回
            return [RankedResult(context=contexts[0], score=10.0)]

        try:
            # 构建候选列表文本
            candidates_text = self._build_candidates_text(contexts)

            # 调用 LLM 评分
            prompt = RERANK_PROMPT.format(
                question=question,
                candidates=candidates_text,
            )
            messages = [{"role": "user", "content": prompt}]
            response = await self._llm.chat(messages)

            # 解析响应
            scores = self._parse_response(response, contexts)

            if scores:
                # 按分数排序
                ranked = sorted(scores, key=lambda x: x.score, reverse=True)
                return ranked[:top_k]

        except Exception as e:
            logger.warning(f"[Reranker] 重排序失败: {e}，返回原始顺序")

        # 回退：返回原始顺序
        return [
            RankedResult(context=ctx, score=10.0 - i * 0.1)
            for i, ctx in enumerate(contexts[:top_k])
        ]

    def _build_candidates_text(self, contexts: list[NodeContext]) -> str:
        """构建候选列表文本

        Args:
            contexts: 上下文列表

        Returns:
            格式化的候选文本
        """
        lines = []
        for i, ctx in enumerate(contexts):
            path = ctx.get_path_string()
            # 获取内容预览：优先使用 content，若为空则使用路径中的标题
            content = ctx.content or ""
            if content:
                content_preview = content[:500]
            else:
                # content 为空时，使用路径（包含标题层级）作为内容提示
                content_preview = f"[标题节点] {path}"

            lines.append(f"[{i}] 文档: {ctx.document_name}")
            lines.append(f"    路径: {path}")
            lines.append(f"    内容: {content_preview}")
            lines.append("")
        return "\n".join(lines)

    def _parse_response(
        self,
        response: str,
        contexts: list[NodeContext],
    ) -> Optional[list[RankedResult]]:
        """解析 LLM 响应

        Args:
            response: LLM 返回的文本
            contexts: 原始上下文列表

        Returns:
            排序后的结果列表，解析失败返回 None
        """
        if not response:
            return None

        # 尝试解析 JSON
        data = self._extract_json(response)
        if not data or "scores" not in data:
            return None

        # 构建结果
        results = []
        scores_data = data["scores"]

        for item in scores_data:
            try:
                # ID 可能是数字索引或字符串
                idx = item.get("id")
                if isinstance(idx, str):
                    idx = int(idx)
                if idx is None or idx >= len(contexts):
                    continue

                score = float(item.get("score", 0))
                reason = item.get("reason")

                results.append(RankedResult(
                    context=contexts[idx],
                    score=score,
                    reason=reason,
                ))
            except (ValueError, TypeError, IndexError):
                continue

        return results if results else None

    def _extract_json(self, response: str) -> Optional[dict]:
        """从响应中提取 JSON

        Args:
            response: LLM 响应文本

        Returns:
            解析后的字典，失败返回 None
        """
        # 尝试直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # 尝试从 markdown 代码块提取
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

        return None
