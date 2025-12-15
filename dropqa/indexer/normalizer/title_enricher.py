"""无意义标题增强器

检测并增强无意义的标题（如纯编号标题），使用 LLM 生成有意义的描述性标题。
"""

import re
from typing import TYPE_CHECKING, Optional, Callable, Awaitable

from dropqa.common.config import TitleEnrichConfig
from dropqa.indexer.normalizer.base import NodeNormalizer

if TYPE_CHECKING:
    from dropqa.indexer.parser import ParsedNode


# 异步 LLM 函数类型
LLMFunc = Callable[[str], Awaitable[str]]


class TitleEnricher(NodeNormalizer):
    """无意义标题增强器

    检测无意义的标题（如 "一、"、"1."、"第一章" 等），
    使用 LLM 根据内容生成有意义的描述性标题。

    支持两种模式：
    1. preserve_original=True：保留原编号，追加描述（如 "一、" -> "一、项目概述"）
    2. preserve_original=False：完全替换为新标题
    """

    def __init__(
        self,
        config: TitleEnrichConfig,
        llm_func: Optional[LLMFunc] = None,
    ):
        """初始化标题增强器

        Args:
            config: 增强配置
            llm_func: 异步 LLM 调用函数，签名为 async (prompt: str) -> str
                     如果为 None，则不进行增强
        """
        self._config = config
        self._llm_func = llm_func
        # 编译正则表达式
        self._patterns = [re.compile(p) for p in config.patterns]

    async def normalize(self, node: "ParsedNode") -> "ParsedNode":
        """规范化节点树，增强无意义标题

        Args:
            node: 待规范化的节点

        Returns:
            规范化后的节点
        """
        if not self._config.enabled:
            return node

        if self._llm_func is None:
            # 没有 LLM 函数，无法增强
            return node

        await self._normalize_recursive(node)
        return node

    async def _normalize_recursive(self, node: "ParsedNode") -> None:
        """递归处理节点"""
        # 如果是 section 且标题无意义，尝试增强
        if node.node_type == "section" and self._is_meaningless_title(node.title):
            new_title = await self._enrich_title(node)
            if new_title:
                node.title = new_title

        # 递归处理子节点
        for child in node.children:
            await self._normalize_recursive(child)

    def _is_meaningless_title(self, title: Optional[str]) -> bool:
        """判断标题是否无意义

        Args:
            title: 标题文本

        Returns:
            True 表示标题无意义，需要增强
        """
        if not title:
            return False

        title = title.strip()
        if not title:
            return False

        # 检查是否匹配任一无意义模式
        for pattern in self._patterns:
            if pattern.match(title):
                return True

        return False

    async def _enrich_title(self, node: "ParsedNode") -> Optional[str]:
        """增强标题

        Args:
            node: section 节点

        Returns:
            增强后的标题，失败返回 None
        """
        # 收集内容预览
        content_preview = self._get_content_preview(node, max_length=500)
        if not content_preview:
            return None

        try:
            prompt = f"""以下是一个文档章节的内容预览。该章节的标题是 "{node.title}"，但这个标题缺乏描述性。
请根据内容生成一个简短的描述性标题（5-15个字）。
只输出标题文字，不要输出其他内容。

内容预览：
{content_preview}

描述性标题："""

            generated_title = await self._llm_func(prompt)
            if not generated_title:
                return None

            # 清理生成的标题
            generated_title = generated_title.strip().strip('"').strip("'").strip("《》")

            # 验证标题长度
            if not generated_title or len(generated_title) > 30:
                return None

            # 根据配置决定是否保留原编号
            if self._config.preserve_original and node.title:
                # 提取原始编号部分
                original_prefix = self._extract_prefix(node.title)
                if original_prefix:
                    return f"{original_prefix}{generated_title}"

            return generated_title

        except Exception:
            return None

    def _get_content_preview(self, node: "ParsedNode", max_length: int = 500) -> str:
        """获取节点内容预览

        Args:
            node: 节点
            max_length: 最大长度

        Returns:
            内容预览文本
        """
        content = self._collect_all_content(node)
        if len(content) > max_length:
            content = content[:max_length] + "..."
        return content

    def _extract_prefix(self, title: str) -> Optional[str]:
        """从标题中提取编号前缀

        例如：
        - "一、" -> "一、"
        - "1." -> "1. "
        - "第一章" -> "第一章 "

        Args:
            title: 原始标题

        Returns:
            编号前缀（包含空格或标点），如果没有则返回 None
        """
        title = title.strip()

        # 中文数字编号：一、二、三、等
        match = re.match(r"^([一二三四五六七八九十]+[、.]?)\s*", title)
        if match:
            prefix = match.group(1)
            if not prefix.endswith(("、", ".")):
                prefix += " "
            return prefix

        # 阿拉伯数字编号：1. 2. 等
        match = re.match(r"^(\d+[、.])\s*", title)
        if match:
            prefix = match.group(1)
            if not prefix.endswith(" "):
                prefix += " "
            return prefix

        # 章节编号：第一章、第1节 等
        match = re.match(r"^(第[一二三四五六七八九十\d]+[章节部分])\s*", title)
        if match:
            return match.group(1) + " "

        # 字母编号：A. B. 等
        match = re.match(r"^([A-Z][、.])\s*", title)
        if match:
            prefix = match.group(1)
            if not prefix.endswith(" "):
                prefix += " "
            return prefix

        return None
