"""长 Section 拆分器

将过长且没有子 section 的 section 自动拆分为多个子 section。
"""

import re
from typing import TYPE_CHECKING, Optional, Callable, Awaitable

from dropqa.common.config import SectionSplitConfig
from dropqa.indexer.normalizer.base import NodeNormalizer

if TYPE_CHECKING:
    from dropqa.indexer.parser import ParsedNode


# 异步函数类型
LLMFunc = Callable[[str], Awaitable[str]]
EmbeddingFunc = Callable[[list[str]], Awaitable[list[list[float]]]]


class SectionSplitter(NodeNormalizer):
    """长 Section 拆分器

    将过长且没有子 section 的 section 拆分为多个子 section。

    拆分条件：
    1. section 没有子 section（只有 paragraph 子节点）
    2. 总内容长度超过 max_length

    支持两种标题生成模式：
    1. LLM 生成：使用 LLM 根据内容生成有意义的标题
    2. 简单编号：使用 "部分 1"、"部分 2" 等简单标题
    """

    # 句子分隔符
    SENTENCE_PATTERN = re.compile(r"(?<=[。！？.!?])\s*")

    def __init__(
        self,
        config: SectionSplitConfig,
        llm_func: Optional[LLMFunc] = None,
        embedding_func: Optional[EmbeddingFunc] = None,
    ):
        """初始化 Section 拆分器

        Args:
            config: 拆分配置
            llm_func: 异步 LLM 调用函数，签名为 async (prompt: str) -> str
                     如果为 None 且 use_llm_title=True，则使用简单编号
            embedding_func: 异步 embedding 函数，用于语义切分
        """
        self._config = config
        self._llm_func = llm_func
        self._embedding_func = embedding_func

    async def normalize(self, node: "ParsedNode") -> "ParsedNode":
        """规范化节点树，拆分过长的 section

        Args:
            node: 待规范化的节点

        Returns:
            规范化后的节点
        """
        if not self._config.enabled:
            return node

        await self._normalize_recursive(node)
        return node

    async def _normalize_recursive(self, node: "ParsedNode") -> None:
        """递归处理节点"""
        # 先递归处理子节点
        new_children = []
        for child in node.children:
            if child.node_type == "section" and self._should_split(child):
                # 拆分 section
                sub_sections = await self._split_section(child)
                # 用原 section 作为父容器，把子 section 作为其子节点
                child.children = sub_sections
                new_children.append(child)
            else:
                await self._normalize_recursive(child)
                new_children.append(child)

        node.children = new_children

        # 重新分配 position
        for i, child in enumerate(node.children):
            child.position = i

    def _should_split(self, node: "ParsedNode") -> bool:
        """判断 section 是否需要拆分

        条件：
        1. 没有子 section
        2. 总内容长度超过 max_length
        """
        # 检查是否有子 section
        if self._has_subsections(node):
            return False

        # 计算总内容长度
        total_length = self._get_total_content_length(node)
        return total_length > self._config.max_length

    async def _split_section(self, node: "ParsedNode") -> list["ParsedNode"]:
        """拆分 section

        Args:
            node: 待拆分的 section 节点

        Returns:
            子 section 列表
        """
        from dropqa.indexer.parser import ParsedNode

        # 收集所有段落内容
        all_content = self._collect_all_content(node)
        if not all_content:
            return node.children

        # 计算目标子 section 数量
        total_length = len(all_content)
        target_count = max(
            self._config.min_subsections,
            total_length // self._config.max_length + 1
        )

        # 切分内容
        chunks = await self._split_content(all_content, target_count)
        if len(chunks) < self._config.min_subsections:
            # 切分结果不足，不拆分
            return node.children

        # 生成子 section 标题
        titles = await self._generate_titles(chunks)

        # 创建子 section 节点
        sub_sections = []
        new_depth = node.depth + 1

        for i, (chunk, title) in enumerate(zip(chunks, titles)):
            sub_section = ParsedNode(
                node_type="section",
                depth=new_depth,
                title=title,
                position=i,
            )
            # 创建 paragraph 子节点
            para = ParsedNode(
                node_type="paragraph",
                depth=new_depth + 1,
                content=chunk,
                position=0,
            )
            sub_section.children = [para]
            sub_sections.append(sub_section)

        return sub_sections

    async def _split_content(self, content: str, target_count: int) -> list[str]:
        """将内容切分为指定数量的块

        Args:
            content: 内容文本
            target_count: 目标块数量

        Returns:
            切分后的文本块列表
        """
        # 分句
        sentences = self.SENTENCE_PATTERN.split(content)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= target_count:
            # 句子数量不足，每个句子一块
            return sentences

        # 使用语义切分或简单切分
        if self._embedding_func is not None:
            return await self._semantic_split(sentences, target_count)
        else:
            return self._simple_split(sentences, target_count)

    def _simple_split(self, sentences: list[str], target_count: int) -> list[str]:
        """简单切分：均匀分配句子

        Args:
            sentences: 句子列表
            target_count: 目标块数量

        Returns:
            切分后的文本块列表
        """
        if not sentences:
            return []

        # 计算每块的目标句子数
        sentences_per_chunk = len(sentences) // target_count
        if sentences_per_chunk < 1:
            sentences_per_chunk = 1

        chunks = []
        for i in range(target_count):
            start = i * sentences_per_chunk
            if i == target_count - 1:
                # 最后一块包含剩余所有句子
                end = len(sentences)
            else:
                end = start + sentences_per_chunk

            chunk_sentences = sentences[start:end]
            if chunk_sentences:
                chunks.append("".join(chunk_sentences))

        return chunks

    async def _semantic_split(self, sentences: list[str], target_count: int) -> list[str]:
        """语义切分：在话题转换点切分

        Args:
            sentences: 句子列表
            target_count: 目标块数量

        Returns:
            切分后的文本块列表
        """
        try:
            # 计算所有句子的 embedding
            embeddings = await self._embedding_func(sentences)
            if not embeddings or len(embeddings) != len(sentences):
                return self._simple_split(sentences, target_count)

            # 计算相邻句子的相似度
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
                similarities.append((i, sim))

            # 找到相似度最低的 (target_count - 1) 个点作为切分点
            sorted_sims = sorted(similarities, key=lambda x: x[1])
            split_points = sorted([s[0] for s in sorted_sims[:target_count - 1]])

            # 根据切分点切分句子
            chunks = []
            prev_point = 0
            for point in split_points:
                chunk_sentences = sentences[prev_point:point + 1]
                if chunk_sentences:
                    chunks.append("".join(chunk_sentences))
                prev_point = point + 1

            # 最后一块
            if prev_point < len(sentences):
                chunks.append("".join(sentences[prev_point:]))

            return chunks

        except Exception:
            return self._simple_split(sentences, target_count)

    async def _generate_titles(self, chunks: list[str]) -> list[str]:
        """为每个块生成标题

        Args:
            chunks: 文本块列表

        Returns:
            标题列表
        """
        if self._config.use_llm_title and self._llm_func is not None:
            return await self._generate_titles_with_llm(chunks)
        else:
            return self._generate_simple_titles(len(chunks))

    def _generate_simple_titles(self, count: int) -> list[str]:
        """生成简单编号标题

        Args:
            count: 标题数量

        Returns:
            标题列表
        """
        return [f"部分 {i + 1}" for i in range(count)]

    async def _generate_titles_with_llm(self, chunks: list[str]) -> list[str]:
        """使用 LLM 生成标题

        Args:
            chunks: 文本块列表

        Returns:
            标题列表
        """
        titles = []
        for i, chunk in enumerate(chunks):
            try:
                # 截取内容预览
                preview = chunk[:500] if len(chunk) > 500 else chunk

                prompt = f"""请为以下文本内容生成一个简短的标题（5-15个字）。
只输出标题文字，不要输出其他内容。

文本内容：
{preview}

标题："""

                title = await self._llm_func(prompt)
                # 清理标题
                title = title.strip().strip('"').strip("'").strip("《》")
                if title and len(title) <= 30:
                    titles.append(title)
                else:
                    titles.append(f"部分 {i + 1}")
            except Exception:
                titles.append(f"部分 {i + 1}")

        return titles

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """计算余弦相似度"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)
