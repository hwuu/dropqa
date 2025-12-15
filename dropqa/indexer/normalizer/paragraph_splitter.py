"""长段落切分器

将过长的段落切分为多个较短的段落，支持简单切分和语义切分两种模式。
"""

import re
from typing import TYPE_CHECKING, Optional, Callable, Awaitable

from dropqa.common.config import ParagraphSplitConfig
from dropqa.indexer.normalizer.base import NodeNormalizer

if TYPE_CHECKING:
    from dropqa.indexer.parser import ParsedNode


# 异步 embedding 函数类型
EmbeddingFunc = Callable[[list[str]], Awaitable[list[list[float]]]]


class ParagraphSplitter(NodeNormalizer):
    """长段落切分器

    将超过 max_length 的段落切分为多个较短的段落。

    支持两种模式：
    1. 简单切分：在 target_size 附近的句子边界处切分
    2. 语义切分：使用 embedding 计算相邻句子相似度，在相似度最低处切分

    语义切分需要提供 embedding_func。
    """

    # 中英文句子分隔符
    SENTENCE_PATTERN = re.compile(r"(?<=[。！？.!?])\s*")

    def __init__(
        self,
        config: ParagraphSplitConfig,
        embedding_func: Optional[EmbeddingFunc] = None,
    ):
        """初始化段落切分器

        Args:
            config: 切分配置
            embedding_func: 异步 embedding 函数，签名为 async (texts: list[str]) -> list[list[float]]
                           如果为 None，则使用简单切分模式
        """
        self._config = config
        self._embedding_func = embedding_func

    async def normalize(self, node: "ParsedNode") -> "ParsedNode":
        """规范化节点树，切分过长的段落

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
        # 处理子节点（需要注意：切分可能增加子节点数量）
        new_children = []
        for child in node.children:
            if child.node_type == "paragraph" and self._should_split(child):
                # 切分段落
                split_nodes = await self._split_paragraph(child)
                new_children.extend(split_nodes)
            else:
                # 递归处理
                await self._normalize_recursive(child)
                new_children.append(child)

        node.children = new_children

        # 重新分配 position
        for i, child in enumerate(node.children):
            child.position = i

    def _should_split(self, node: "ParsedNode") -> bool:
        """判断段落是否需要切分"""
        if not node.content:
            return False
        return len(node.content) > self._config.max_length

    async def _split_paragraph(self, node: "ParsedNode") -> list["ParsedNode"]:
        """切分段落

        Args:
            node: 待切分的段落节点

        Returns:
            切分后的段落节点列表
        """
        from dropqa.indexer.parser import ParsedNode

        content = node.content
        if not content:
            return [node]

        # 分句
        sentences = self._split_into_sentences(content)
        if len(sentences) <= 1:
            # 只有一个句子，无法切分
            return [node]

        # 选择切分模式
        if self._config.use_semantic and self._embedding_func is not None:
            chunks = await self._semantic_split(sentences)
        else:
            chunks = self._simple_split(sentences)

        # 创建新节点
        result = []
        for i, chunk in enumerate(chunks):
            new_node = ParsedNode(
                node_type="paragraph",
                depth=node.depth,
                content=chunk,
                position=i,
            )
            result.append(new_node)

        return result

    def _split_into_sentences(self, text: str) -> list[str]:
        """将文本分割为句子列表

        Args:
            text: 文本

        Returns:
            句子列表
        """
        # 使用句号、感叹号、问号作为分隔符
        sentences = self.SENTENCE_PATTERN.split(text)
        # 过滤空句子
        return [s.strip() for s in sentences if s.strip()]

    def _simple_split(self, sentences: list[str]) -> list[str]:
        """简单切分：在 target_size 附近的句子边界处切分

        Args:
            sentences: 句子列表

        Returns:
            切分后的文本块列表
        """
        if not sentences:
            return []

        target_size = self._config.target_size
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # 如果当前块加上新句子不会超过 target_size 的 1.2 倍，继续添加
            if current_length + sentence_len <= target_size * 1.2:
                current_chunk.append(sentence)
                current_length += sentence_len
            else:
                # 保存当前块，开始新块
                if current_chunk:
                    chunks.append("".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_len

        # 保存最后一个块
        if current_chunk:
            chunks.append("".join(current_chunk))

        return chunks

    async def _semantic_split(self, sentences: list[str]) -> list[str]:
        """语义切分：在相邻句子相似度最低处切分

        使用滑动窗口 + embedding 相似度检测话题转换点。

        Args:
            sentences: 句子列表

        Returns:
            切分后的文本块列表
        """
        if len(sentences) <= 1:
            return ["".join(sentences)]

        try:
            # 计算所有句子的 embedding
            embeddings = await self._embedding_func(sentences)
            if not embeddings or len(embeddings) != len(sentences):
                # embedding 计算失败，回退到简单切分
                return self._simple_split(sentences)

            # 计算相邻句子的相似度
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
                similarities.append(sim)

            # 找到切分点
            return self._find_split_points(sentences, similarities)

        except Exception:
            # 任何异常都回退到简单切分
            return self._simple_split(sentences)

    def _find_split_points(
        self,
        sentences: list[str],
        similarities: list[float],
    ) -> list[str]:
        """根据相似度找到切分点

        在 target_size 附近寻找相似度最低的位置作为切分点。

        Args:
            sentences: 句子列表
            similarities: 相邻句子的相似度列表

        Returns:
            切分后的文本块列表
        """
        target_size = self._config.target_size
        threshold = self._config.similarity_threshold

        chunks = []
        current_chunk = [sentences[0]]
        current_length = len(sentences[0])

        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_len = len(sentence)
            similarity = similarities[i - 1]

            # 判断是否应该在这里切分
            should_split = False

            # 条件1：当前长度已达到 target_size 的 0.8 倍
            if current_length >= target_size * 0.8:
                # 条件2：相似度低于阈值（话题转换）
                if similarity < threshold:
                    should_split = True
                # 条件3：长度已超过 target_size 的 1.2 倍，强制切分
                elif current_length >= target_size * 1.2:
                    should_split = True

            if should_split:
                chunks.append("".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_len
            else:
                current_chunk.append(sentence)
                current_length += sentence_len

        # 保存最后一个块
        if current_chunk:
            chunks.append("".join(current_chunk))

        return chunks

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """计算两个向量的余弦相似度

        Args:
            vec1: 向量1
            vec2: 向量2

        Returns:
            余弦相似度（0-1）
        """
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)
