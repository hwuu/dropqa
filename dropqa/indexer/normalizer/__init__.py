"""文档规范化模块

提供文档结构规范化功能，包括：
- 长段落切分
- 长 Section 拆分
- 无意义标题增强
"""

import logging
from typing import TYPE_CHECKING, Optional, Callable, Awaitable

from dropqa.common.config import NormalizationConfig
from dropqa.indexer.normalizer.paragraph_splitter import ParagraphSplitter
from dropqa.indexer.normalizer.section_splitter import SectionSplitter
from dropqa.indexer.normalizer.title_enricher import TitleEnricher

if TYPE_CHECKING:
    from dropqa.indexer.parser import ParsedNode

logger = logging.getLogger(__name__)


# 异步函数类型
LLMFunc = Callable[[str], Awaitable[str]]
EmbeddingFunc = Callable[[list[str]], Awaitable[list[list[float]]]]


class DocumentNormalizer:
    """文档规范化器

    编排多个规范化器，按正确顺序对文档进行规范化处理。

    处理顺序：
    1. Section 拆分 - 将过长的 section 拆分为子 section
    2. 段落切分 - 将过长的段落切分为多个段落
    3. 标题增强 - 增强无意义的标题（包括新创建的）

    使用示例：
        config = NormalizationConfig(enabled=True)
        normalizer = DocumentNormalizer(config, llm_func, embedding_func)
        normalized_tree = await normalizer.normalize(parsed_tree)
    """

    def __init__(
        self,
        config: NormalizationConfig,
        llm_func: Optional[LLMFunc] = None,
        embedding_func: Optional[EmbeddingFunc] = None,
    ):
        """初始化文档规范化器

        Args:
            config: 规范化配置
            llm_func: 异步 LLM 调用函数，签名为 async (prompt: str) -> str
                     用于生成标题
            embedding_func: 异步 embedding 函数，签名为 async (texts: list[str]) -> list[list[float]]
                           用于语义切分
        """
        self._config = config
        self._llm_func = llm_func
        self._embedding_func = embedding_func

        # 创建各个规范化器
        self._normalizers = []

        if config.enabled:
            # 1. Section 拆分器
            if config.section_split.enabled:
                self._normalizers.append(
                    SectionSplitter(
                        config.section_split,
                        llm_func=llm_func,
                        embedding_func=embedding_func,
                    )
                )
                logger.debug("[Normalizer] Section 拆分器已启用")

            # 2. 段落切分器
            if config.paragraph_split.enabled:
                self._normalizers.append(
                    ParagraphSplitter(
                        config.paragraph_split,
                        embedding_func=embedding_func,
                    )
                )
                logger.debug("[Normalizer] 段落切分器已启用")

            # 3. 标题增强器（最后执行，可以处理新创建的标题）
            if config.title_enrich.enabled:
                self._normalizers.append(
                    TitleEnricher(
                        config.title_enrich,
                        llm_func=llm_func,
                    )
                )
                logger.debug("[Normalizer] 标题增强器已启用")

    async def normalize(self, node: "ParsedNode") -> "ParsedNode":
        """规范化文档节点树

        Args:
            node: 待规范化的文档节点树（通常是根节点）

        Returns:
            规范化后的节点树（原地修改并返回）
        """
        if not self._config.enabled:
            logger.debug("[Normalizer] 规范化功能已禁用，跳过")
            return node

        if not self._normalizers:
            logger.debug("[Normalizer] 没有启用的规范化器，跳过")
            return node

        logger.info(f"[Normalizer] 开始规范化，启用 {len(self._normalizers)} 个规范化器")

        for normalizer in self._normalizers:
            normalizer_name = type(normalizer).__name__
            logger.debug(f"[Normalizer] 应用 {normalizer_name}")
            try:
                node = await normalizer.normalize(node)
            except Exception as e:
                logger.warning(f"[Normalizer] {normalizer_name} 处理失败: {e}")
                # 继续处理下一个规范化器

        logger.info("[Normalizer] 规范化完成")
        return node

    @property
    def enabled(self) -> bool:
        """是否启用规范化"""
        return self._config.enabled and len(self._normalizers) > 0


# 导出
__all__ = [
    "DocumentNormalizer",
    "ParagraphSplitter",
    "SectionSplitter",
    "TitleEnricher",
]
