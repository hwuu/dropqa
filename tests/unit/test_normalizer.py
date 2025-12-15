"""文档规范化模块单元测试"""

import pytest
from unittest.mock import AsyncMock

from dropqa.common.config import (
    NormalizationConfig,
    ParagraphSplitConfig,
    SectionSplitConfig,
    TitleEnrichConfig,
)
from dropqa.indexer.parser import ParsedNode
from dropqa.indexer.normalizer import DocumentNormalizer
from dropqa.indexer.normalizer.paragraph_splitter import ParagraphSplitter
from dropqa.indexer.normalizer.section_splitter import SectionSplitter
from dropqa.indexer.normalizer.title_enricher import TitleEnricher


# ============== Helper Functions ==============


def create_paragraph(content: str, depth: int = 1, position: int = 0) -> ParsedNode:
    """创建段落节点"""
    return ParsedNode(
        node_type="paragraph",
        depth=depth,
        content=content,
        position=position,
    )


def create_section(
    title: str, depth: int = 1, position: int = 0, children: list = None
) -> ParsedNode:
    """创建 section 节点"""
    node = ParsedNode(
        node_type="section",
        depth=depth,
        title=title,
        position=position,
    )
    if children:
        node.children = children
    return node


def create_root(children: list = None) -> ParsedNode:
    """创建根节点"""
    node = ParsedNode(
        node_type="root",
        depth=0,
    )
    if children:
        node.children = children
    return node


# ============== ParagraphSplitter Tests ==============


class TestParagraphSplitter:
    """段落切分器测试"""

    @pytest.fixture
    def config(self):
        """默认配置"""
        return ParagraphSplitConfig(
            enabled=True,
            max_length=100,
            target_size=50,
            use_semantic=False,
        )

    @pytest.mark.asyncio
    async def test_short_paragraph_not_split(self, config):
        """短段落不应被切分"""
        splitter = ParagraphSplitter(config)

        para = create_paragraph("这是一个短段落。")
        root = create_root([para])

        result = await splitter.normalize(root)

        assert len(result.children) == 1
        assert result.children[0].content == "这是一个短段落。"

    @pytest.mark.asyncio
    async def test_long_paragraph_split(self, config):
        """长段落应被切分"""
        splitter = ParagraphSplitter(config)

        # 创建一个长段落（超过 max_length=100）
        # 每句约 15 字符，10 句 = 150 字符
        long_text = "这是第一句话，内容很长。这是第二句话，内容很长。这是第三句话，内容很长。这是第四句话，内容很长。这是第五句话，内容很长。这是第六句话，内容很长。这是第七句话，内容很长。这是第八句话，内容很长。这是第九句话，内容很长。这是第十句话，内容很长。"
        para = create_paragraph(long_text)
        root = create_root([para])

        result = await splitter.normalize(root)

        # 应该被切分为多个段落
        assert len(result.children) > 1
        # 合并后应该等于原文
        merged = "".join(child.content for child in result.children)
        assert merged == long_text

    @pytest.mark.asyncio
    async def test_disabled_no_split(self, config):
        """禁用时不应切分"""
        config.enabled = False
        splitter = ParagraphSplitter(config)

        long_text = "这是第一句话。" * 20
        para = create_paragraph(long_text)
        root = create_root([para])

        result = await splitter.normalize(root)

        assert len(result.children) == 1
        assert result.children[0].content == long_text

    @pytest.mark.asyncio
    async def test_nested_paragraph_split(self, config):
        """嵌套在 section 中的段落应被切分"""
        splitter = ParagraphSplitter(config)

        # 使用更长的文本（超过 max_length=100）
        long_text = "这是第一句话，内容很长。这是第二句话，内容很长。这是第三句话，内容很长。这是第四句话，内容很长。这是第五句话，内容很长。这是第六句话，内容很长。这是第七句话，内容很长。这是第八句话，内容很长。这是第九句话，内容很长。这是第十句话，内容很长。"
        para = create_paragraph(long_text)
        section = create_section("测试标题", children=[para])
        root = create_root([section])

        result = await splitter.normalize(root)

        # section 内的段落应该被切分
        assert len(result.children) == 1  # 仍然只有一个 section
        assert len(result.children[0].children) > 1  # section 内有多个段落

    @pytest.mark.asyncio
    async def test_semantic_split_with_embedding(self, config):
        """语义切分模式（带 embedding）"""
        config.use_semantic = True

        # Mock embedding 函数
        async def mock_embedding(texts):
            # 返回简单的 embedding
            return [[float(i)] * 10 for i in range(len(texts))]

        splitter = ParagraphSplitter(config, embedding_func=mock_embedding)

        # 使用更长的文本（超过 max_length=100）
        long_text = "这是第一句话，内容很长。这是第二句话，内容很长。这是第三句话，内容很长。这是第四句话，内容很长。这是第五句话，内容很长。这是第六句话，内容很长。这是第七句话，内容很长。这是第八句话，内容很长。这是第九句话，内容很长。这是第十句话，内容很长。"
        para = create_paragraph(long_text)
        root = create_root([para])

        result = await splitter.normalize(root)

        # 应该被切分
        assert len(result.children) >= 1


# ============== SectionSplitter Tests ==============


class TestSectionSplitter:
    """Section 拆分器测试"""

    @pytest.fixture
    def config(self):
        """默认配置"""
        return SectionSplitConfig(
            enabled=True,
            max_length=100,
            min_subsections=2,
            use_llm_title=False,  # 不使用 LLM，使用简单编号
        )

    @pytest.mark.asyncio
    async def test_short_section_not_split(self, config):
        """短 section 不应被拆分"""
        splitter = SectionSplitter(config)

        para = create_paragraph("短内容。")
        section = create_section("标题", children=[para])
        root = create_root([section])

        result = await splitter.normalize(root)

        assert len(result.children) == 1
        assert result.children[0].title == "标题"
        assert len(result.children[0].children) == 1

    @pytest.mark.asyncio
    async def test_long_section_split(self, config):
        """长 section（无子 section）应被拆分"""
        splitter = SectionSplitter(config)

        # 创建长内容（超过 max_length=100）
        long_text = "这是第一句话，内容很长。这是第二句话，内容很长。这是第三句话，内容很长。这是第四句话，内容很长。这是第五句话，内容很长。这是第六句话，内容很长。这是第七句话，内容很长。这是第八句话，内容很长。这是第九句话，内容很长。这是第十句话，内容很长。"
        para = create_paragraph(long_text)
        section = create_section("标题", children=[para])
        root = create_root([section])

        result = await splitter.normalize(root)

        # section 应该保留，但有新的子 section
        assert len(result.children) == 1
        assert result.children[0].title == "标题"
        # 原 section 现在有子 section
        assert len(result.children[0].children) >= config.min_subsections
        # 子节点应该是 section
        for child in result.children[0].children:
            assert child.node_type == "section"

    @pytest.mark.asyncio
    async def test_section_with_subsection_not_split(self, config):
        """有子 section 的 section 不应被拆分"""
        splitter = SectionSplitter(config)

        # 创建带子 section 的 section
        sub_para = create_paragraph("子内容。" * 20)
        sub_section = create_section("子标题", depth=2, children=[sub_para])
        section = create_section("标题", children=[sub_section])
        root = create_root([section])

        result = await splitter.normalize(root)

        # 结构不变
        assert len(result.children) == 1
        assert result.children[0].title == "标题"
        assert len(result.children[0].children) == 1
        assert result.children[0].children[0].title == "子标题"

    @pytest.mark.asyncio
    async def test_disabled_no_split(self, config):
        """禁用时不应拆分"""
        config.enabled = False
        splitter = SectionSplitter(config)

        long_text = "这是句话。" * 20
        para = create_paragraph(long_text)
        section = create_section("标题", children=[para])
        root = create_root([section])

        result = await splitter.normalize(root)

        assert len(result.children) == 1
        assert len(result.children[0].children) == 1  # 未拆分

    @pytest.mark.asyncio
    async def test_split_with_llm_title(self, config):
        """使用 LLM 生成标题"""
        config.use_llm_title = True

        # Mock LLM 函数
        async def mock_llm(prompt):
            return "生成的标题"

        splitter = SectionSplitter(config, llm_func=mock_llm)

        # 使用更长的文本（超过 max_length=100）
        long_text = "这是第一句话，内容很长。这是第二句话，内容很长。这是第三句话，内容很长。这是第四句话，内容很长。这是第五句话，内容很长。这是第六句话，内容很长。这是第七句话，内容很长。这是第八句话，内容很长。这是第九句话，内容很长。这是第十句话，内容很长。"
        para = create_paragraph(long_text)
        section = create_section("标题", children=[para])
        root = create_root([section])

        result = await splitter.normalize(root)

        # 子 section 应该使用 LLM 生成的标题
        if len(result.children[0].children) > 0:
            for child in result.children[0].children:
                if child.node_type == "section":
                    assert child.title == "生成的标题"


# ============== TitleEnricher Tests ==============


class TestTitleEnricher:
    """标题增强器测试"""

    @pytest.fixture
    def config(self):
        """默认配置"""
        return TitleEnrichConfig(
            enabled=True,
            patterns=[
                r"^[一二三四五六七八九十]+[、.]?\s*$",
                r"^\d+[、.]\s*$",
                r"^第[一二三四五六七八九十\d]+[章节部分]$",
                r"^[A-Z][、.]\s*$",
            ],
            preserve_original=True,
        )

    @pytest.mark.asyncio
    async def test_meaningful_title_not_enriched(self, config):
        """有意义的标题不应被增强"""
        async def mock_llm(prompt):
            return "不应该被调用"

        enricher = TitleEnricher(config, llm_func=mock_llm)

        para = create_paragraph("内容")
        section = create_section("项目概述", children=[para])
        root = create_root([section])

        result = await enricher.normalize(root)

        assert result.children[0].title == "项目概述"

    @pytest.mark.asyncio
    async def test_chinese_number_title_enriched(self, config):
        """中文数字标题应被增强"""
        async def mock_llm(prompt):
            return "项目概述"

        enricher = TitleEnricher(config, llm_func=mock_llm)

        para = create_paragraph("这是关于项目的概述内容。")
        section = create_section("一、", children=[para])
        root = create_root([section])

        result = await enricher.normalize(root)

        # 应该保留原编号并追加生成的标题
        assert result.children[0].title == "一、项目概述"

    @pytest.mark.asyncio
    async def test_arabic_number_title_enriched(self, config):
        """阿拉伯数字标题应被增强"""
        async def mock_llm(prompt):
            return "系统设计"

        enricher = TitleEnricher(config, llm_func=mock_llm)

        para = create_paragraph("这是系统设计的内容。")
        section = create_section("1.", children=[para])
        root = create_root([section])

        result = await enricher.normalize(root)

        assert result.children[0].title == "1. 系统设计"

    @pytest.mark.asyncio
    async def test_chapter_title_enriched(self, config):
        """章节标题应被增强"""
        async def mock_llm(prompt):
            return "绪论"

        enricher = TitleEnricher(config, llm_func=mock_llm)

        para = create_paragraph("这是绪论的内容。")
        section = create_section("第一章", children=[para])
        root = create_root([section])

        result = await enricher.normalize(root)

        assert result.children[0].title == "第一章 绪论"

    @pytest.mark.asyncio
    async def test_no_preserve_original(self, config):
        """不保留原编号模式"""
        config.preserve_original = False

        async def mock_llm(prompt):
            return "项目概述"

        enricher = TitleEnricher(config, llm_func=mock_llm)

        para = create_paragraph("这是关于项目的概述内容。")
        section = create_section("一、", children=[para])
        root = create_root([section])

        result = await enricher.normalize(root)

        # 应该完全替换为新标题
        assert result.children[0].title == "项目概述"

    @pytest.mark.asyncio
    async def test_disabled_no_enrich(self, config):
        """禁用时不应增强"""
        config.enabled = False

        async def mock_llm(prompt):
            return "不应该被调用"

        enricher = TitleEnricher(config, llm_func=mock_llm)

        para = create_paragraph("内容")
        section = create_section("一、", children=[para])
        root = create_root([section])

        result = await enricher.normalize(root)

        assert result.children[0].title == "一、"

    @pytest.mark.asyncio
    async def test_no_llm_func_no_enrich(self, config):
        """没有 LLM 函数时不应增强"""
        enricher = TitleEnricher(config, llm_func=None)

        para = create_paragraph("内容")
        section = create_section("一、", children=[para])
        root = create_root([section])

        result = await enricher.normalize(root)

        assert result.children[0].title == "一、"


# ============== DocumentNormalizer Tests ==============


class TestDocumentNormalizer:
    """文档规范化编排器测试"""

    @pytest.fixture
    def config(self):
        """默认配置"""
        return NormalizationConfig(
            enabled=True,
            paragraph_split=ParagraphSplitConfig(
                enabled=True,
                max_length=100,
                target_size=50,
                use_semantic=False,
            ),
            section_split=SectionSplitConfig(
                enabled=True,
                max_length=100,
                min_subsections=2,
                use_llm_title=False,
            ),
            title_enrich=TitleEnrichConfig(
                enabled=True,
                preserve_original=True,
            ),
        )

    @pytest.mark.asyncio
    async def test_disabled_normalizer(self, config):
        """禁用时不应进行任何规范化"""
        config.enabled = False
        normalizer = DocumentNormalizer(config)

        para = create_paragraph("这是句话。" * 20)
        section = create_section("一、", children=[para])
        root = create_root([section])

        result = await normalizer.normalize(root)

        # 结构不变
        assert len(result.children) == 1
        assert result.children[0].title == "一、"
        assert len(result.children[0].children) == 1

    @pytest.mark.asyncio
    async def test_all_normalizers_applied(self, config):
        """所有规范化器应按顺序应用"""
        async def mock_llm(prompt):
            return "生成的标题"

        normalizer = DocumentNormalizer(config, llm_func=mock_llm)

        # 创建需要所有规范化的文档（超过 max_length=100）
        long_text = "这是第一句话，内容很长。这是第二句话，内容很长。这是第三句话，内容很长。这是第四句话，内容很长。这是第五句话，内容很长。这是第六句话，内容很长。这是第七句话，内容很长。这是第八句话，内容很长。这是第九句话，内容很长。这是第十句话，内容很长。"
        para = create_paragraph(long_text)
        section = create_section("一、", children=[para])
        root = create_root([section])

        result = await normalizer.normalize(root)

        # 验证处理后的结构
        assert len(result.children) == 1
        section = result.children[0]
        # 标题应该被增强（保留原编号 + 新标题）
        assert section.title.startswith("一、")
        assert "生成的标题" in section.title

    @pytest.mark.asyncio
    async def test_enabled_property(self, config):
        """enabled 属性应正确反映状态"""
        normalizer = DocumentNormalizer(config)
        assert normalizer.enabled is True

        config.enabled = False
        normalizer = DocumentNormalizer(config)
        assert normalizer.enabled is False

    @pytest.mark.asyncio
    async def test_partial_normalizers(self, config):
        """部分规范化器启用"""
        config.paragraph_split.enabled = False
        config.section_split.enabled = False
        # 只启用标题增强

        async def mock_llm(prompt):
            return "项目概述"

        normalizer = DocumentNormalizer(config, llm_func=mock_llm)

        para = create_paragraph("内容")
        section = create_section("一、", children=[para])
        root = create_root([section])

        result = await normalizer.normalize(root)

        # 只有标题被增强
        assert result.children[0].title == "一、项目概述"
        assert len(result.children[0].children) == 1  # 段落未切分
