"""Markdown 解析器测试"""

import uuid
from pathlib import Path

import pytest

from dropqa.indexer.parser import MarkdownParser, ParsedNode, flatten_nodes


class TestMarkdownParser:
    """Markdown 解析器测试"""

    def test_parse_empty_document(self):
        """测试解析空文档"""
        parser = MarkdownParser()
        result = parser.parse("", "test.md")

        assert result.node_type == "document"
        assert result.depth == 0
        assert result.title == "test.md"
        assert result.children == []

    def test_parse_simple_heading(self):
        """测试解析简单标题"""
        parser = MarkdownParser()
        content = "# Hello World"
        result = parser.parse(content, "test.md")

        assert len(result.children) == 1
        section = result.children[0]
        assert section.node_type == "section"
        assert section.depth == 1
        assert section.title == "Hello World"

    def test_parse_multiple_headings(self):
        """测试解析多个同级标题"""
        parser = MarkdownParser()
        content = """# First
# Second
# Third"""
        result = parser.parse(content, "test.md")

        assert len(result.children) == 3
        assert result.children[0].title == "First"
        assert result.children[1].title == "Second"
        assert result.children[2].title == "Third"

    def test_parse_nested_headings(self):
        """测试解析嵌套标题"""
        parser = MarkdownParser()
        content = """# Chapter 1
## Section 1.1
### Subsection 1.1.1
## Section 1.2
# Chapter 2"""
        result = parser.parse(content, "test.md")

        # 根节点有 2 个 chapter
        assert len(result.children) == 2

        chapter1 = result.children[0]
        assert chapter1.title == "Chapter 1"
        assert len(chapter1.children) == 2  # Section 1.1 和 Section 1.2

        section11 = chapter1.children[0]
        assert section11.title == "Section 1.1"
        assert len(section11.children) == 1  # Subsection 1.1.1

        subsection = section11.children[0]
        assert subsection.title == "Subsection 1.1.1"

    def test_parse_with_paragraphs(self):
        """测试解析带段落的文档"""
        parser = MarkdownParser()
        content = """# Introduction

This is the introduction paragraph.

## Details

Here are the details.
Multiple lines in the same paragraph."""
        result = parser.parse(content, "test.md")

        # 检查结构
        assert len(result.children) == 1  # Introduction

        intro = result.children[0]
        assert intro.title == "Introduction"
        assert len(intro.children) == 2  # paragraph + Details

        # 第一个子节点是段落
        para1 = intro.children[0]
        assert para1.node_type == "paragraph"
        assert "introduction paragraph" in para1.content

        # 第二个子节点是 Details section
        details = intro.children[1]
        assert details.node_type == "section"
        assert details.title == "Details"
        assert len(details.children) == 1  # paragraph

    def test_parse_content_before_first_heading(self):
        """测试第一个标题前的内容"""
        parser = MarkdownParser()
        content = """Some content before heading.

# First Heading"""
        result = parser.parse(content, "test.md")

        # 第一个子节点是段落（挂在根节点下）
        assert len(result.children) == 2
        assert result.children[0].node_type == "paragraph"
        assert result.children[1].node_type == "section"

    def test_heading_levels(self):
        """测试各级标题"""
        parser = MarkdownParser()
        content = """# H1
## H2
### H3
#### H4
##### H5
###### H6"""
        result = parser.parse(content, "test.md")

        # 遍历检查深度
        def check_depth(node, expected_depth):
            assert node.depth == expected_depth
            for child in node.children:
                if child.node_type == "section":
                    check_depth(child, expected_depth + 1)

        check_depth(result.children[0], 1)


class TestFlattenNodes:
    """节点展平测试"""

    def test_flatten_simple(self):
        """测试简单展平"""
        root = ParsedNode(
            node_type="document",
            depth=0,
            title="test.md",
        )
        root.children.append(
            ParsedNode(
                node_type="section",
                depth=1,
                title="Chapter 1",
            )
        )

        doc_id = uuid.uuid4()
        nodes = flatten_nodes(root, doc_id)

        assert len(nodes) == 2
        assert nodes[0]["node_type"] == "document"
        assert nodes[0]["parent_id"] is None
        assert nodes[1]["node_type"] == "section"
        assert nodes[1]["parent_id"] == nodes[0]["id"]

    def test_flatten_preserves_structure(self):
        """测试展平保留结构"""
        parser = MarkdownParser()
        content = """# Chapter 1
## Section 1.1
Paragraph content"""
        root = parser.parse(content, "test.md")

        doc_id = uuid.uuid4()
        nodes = flatten_nodes(root, doc_id)

        # document, section, section, paragraph
        assert len(nodes) == 4

        # 检查类型
        types = [n["node_type"] for n in nodes]
        assert types == ["document", "section", "section", "paragraph"]

        # 检查父子关系
        doc_node = nodes[0]
        chapter_node = nodes[1]
        section_node = nodes[2]
        para_node = nodes[3]

        assert chapter_node["parent_id"] == doc_node["id"]
        assert section_node["parent_id"] == chapter_node["id"]
        assert para_node["parent_id"] == section_node["id"]
