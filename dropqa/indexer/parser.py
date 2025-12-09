"""Markdown 文档解析器"""

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ParsedNode:
    """解析后的节点"""
    node_type: str  # document, section, paragraph
    depth: int
    title: Optional[str] = None
    content: Optional[str] = None
    position: int = 0
    children: list["ParsedNode"] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"ParsedNode(type={self.node_type}, depth={self.depth}, title={self.title!r})"


class MarkdownParser:
    """Markdown 解析器

    将 Markdown 文档解析为层级节点树。
    支持标题层级 (#, ##, ###, ...)
    """

    # 匹配 Markdown 标题
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")

    def parse(self, content: str, filename: str) -> ParsedNode:
        """解析 Markdown 内容

        Args:
            content: Markdown 文本内容
            filename: 文件名（用于根节点标题）

        Returns:
            解析后的文档节点树
        """
        lines = content.split("\n")

        # 创建根节点（document）
        root = ParsedNode(
            node_type="document",
            depth=0,
            title=filename,
        )

        # 解析行
        current_paragraph_lines: list[str] = []
        position = 0

        for line in lines:
            heading_match = self.HEADING_PATTERN.match(line)

            if heading_match:
                # 先保存之前积累的段落
                if current_paragraph_lines:
                    para_content = "\n".join(current_paragraph_lines).strip()
                    if para_content:
                        self._add_paragraph(root, para_content, position)
                        position += 1
                    current_paragraph_lines = []

                # 解析标题
                level = len(heading_match.group(1))  # # 数量
                title = heading_match.group(2).strip()

                section = ParsedNode(
                    node_type="section",
                    depth=level,
                    title=title,
                    position=position,
                )
                self._insert_node(root, section)
                position += 1
            else:
                # 收集段落行
                current_paragraph_lines.append(line)

        # 处理最后的段落
        if current_paragraph_lines:
            para_content = "\n".join(current_paragraph_lines).strip()
            if para_content:
                self._add_paragraph(root, para_content, position)

        return root

    def parse_file(self, file_path: Path) -> ParsedNode:
        """解析 Markdown 文件

        Args:
            file_path: 文件路径

        Returns:
            解析后的文档节点树
        """
        content = file_path.read_text(encoding="utf-8")
        return self.parse(content, file_path.name)

    def _insert_node(self, root: ParsedNode, node: ParsedNode) -> None:
        """将节点插入到正确的位置

        根据 depth 找到合适的父节点。
        """
        # 从根节点开始，找到合适的父节点
        parent = self._find_parent(root, node.depth)
        parent.children.append(node)

    def _find_parent(self, root: ParsedNode, target_depth: int) -> ParsedNode:
        """找到目标深度的父节点

        例如：目标 depth=2 (##)，应该找到 depth=1 (#) 的最后一个节点
        如果没有 depth=1，则挂在根节点下
        """
        if target_depth <= 1:
            return root

        # 从根节点递归查找
        return self._find_parent_recursive(root, target_depth)

    def _find_parent_recursive(self, node: ParsedNode, target_depth: int) -> ParsedNode:
        """递归查找父节点"""
        # 如果当前节点的子节点中没有 section，返回当前节点
        if not node.children:
            return node

        # 找最后一个 section 子节点
        last_section = None
        for child in reversed(node.children):
            if child.node_type == "section":
                last_section = child
                break

        if last_section is None:
            return node

        # 如果最后一个 section 的 depth < target_depth - 1，继续往下找
        if last_section.depth < target_depth - 1:
            return self._find_parent_recursive(last_section, target_depth)

        # 如果最后一个 section 的 depth == target_depth - 1，它就是父节点
        if last_section.depth == target_depth - 1:
            return last_section

        # 如果最后一个 section 的 depth >= target_depth，返回当前节点
        return node

    def _add_paragraph(self, root: ParsedNode, content: str, position: int) -> None:
        """添加段落节点"""
        # 找到当前最深的 section
        parent = self._find_last_section(root)
        para = ParsedNode(
            node_type="paragraph",
            depth=parent.depth + 1,
            content=content,
            position=position,
        )
        parent.children.append(para)

    def _find_last_section(self, node: ParsedNode) -> ParsedNode:
        """找到最后一个 section（用于添加段落）"""
        if not node.children:
            return node

        # 找最后一个 section 子节点
        for child in reversed(node.children):
            if child.node_type == "section":
                return self._find_last_section(child)

        return node


def flatten_nodes(root: ParsedNode, document_id: uuid.UUID, version: int = 1) -> list[dict]:
    """将节点树展平为列表（用于数据库插入）

    Args:
        root: 根节点
        document_id: 文档 ID
        version: 版本号

    Returns:
        展平后的节点列表，每个节点包含 id 和 parent_id
    """
    nodes = []
    _flatten_recursive(root, None, document_id, version, nodes)
    return nodes


def _flatten_recursive(
    node: ParsedNode,
    parent_id: Optional[uuid.UUID],
    document_id: uuid.UUID,
    version: int,
    nodes: list[dict],
) -> uuid.UUID:
    """递归展平节点"""
    node_id = uuid.uuid4()

    nodes.append({
        "id": node_id,
        "document_id": document_id,
        "version": version,
        "parent_id": parent_id,
        "node_type": node.node_type,
        "depth": node.depth,
        "title": node.title,
        "content": node.content,
        "position": node.position,
    })

    for child in node.children:
        _flatten_recursive(child, node_id, document_id, version, nodes)

    return node_id
