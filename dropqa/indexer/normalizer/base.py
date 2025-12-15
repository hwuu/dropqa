"""规范化器基类

定义节点规范化器的抽象接口。
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dropqa.indexer.parser import ParsedNode


class NodeNormalizer(ABC):
    """节点规范化器基类

    所有具体的规范化器都应该继承此类，实现 normalize 方法。

    规范化器的职责：
    - 接收一个 ParsedNode 树
    - 对树进行修改（原地修改）
    - 返回修改后的树

    规范化器应该是幂等的：多次应用应该得到相同的结果。
    """

    @abstractmethod
    async def normalize(self, node: "ParsedNode") -> "ParsedNode":
        """规范化节点树

        Args:
            node: 待规范化的节点（通常是根节点）

        Returns:
            规范化后的节点（原地修改并返回）
        """
        pass

    def _get_total_content_length(self, node: "ParsedNode") -> int:
        """计算节点及其所有子孙的总内容长度

        Args:
            node: 节点

        Returns:
            总内容长度（字符数）
        """
        total = 0
        if node.content:
            total += len(node.content)
        for child in node.children:
            total += self._get_total_content_length(child)
        return total

    def _has_subsections(self, node: "ParsedNode") -> bool:
        """检查节点是否有 section 类型的子节点

        Args:
            node: 节点

        Returns:
            True 表示有子 section
        """
        for child in node.children:
            if child.node_type == "section":
                return True
        return False

    def _collect_paragraphs(self, node: "ParsedNode") -> list["ParsedNode"]:
        """收集节点下所有的 paragraph 节点

        Args:
            node: 节点

        Returns:
            paragraph 节点列表
        """
        paragraphs = []
        if node.node_type == "paragraph":
            paragraphs.append(node)
        for child in node.children:
            paragraphs.extend(self._collect_paragraphs(child))
        return paragraphs

    def _collect_all_content(self, node: "ParsedNode") -> str:
        """收集节点下所有内容，合并为一个字符串

        Args:
            node: 节点

        Returns:
            合并后的内容字符串
        """
        parts = []
        if node.content:
            parts.append(node.content)
        for child in node.children:
            child_content = self._collect_all_content(child)
            if child_content:
                parts.append(child_content)
        return "\n\n".join(parts)
