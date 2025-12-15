"""索引写入模块"""

import hashlib
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from dropqa.common.repository.base import (
    DocumentData,
    DocumentRepository,
    NodeData,
    NodeRepository,
    SearchRepository,
)
from dropqa.indexer.parser import flatten_nodes, parse_document

if TYPE_CHECKING:
    from dropqa.indexer.normalizer import DocumentNormalizer

logger = logging.getLogger(__name__)


def calculate_file_hash(file_path: Path) -> str:
    """计算文件的 SHA256 哈希值

    Args:
        file_path: 文件路径

    Returns:
        SHA256 哈希字符串（64 字符）
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # 分块读取以处理大文件
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


class Indexer:
    """索引管理器

    负责将解析后的文档写入数据库。
    使用 Repository 模式，支持多种存储后端。
    """

    def __init__(
        self,
        doc_repo: DocumentRepository,
        node_repo: NodeRepository,
        search_repo: Optional[SearchRepository] = None,
        embedding_service: Optional["EmbeddingService"] = None,
        normalizer: Optional["DocumentNormalizer"] = None,
    ):
        """初始化 Indexer

        Args:
            doc_repo: 文档仓库
            node_repo: 节点仓库
            search_repo: 可选的搜索仓库，用于保存向量
            embedding_service: 可选的 Embedding 服务，用于生成向量
            normalizer: 可选的文档规范化器
        """
        self._doc_repo = doc_repo
        self._node_repo = node_repo
        self._search_repo = search_repo
        self._embedding_service = embedding_service
        self._normalizer = normalizer

    async def index_file(self, file_path: Path) -> DocumentData:
        """索引单个文件

        如果文件已存在且内容未变化，直接返回已有记录。
        如果文件已存在但内容变化，更新记录。
        如果文件不存在，创建新记录。

        Args:
            file_path: 文件路径

        Returns:
            DocumentData 对象
        """
        file_path = Path(file_path)
        file_hash = calculate_file_hash(file_path)
        file_size = file_path.stat().st_size
        file_type = self._extract_file_type(file_path)

        # 检查是否已存在相同路径的文档
        existing = await self._doc_repo.get_by_path(str(file_path))

        if existing:
            if existing.file_hash == file_hash:
                # 内容未变化，直接返回
                logger.debug(f"文件未变化，跳过: {file_path}")
                return existing
            else:
                # 内容变化，删除旧的 embedding 和节点并更新
                logger.info(f"文件已更新，重新索引: {file_path}")
                if self._search_repo:
                    await self._search_repo.delete_embeddings_by_document(existing.id)
                await self._node_repo.delete_by_document(existing.id)
                existing.file_hash = file_hash
                existing.file_size = file_size
                await self._doc_repo.update(existing)
                document = existing
        else:
            # 创建新文档
            logger.info(f"索引新文件: {file_path}")
            document = DocumentData(
                id=uuid.uuid4(),
                filename=file_path.name,
                file_type=file_type,
                file_hash=file_hash,
                file_size=file_size,
                storage_path=str(file_path),
                current_version=1,
            )
            await self._doc_repo.save(document)

        # 解析文件
        parsed_root = parse_document(file_path)

        # 规范化（如果启用）
        if self._normalizer:
            logger.debug(f"应用文档规范化: {file_path.name}")
            parsed_root = await self._normalizer.normalize(parsed_root)

        # 创建节点
        nodes_data_dicts = flatten_nodes(parsed_root, document.id, version=1)

        # 转换为 NodeData 对象
        nodes_data = [
            NodeData(
                id=n["id"],
                document_id=n["document_id"],
                parent_id=n.get("parent_id"),
                node_type=n["node_type"],
                depth=n["depth"],
                title=n.get("title"),
                content=n.get("content"),
                summary=n.get("summary"),
                position=n.get("position", 0),
                version=n.get("version", 1),
            )
            for n in nodes_data_dicts
        ]

        # 批量保存节点
        await self._node_repo.save_batch(nodes_data)
        logger.debug(f"已保存 {len(nodes_data)} 个节点")

        # 生成并保存 embedding（如果配置了服务）
        await self._generate_embeddings(nodes_data)

        return document

    async def delete_document(self, document_id: uuid.UUID) -> bool:
        """删除文档及其所有节点

        Args:
            document_id: 文档 ID

        Returns:
            是否成功删除
        """
        # 先删除 embedding
        if self._search_repo:
            await self._search_repo.delete_embeddings_by_document(document_id)
        # 删除节点
        await self._node_repo.delete_by_document(document_id)
        # 删除文档
        return await self._doc_repo.delete(document_id)

    async def delete_document_by_path(self, storage_path: str) -> bool:
        """根据存储路径删除文档

        Args:
            storage_path: 文件存储路径

        Returns:
            是否成功删除
        """
        # 先获取文档以获取 ID
        doc = await self._doc_repo.get_by_path(storage_path)
        if doc:
            return await self.delete_document(doc.id)
        return False

    async def get_document_by_hash(self, file_hash: str) -> DocumentData | None:
        """根据文件哈希查询文档

        Args:
            file_hash: 文件哈希

        Returns:
            DocumentData 对象或 None
        """
        return await self._doc_repo.get_by_hash(file_hash)

    def _extract_file_type(self, file_path: Path) -> str:
        """提取文件类型

        Args:
            file_path: 文件路径

        Returns:
            文件扩展名（不含点号，小写）
        """
        return file_path.suffix.lstrip(".").lower()

    async def _generate_embeddings(self, nodes_data: list[NodeData]) -> None:
        """为节点生成并保存 embedding

        Args:
            nodes_data: 节点数据列表
        """
        if not self._embedding_service or not self._search_repo:
            return

        if not nodes_data:
            return

        # 构建文本（title + content）
        texts = [
            f"{n.title or ''}\n{n.content or ''}".strip()
            for n in nodes_data
        ]

        logger.debug(f"为 {len(texts)} 个节点生成 embedding")

        # 批量生成 embedding
        embeddings = await self._embedding_service.embed_batch(texts)

        # 保存到搜索仓库
        await self._search_repo.save_embeddings(
            nodes_data,
            embeddings,
            self._embedding_service.model,
        )

        logger.debug(f"已保存 {len(embeddings)} 个 embedding")
