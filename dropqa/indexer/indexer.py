"""索引写入模块"""

import hashlib
import logging
import uuid
from pathlib import Path
from typing import Optional

from sqlalchemy import delete, select

from dropqa.common.db import Database
from dropqa.common.models import Document, Node
from dropqa.common.repository.base import NodeData, SearchRepository
from dropqa.indexer.parser import flatten_nodes, parse_document

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
    """

    def __init__(
        self,
        db: Database,
        embedding_service: Optional["EmbeddingService"] = None,
        search_repo: Optional[SearchRepository] = None,
    ):
        """初始化 Indexer

        Args:
            db: 数据库实例
            embedding_service: 可选的 Embedding 服务，用于生成向量
            search_repo: 可选的搜索仓库，用于保存向量
        """
        self.db = db
        self._embedding_service = embedding_service
        self._search_repo = search_repo

    async def index_file(self, file_path: Path) -> Document:
        """索引单个文件

        如果文件已存在且内容未变化，直接返回已有记录。
        如果文件已存在但内容变化，更新记录。
        如果文件不存在，创建新记录。

        Args:
            file_path: 文件路径

        Returns:
            Document 对象
        """
        file_path = Path(file_path)
        file_hash = calculate_file_hash(file_path)
        file_size = file_path.stat().st_size
        file_type = self._extract_file_type(file_path)

        async with self.db.session() as session:
            # 检查是否已存在相同路径的文档
            existing = await self._get_document_by_path(session, str(file_path))

            if existing:
                if existing.file_hash == file_hash:
                    # 内容未变化，直接返回
                    return existing
                else:
                    # 内容变化，删除旧的 embedding 和节点并更新
                    if self._search_repo:
                        await self._search_repo.delete_embeddings_by_document(existing.id)
                    await self._delete_nodes(session, existing.id)
                    existing.file_hash = file_hash
                    existing.file_size = file_size
                    document = existing
            else:
                # 创建新文档
                document = Document(
                    id=uuid.uuid4(),
                    filename=file_path.name,
                    file_type=file_type,
                    file_hash=file_hash,
                    file_size=file_size,
                    storage_path=str(file_path),
                    current_version=1,
                )
                session.add(document)

            # 解析文件并创建节点
            parsed_root = parse_document(file_path)
            nodes_data = flatten_nodes(parsed_root, document.id, version=1)

            # 批量创建节点
            for node_data in nodes_data:
                node = Node(**node_data)
                session.add(node)

            await session.flush()

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
        async with self.db.session() as session:
            # 由于设置了 cascade，删除 document 会自动删除 nodes
            result = await session.execute(
                delete(Document).where(Document.id == document_id)
            )
            return result.rowcount > 0

    async def delete_document_by_path(self, storage_path: str) -> bool:
        """根据存储路径删除文档

        Args:
            storage_path: 文件存储路径

        Returns:
            是否成功删除
        """
        async with self.db.session() as session:
            result = await session.execute(
                delete(Document).where(Document.storage_path == storage_path)
            )
            return result.rowcount > 0

    async def get_document_by_hash(self, file_hash: str) -> Document | None:
        """根据文件哈希查询文档

        Args:
            file_hash: 文件哈希

        Returns:
            Document 对象或 None
        """
        async with self.db.session() as session:
            result = await session.execute(
                select(Document).where(Document.file_hash == file_hash)
            )
            return result.scalar_one_or_none()

    async def _get_document_by_path(self, session, storage_path: str) -> Document | None:
        """根据存储路径查询文档

        Args:
            session: 数据库会话
            storage_path: 存储路径

        Returns:
            Document 对象或 None
        """
        result = await session.execute(
            select(Document).where(Document.storage_path == storage_path)
        )
        return result.scalar_one_or_none()

    async def _delete_nodes(self, session, document_id: uuid.UUID) -> None:
        """删除文档的所有节点

        Args:
            session: 数据库会话
            document_id: 文档 ID
        """
        await session.execute(
            delete(Node).where(Node.document_id == document_id)
        )

    def _extract_file_type(self, file_path: Path) -> str:
        """提取文件类型

        Args:
            file_path: 文件路径

        Returns:
            文件扩展名（不含点号，小写）
        """
        return file_path.suffix.lstrip(".").lower()

    async def _generate_embeddings(self, nodes_data: list[dict]) -> None:
        """为节点生成并保存 embedding

        Args:
            nodes_data: 节点数据列表（dict 格式）
        """
        if not self._embedding_service or not self._search_repo:
            return

        if not nodes_data:
            return

        # 转换为 NodeData 格式
        from dropqa.common.repository.base import NodeData

        node_data_list = [
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
            for n in nodes_data
        ]

        # 构建文本（title + content）
        texts = [
            f"{n.title or ''}\n{n.content or ''}".strip()
            for n in node_data_list
        ]

        logger.debug(f"为 {len(texts)} 个节点生成 embedding")

        # 批量生成 embedding
        embeddings = await self._embedding_service.embed_batch(texts)

        # 保存到搜索仓库
        await self._search_repo.save_embeddings(
            node_data_list,
            embeddings,
            self._embedding_service.model,
        )

        logger.debug(f"已保存 {len(embeddings)} 个 embedding")
