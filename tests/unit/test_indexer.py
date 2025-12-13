"""Indexer 索引写入测试"""

import hashlib
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dropqa.common.repository.base import DocumentData, NodeData
from dropqa.indexer.indexer import Indexer, calculate_file_hash
from dropqa.indexer.parser import MarkdownParser


class TestCalculateFileHash:
    """文件哈希计算测试"""

    def test_calculate_hash(self, tmp_path):
        """测试计算文件哈希"""
        # 创建测试文件
        test_file = tmp_path / "test.md"
        test_file.write_text("Hello World", encoding="utf-8")

        # 计算哈希
        result = calculate_file_hash(test_file)

        # 验证是 SHA256
        expected = hashlib.sha256(b"Hello World").hexdigest()
        assert result == expected
        assert len(result) == 64  # SHA256 长度

    def test_same_content_same_hash(self, tmp_path):
        """相同内容产生相同哈希"""
        file1 = tmp_path / "file1.md"
        file2 = tmp_path / "file2.md"
        file1.write_text("Same content", encoding="utf-8")
        file2.write_text("Same content", encoding="utf-8")

        assert calculate_file_hash(file1) == calculate_file_hash(file2)

    def test_different_content_different_hash(self, tmp_path):
        """不同内容产生不同哈希"""
        file1 = tmp_path / "file1.md"
        file2 = tmp_path / "file2.md"
        file1.write_text("Content A", encoding="utf-8")
        file2.write_text("Content B", encoding="utf-8")

        assert calculate_file_hash(file1) != calculate_file_hash(file2)


class TestIndexer:
    """Indexer 测试"""

    @pytest.fixture
    def mock_doc_repo(self):
        """创建模拟文档仓库"""
        repo = AsyncMock()
        repo.get_by_path = AsyncMock(return_value=None)
        repo.save = AsyncMock()
        repo.update = AsyncMock()
        repo.delete = AsyncMock(return_value=True)
        repo.get_by_hash = AsyncMock(return_value=None)
        return repo

    @pytest.fixture
    def mock_node_repo(self):
        """创建模拟节点仓库"""
        repo = AsyncMock()
        repo.save_batch = AsyncMock()
        repo.delete_by_document = AsyncMock()
        return repo

    @pytest.fixture
    def indexer(self, mock_doc_repo, mock_node_repo):
        """创建 Indexer 实例"""
        return Indexer(
            doc_repo=mock_doc_repo,
            node_repo=mock_node_repo,
        )

    @pytest.fixture
    def sample_md_file(self, tmp_path):
        """创建示例 Markdown 文件"""
        file = tmp_path / "sample.md"
        file.write_text("""# Chapter 1

Introduction paragraph.

## Section 1.1

Details here.
""", encoding="utf-8")
        return file

    def test_indexer_init(self, mock_doc_repo, mock_node_repo):
        """测试 Indexer 初始化"""
        indexer = Indexer(
            doc_repo=mock_doc_repo,
            node_repo=mock_node_repo,
        )
        assert indexer._doc_repo == mock_doc_repo
        assert indexer._node_repo == mock_node_repo

    @pytest.mark.asyncio
    async def test_index_new_file(self, indexer, sample_md_file, mock_doc_repo, mock_node_repo):
        """测试索引新文件"""
        # 执行索引
        result = await indexer.index_file(sample_md_file)

        # 验证结果
        assert result is not None
        assert result.filename == "sample.md"
        assert result.file_type == "md"
        assert len(result.file_hash) == 64

        # 验证调用了 save 和 save_batch
        mock_doc_repo.save.assert_called_once()
        mock_node_repo.save_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_index_existing_unchanged_file(self, indexer, sample_md_file, mock_doc_repo):
        """测试索引已存在且未变化的文件（应跳过）"""
        file_hash = calculate_file_hash(sample_md_file)
        doc_id = uuid.uuid4()

        # 模拟返回已存在的文档
        existing_doc = DocumentData(
            id=doc_id,
            filename="sample.md",
            file_type="md",
            file_hash=file_hash,
            file_size=100,
            storage_path=str(sample_md_file),
        )
        mock_doc_repo.get_by_path.return_value = existing_doc

        # 执行索引
        result = await indexer.index_file(sample_md_file)

        # 应返回已存在的文档，不应调用 save
        assert result.id == doc_id
        mock_doc_repo.save.assert_not_called()

    @pytest.mark.asyncio
    async def test_index_existing_changed_file(self, indexer, sample_md_file, mock_doc_repo, mock_node_repo):
        """测试索引已存在但内容变化的文件（应更新）"""
        doc_id = uuid.uuid4()

        # 模拟返回已存在的文档（但哈希不同）
        existing_doc = DocumentData(
            id=doc_id,
            filename="sample.md",
            file_type="md",
            file_hash="old_hash_different_from_current",
            file_size=100,
            storage_path=str(sample_md_file),
        )
        mock_doc_repo.get_by_path.return_value = existing_doc

        # 执行索引
        result = await indexer.index_file(sample_md_file)

        # 应更新文档
        assert result.id == doc_id
        mock_doc_repo.update.assert_called_once()
        mock_node_repo.delete_by_document.assert_called_once_with(doc_id)
        mock_node_repo.save_batch.assert_called_once()

    def test_extract_file_type(self, indexer):
        """测试提取文件类型"""
        assert indexer._extract_file_type(Path("test.md")) == "md"
        assert indexer._extract_file_type(Path("test.markdown")) == "markdown"
        assert indexer._extract_file_type(Path("test.MD")) == "md"
        assert indexer._extract_file_type(Path("path/to/file.md")) == "md"

    @pytest.mark.asyncio
    async def test_delete_document(self, indexer, mock_doc_repo, mock_node_repo):
        """测试删除文档"""
        doc_id = uuid.uuid4()

        result = await indexer.delete_document(doc_id)

        assert result is True
        mock_node_repo.delete_by_document.assert_called_once_with(doc_id)
        mock_doc_repo.delete.assert_called_once_with(doc_id)

    @pytest.mark.asyncio
    async def test_delete_document_by_path(self, indexer, mock_doc_repo, mock_node_repo):
        """测试通过路径删除文档"""
        doc_id = uuid.uuid4()
        storage_path = "/path/to/file.md"

        # 模拟 get_by_path 返回文档
        mock_doc_repo.get_by_path.return_value = DocumentData(
            id=doc_id,
            filename="file.md",
            file_type="md",
            file_hash="abc123",
            file_size=100,
            storage_path=storage_path,
        )

        result = await indexer.delete_document_by_path(storage_path)

        assert result is True
        mock_doc_repo.get_by_path.assert_called_once_with(storage_path)

    @pytest.mark.asyncio
    async def test_delete_document_by_path_not_found(self, indexer, mock_doc_repo):
        """测试删除不存在的文档"""
        mock_doc_repo.get_by_path.return_value = None

        result = await indexer.delete_document_by_path("/nonexistent/path.md")

        assert result is False


class TestIndexerIntegration:
    """Indexer 集成测试（需要真实数据库时使用）"""

    @pytest.mark.skip(reason="需要真实数据库环境")
    @pytest.mark.asyncio
    async def test_full_index_flow(self):
        """完整索引流程测试"""
        # 此测试需要真实数据库环境
        pass
