"""Indexer 索引写入测试"""

import hashlib
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
    def mock_db(self):
        """创建模拟数据库"""
        db = MagicMock()
        db.session = MagicMock(return_value=AsyncMock())
        return db

    @pytest.fixture
    def indexer(self, mock_db):
        """创建 Indexer 实例"""
        return Indexer(mock_db)

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

    def test_indexer_init(self, mock_db):
        """测试 Indexer 初始化"""
        indexer = Indexer(mock_db)
        assert indexer.db == mock_db
        assert indexer.parser is not None

    @pytest.mark.asyncio
    async def test_index_new_file(self, indexer, sample_md_file, mock_db):
        """测试索引新文件"""
        # 模拟数据库操作
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None)))
        mock_db.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_db.session.return_value.__aexit__ = AsyncMock(return_value=None)

        # 执行索引
        result = await indexer.index_file(sample_md_file)

        # 验证结果
        assert result is not None
        assert result.filename == "sample.md"
        assert result.file_type == "md"
        assert len(result.file_hash) == 64

    @pytest.mark.asyncio
    async def test_index_existing_unchanged_file(self, indexer, sample_md_file, mock_db):
        """测试索引已存在且未变化的文件（应跳过）"""
        file_hash = calculate_file_hash(sample_md_file)

        # 模拟数据库返回已存在的文档
        existing_doc = MagicMock()
        existing_doc.id = uuid.uuid4()
        existing_doc.file_hash = file_hash

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=existing_doc)))
        mock_db.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_db.session.return_value.__aexit__ = AsyncMock(return_value=None)

        # 执行索引
        result = await indexer.index_file(sample_md_file)

        # 应返回已存在的文档，不应有新的插入
        assert result.id == existing_doc.id

    def test_extract_file_type(self, indexer):
        """测试提取文件类型"""
        assert indexer._extract_file_type(Path("test.md")) == "md"
        assert indexer._extract_file_type(Path("test.markdown")) == "markdown"
        assert indexer._extract_file_type(Path("test.MD")) == "md"
        assert indexer._extract_file_type(Path("path/to/file.md")) == "md"


class TestIndexerIntegration:
    """Indexer 集成测试（需要真实数据库时使用）"""

    @pytest.mark.skip(reason="需要真实数据库环境")
    @pytest.mark.asyncio
    async def test_full_index_flow(self):
        """完整索引流程测试"""
        # 此测试需要真实数据库环境
        pass
