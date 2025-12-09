"""FileWatcher 文件监控测试"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dropqa.common.config import WatchConfig
from dropqa.indexer.watcher import FileWatcher


class TestFileWatcher:
    """FileWatcher 测试"""

    @pytest.fixture
    def mock_indexer(self):
        """创建模拟 Indexer"""
        indexer = MagicMock()
        indexer.index_file = AsyncMock()
        indexer.delete_document = AsyncMock()
        return indexer

    @pytest.fixture
    def watch_config(self, tmp_path):
        """创建监控配置"""
        return WatchConfig(
            directories=[str(tmp_path)],
            extensions=[".md"],
        )

    @pytest.fixture
    def watcher(self, watch_config, mock_indexer):
        """创建 FileWatcher 实例"""
        return FileWatcher(watch_config, mock_indexer)

    def test_watcher_init(self, watcher, watch_config, mock_indexer):
        """测试 FileWatcher 初始化"""
        assert watcher.config == watch_config
        assert watcher.indexer == mock_indexer
        assert watcher._observer is None

    def test_should_process_file(self, watcher, tmp_path):
        """测试文件过滤逻辑"""
        # 应该处理的文件
        assert watcher._should_process(tmp_path / "test.md") is True
        assert watcher._should_process(tmp_path / "TEST.MD") is True
        assert watcher._should_process(tmp_path / "sub/test.md") is True

        # 不应该处理的文件
        assert watcher._should_process(tmp_path / "test.txt") is False
        assert watcher._should_process(tmp_path / "test.docx") is False
        assert watcher._should_process(tmp_path / ".hidden.md") is False

    @pytest.mark.asyncio
    async def test_scan_existing_files(self, watcher, mock_indexer, tmp_path):
        """测试扫描现有文件"""
        # 创建测试文件
        (tmp_path / "file1.md").write_text("# File 1", encoding="utf-8")
        (tmp_path / "file2.md").write_text("# File 2", encoding="utf-8")
        (tmp_path / "ignore.txt").write_text("ignore", encoding="utf-8")

        # 创建子目录
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file3.md").write_text("# File 3", encoding="utf-8")

        # 扫描
        count = await watcher.scan_existing_files()

        # 验证
        assert count == 3
        assert mock_indexer.index_file.call_count == 3

    @pytest.mark.asyncio
    async def test_scan_empty_directory(self, watcher, mock_indexer):
        """测试扫描空目录"""
        count = await watcher.scan_existing_files()
        assert count == 0
        assert mock_indexer.index_file.call_count == 0

    @pytest.mark.asyncio
    async def test_scan_skips_hidden_files(self, watcher, mock_indexer, tmp_path):
        """测试扫描跳过隐藏文件"""
        (tmp_path / ".hidden.md").write_text("# Hidden", encoding="utf-8")
        (tmp_path / "visible.md").write_text("# Visible", encoding="utf-8")

        count = await watcher.scan_existing_files()

        assert count == 1
        mock_indexer.index_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_file_created(self, watcher, mock_indexer, tmp_path):
        """测试处理文件创建事件"""
        file_path = tmp_path / "new.md"
        file_path.write_text("# New", encoding="utf-8")

        await watcher._handle_created(str(file_path))

        mock_indexer.index_file.assert_called_once_with(Path(file_path))

    @pytest.mark.asyncio
    async def test_handle_file_modified(self, watcher, mock_indexer, tmp_path):
        """测试处理文件修改事件"""
        file_path = tmp_path / "modified.md"
        file_path.write_text("# Modified", encoding="utf-8")

        await watcher._handle_modified(str(file_path))

        mock_indexer.index_file.assert_called_once_with(Path(file_path))

    @pytest.mark.asyncio
    async def test_handle_file_deleted(self, watcher, mock_indexer, tmp_path):
        """测试处理文件删除事件"""
        file_path = tmp_path / "deleted.md"

        await watcher._handle_deleted(str(file_path))

        mock_indexer.delete_document_by_path.assert_called_once_with(str(file_path))

    @pytest.mark.asyncio
    async def test_start_stop(self, watcher):
        """测试启动和停止监控"""
        # 启动
        watcher.start_observer()
        assert watcher._observer is not None
        assert watcher._observer.is_alive()

        # 停止
        watcher.stop()
        assert not watcher._observer.is_alive()


class TestWatchConfigDirectories:
    """监控配置目录测试"""

    def test_multiple_directories(self, tmp_path):
        """测试多目录配置"""
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        config = WatchConfig(
            directories=[str(dir1), str(dir2)],
            extensions=[".md"],
        )

        dirs = config.get_directories()
        assert len(dirs) == 2
        assert dir1.resolve() in dirs
        assert dir2.resolve() in dirs

    def test_multiple_extensions(self, tmp_path):
        """测试多扩展名配置"""
        config = WatchConfig(
            directories=[str(tmp_path)],
            extensions=[".md", ".markdown", ".txt"],
        )

        mock_indexer = MagicMock()
        watcher = FileWatcher(config, mock_indexer)

        assert watcher._should_process(tmp_path / "test.md") is True
        assert watcher._should_process(tmp_path / "test.markdown") is True
        assert watcher._should_process(tmp_path / "test.txt") is True
        assert watcher._should_process(tmp_path / "test.docx") is False
