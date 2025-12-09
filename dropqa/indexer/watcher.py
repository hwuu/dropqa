"""文件监控模块"""

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from dropqa.common.config import WatchConfig

if TYPE_CHECKING:
    from dropqa.indexer.indexer import Indexer

logger = logging.getLogger(__name__)


class FileWatcher:
    """文件监控器

    监控指定目录的文件变化，自动触发索引操作。
    """

    def __init__(self, config: WatchConfig, indexer: "Indexer"):
        """初始化文件监控器

        Args:
            config: 监控配置
            indexer: 索引器实例
        """
        self.config = config
        self.indexer = indexer
        self._observer: Observer | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _should_process(self, file_path: Path) -> bool:
        """判断是否应该处理该文件

        Args:
            file_path: 文件路径

        Returns:
            是否应该处理
        """
        # 跳过隐藏文件
        if file_path.name.startswith("."):
            return False

        # 检查扩展名
        suffix = file_path.suffix.lower()
        return suffix in [ext.lower() for ext in self.config.extensions]

    async def scan_existing_files(self) -> int:
        """扫描并索引现有文件

        Returns:
            处理的文件数量
        """
        count = 0
        for directory in self.config.get_directories():
            if not directory.exists():
                logger.warning(f"监控目录不存在: {directory}")
                continue

            # 递归扫描所有文件
            for file_path in directory.rglob("*"):
                if file_path.is_file() and self._should_process(file_path):
                    try:
                        await self.indexer.index_file(file_path)
                        count += 1
                        logger.info(f"已索引: {file_path}")
                    except Exception as e:
                        logger.error(f"索引失败 {file_path}: {e}")

        return count

    async def _handle_created(self, file_path: str) -> None:
        """处理文件创建事件

        Args:
            file_path: 文件路径
        """
        path = Path(file_path)
        if self._should_process(path):
            try:
                await self.indexer.index_file(path)
                logger.info(f"已索引新文件: {path}")
            except Exception as e:
                logger.error(f"索引新文件失败 {path}: {e}")

    async def _handle_modified(self, file_path: str) -> None:
        """处理文件修改事件

        Args:
            file_path: 文件路径
        """
        path = Path(file_path)
        if self._should_process(path):
            try:
                await self.indexer.index_file(path)
                logger.info(f"已更新索引: {path}")
            except Exception as e:
                logger.error(f"更新索引失败 {path}: {e}")

    async def _handle_deleted(self, file_path: str) -> None:
        """处理文件删除事件

        Args:
            file_path: 文件路径
        """
        path = Path(file_path)
        if self._should_process(path):
            try:
                await self.indexer.delete_document_by_path(str(path))
                logger.info(f"已删除索引: {path}")
            except Exception as e:
                logger.error(f"删除索引失败 {path}: {e}")

    def start_observer(self) -> None:
        """启动文件监控"""
        self._loop = asyncio.get_event_loop()
        self._observer = Observer()

        handler = _AsyncEventHandler(self, self._loop)

        for directory in self.config.get_directories():
            if directory.exists():
                self._observer.schedule(handler, str(directory), recursive=True)
                logger.info(f"开始监控: {directory}")
            else:
                logger.warning(f"监控目录不存在，跳过: {directory}")

        self._observer.start()

    def stop(self) -> None:
        """停止文件监控"""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            logger.info("文件监控已停止")

    async def start(self) -> None:
        """启动监控（包括扫描现有文件）

        完整启动流程：
        1. 扫描并索引现有文件
        2. 启动文件变化监控
        """
        logger.info("开始扫描现有文件...")
        count = await self.scan_existing_files()
        logger.info(f"扫描完成，共索引 {count} 个文件")

        self.start_observer()


class _AsyncEventHandler(FileSystemEventHandler):
    """异步事件处理器

    将 watchdog 的同步事件转换为异步处理。
    """

    def __init__(self, watcher: FileWatcher, loop: asyncio.AbstractEventLoop):
        self.watcher = watcher
        self.loop = loop

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            asyncio.run_coroutine_threadsafe(
                self.watcher._handle_created(event.src_path),
                self.loop,
            )

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            asyncio.run_coroutine_threadsafe(
                self.watcher._handle_modified(event.src_path),
                self.loop,
            )

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            asyncio.run_coroutine_threadsafe(
                self.watcher._handle_deleted(event.src_path),
                self.loop,
            )
