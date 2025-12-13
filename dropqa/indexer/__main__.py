"""Indexer 服务入口

使用方法:
    python -m dropqa.indexer --config config/indexer.yaml
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

from dropqa.common.config import create_repository_factory, load_indexer_config
from dropqa.common.embedding import EmbeddingService
from dropqa.indexer.indexer import Indexer
from dropqa.indexer.watcher import FileWatcher

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


async def main(config_path: str) -> None:
    """主入口函数

    Args:
        config_path: 配置文件路径
    """
    # 1. 加载配置
    logger.info(f"加载配置: {config_path}")
    config = load_indexer_config(config_path)

    # 2. 初始化 Repository Factory
    repo_factory = create_repository_factory(config.storage)
    await repo_factory.initialize()
    logger.info(f"存储后端: {config.storage.backend.value}")

    # 3. 获取 Repository 实例
    doc_repo = repo_factory.get_document_repository()
    node_repo = repo_factory.get_node_repository()
    search_repo = repo_factory.get_search_repository()

    # 4. 初始化 Embedding 服务（可选）
    embedding_service = None
    try:
        embedding_service = EmbeddingService(config.embedding)
        logger.info(f"Embedding 模型: {config.embedding.model}")
    except Exception as e:
        logger.warning(f"Embedding 服务初始化失败，将跳过向量索引: {e}")

    # 5. 创建 Indexer 和 FileWatcher
    indexer = Indexer(
        doc_repo=doc_repo,
        node_repo=node_repo,
        search_repo=search_repo,
        embedding_service=embedding_service,
    )
    watcher = FileWatcher(config.watch, indexer)

    # 6. 设置退出信号处理
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("收到退出信号，正在停止...")
        stop_event.set()

    # 注册信号处理（Windows 和 Unix 兼容）
    if sys.platform != "win32":
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)
    else:
        # Windows 下使用不同方式处理 Ctrl+C
        signal.signal(signal.SIGINT, lambda s, f: signal_handler())

    try:
        # 7. 启动监控
        await watcher.start()
        logger.info("Indexer 服务已启动，按 Ctrl+C 退出")

        # 8. 等待退出信号
        await stop_event.wait()

    except KeyboardInterrupt:
        logger.info("收到键盘中断")
    finally:
        # 9. 清理资源
        watcher.stop()
        await repo_factory.close()
        logger.info("Indexer 服务已停止")


def cli() -> None:
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="DropQA 文档索引服务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/indexer.yaml",
        help="配置文件路径 (默认: config/indexer.yaml)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        logger.info("请复制 config/indexer.example.yaml 为 config/indexer.yaml 并修改配置")
        sys.exit(1)

    asyncio.run(main(str(config_path)))


if __name__ == "__main__":
    cli()
