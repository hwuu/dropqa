"""数据库初始化脚本

使用方法:
    python -m dropqa.common.init_db --config config/indexer.yaml

功能:
    1. 创建数据库表 (documents, nodes)
    2. 创建全文搜索索引 (PostgreSQL: tsvector + GIN, SQLite: FTS5)
"""

import argparse
import asyncio
from pathlib import Path

from dropqa.common.config import (
    StorageBackend,
    load_indexer_config,
    create_repository_factory,
)


async def init_database(config_path: str) -> None:
    """初始化数据库"""
    print(f"加载配置: {config_path}")
    config = load_indexer_config(config_path)

    backend = config.storage.backend
    print(f"存储后端: {backend.value}")

    if backend == StorageBackend.POSTGRES:
        pg_config = config.storage.postgres
        print(f"连接数据库: {pg_config.host}:{pg_config.port}/{pg_config.name}")
    elif backend == StorageBackend.SQLITE:
        sqlite_config = config.storage.sqlite
        print(f"数据库文件: {sqlite_config.db_path}")

    # 使用 RepositoryFactory 初始化
    repo_factory = create_repository_factory(config.storage)

    try:
        print("初始化数据库...")
        await repo_factory.initialize()
        print("  ✓ 数据库初始化完成")

        print("\n数据库初始化完成！")

    finally:
        await repo_factory.close()


def main() -> None:
    """入口函数"""
    parser = argparse.ArgumentParser(description="初始化 DropQA 数据库")
    parser.add_argument(
        "--config",
        type=str,
        default="config/indexer.yaml",
        help="配置文件路径 (默认: config/indexer.yaml)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}")
        print("请复制 config/indexer.example.yaml 为 config/indexer.yaml 并修改配置")
        return

    asyncio.run(init_database(str(config_path)))


if __name__ == "__main__":
    main()
