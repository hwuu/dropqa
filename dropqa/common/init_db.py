"""数据库初始化脚本

使用方法:
    python -m dropqa.common.init_db --config config/indexer.yaml

功能:
    1. 创建数据库表 (documents, nodes)
    2. 创建全文搜索索引 (tsvector + GIN)
"""

import argparse
import asyncio
from pathlib import Path

from sqlalchemy import text

from dropqa.common.config import load_indexer_config
from dropqa.common.db import Database
from dropqa.common.models import Base


# 全文搜索相关 SQL
FULLTEXT_SEARCH_SQL = """
-- 添加全文搜索向量列（如果不存在）
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'nodes' AND column_name = 'search_vector'
    ) THEN
        ALTER TABLE nodes ADD COLUMN search_vector tsvector;
    END IF;
END $$;

-- 创建或替换更新搜索向量的函数
-- 注意：MVP 阶段使用 simple 配置，后续可切换为 chinese（需要 pg_jieba）
CREATE OR REPLACE FUNCTION update_nodes_search_vector()
RETURNS trigger AS $$
BEGIN
    NEW.search_vector :=
        setweight(to_tsvector('simple', COALESCE(NEW.title, '')), 'A') ||
        setweight(to_tsvector('simple', COALESCE(NEW.content, '')), 'B') ||
        setweight(to_tsvector('simple', COALESCE(NEW.summary, '')), 'C');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 创建触发器（如果不存在）
DROP TRIGGER IF EXISTS nodes_search_vector_update ON nodes;
CREATE TRIGGER nodes_search_vector_update
    BEFORE INSERT OR UPDATE ON nodes
    FOR EACH ROW EXECUTE FUNCTION update_nodes_search_vector();

-- 创建 GIN 索引（如果不存在）
CREATE INDEX IF NOT EXISTS nodes_search_idx ON nodes USING gin(search_vector);

-- 更新现有数据的搜索向量
UPDATE nodes SET search_vector =
    setweight(to_tsvector('simple', COALESCE(title, '')), 'A') ||
    setweight(to_tsvector('simple', COALESCE(content, '')), 'B') ||
    setweight(to_tsvector('simple', COALESCE(summary, '')), 'C')
WHERE search_vector IS NULL;
"""


async def init_database(config_path: str) -> None:
    """初始化数据库"""
    print(f"加载配置: {config_path}")
    config = load_indexer_config(config_path)

    print(f"连接数据库: {config.database.host}:{config.database.port}/{config.database.name}")
    db = Database(config.database)

    try:
        # 1. 创建表
        print("创建数据库表...")
        await db.create_tables()
        print("  ✓ 表创建完成")

        # 2. 创建全文搜索索引
        print("创建全文搜索索引...")
        async with db.engine.begin() as conn:
            await conn.execute(text(FULLTEXT_SEARCH_SQL))
        print("  ✓ 全文搜索索引创建完成")

        print("\n数据库初始化完成！")

    finally:
        await db.close()


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
