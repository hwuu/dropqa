"""数据库连接模块"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from dropqa.common.config import PostgresConfig


class Base(DeclarativeBase):
    """SQLAlchemy 基类"""
    pass


class Database:
    """数据库管理类"""

    def __init__(self, config: PostgresConfig):
        self.config = config
        self.engine = create_async_engine(
            config.url,
            echo=False,
            pool_pre_ping=True,
        )
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def create_tables(self) -> None:
        """创建所有表"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_tables(self) -> None:
        """删除所有表（仅用于测试）"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """获取数据库会话"""
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def close(self) -> None:
        """关闭数据库连接"""
        await self.engine.dispose()


# 全局数据库实例（延迟初始化）
_db: Database | None = None


def init_db(config: PostgresConfig) -> Database:
    """初始化数据库"""
    global _db
    _db = Database(config)
    return _db


def get_db() -> Database:
    """获取数据库实例"""
    if _db is None:
        raise RuntimeError("数据库未初始化，请先调用 init_db()")
    return _db
