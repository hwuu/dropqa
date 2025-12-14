"""配置管理模块"""

import os
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class StorageBackend(str, Enum):
    """存储后端类型"""
    POSTGRES = "postgres"
    SQLITE = "sqlite"


class PostgresConfig(BaseModel):
    """PostgreSQL 数据库配置"""
    host: str = "localhost"
    port: int = 5432
    name: str = "dropqa"
    user: str = "postgres"
    password: str = ""

    @property
    def url(self) -> str:
        """生成数据库连接 URL"""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def sync_url(self) -> str:
        """生成同步数据库连接 URL"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class SQLiteConfig(BaseModel):
    """SQLite 配置"""
    db_path: str = "./data/dropqa.db"
    chroma_path: str = "./data/chroma"

    def get_db_path(self) -> Path:
        """获取数据库文件路径"""
        return Path(self.db_path).expanduser().resolve()

    def get_chroma_path(self) -> Path:
        """获取 ChromaDB 路径"""
        return Path(self.chroma_path).expanduser().resolve()


class StorageConfig(BaseModel):
    """存储配置"""
    backend: StorageBackend = StorageBackend.POSTGRES
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    sqlite: SQLiteConfig = Field(default_factory=SQLiteConfig)


class WatchConfig(BaseModel):
    """文件监控配置"""
    directories: list[str] = Field(default_factory=lambda: ["~/dropqa_watching"])
    extensions: list[str] = Field(default_factory=lambda: [".md"])

    def get_directories(self) -> list[Path]:
        """获取展开后的目录路径列表"""
        return [Path(d).expanduser().resolve() for d in self.directories]


class LLMConfig(BaseModel):
    """LLM 配置"""
    api_base: str = "http://localhost:11434/v1"
    api_key: str = "sk-"
    model: str = "Qwen/Qwen3-32B"
    temperature: float = 0.2
    max_tokens: int = 16384
    system_prompt: str = ""


class EmbeddingConfig(BaseModel):
    """Embedding 模型配置"""
    api_base: str = "http://localhost:11435/v1"
    api_key: str = "sk-"
    model: str = "Qwen/Qwen3-Embedding-4B"
    dimension: int = 2560


class ServerConfig(BaseModel):
    """HTTP 服务配置"""
    host: str = "0.0.0.0"
    port: int = 8000


class FulltextConfig(BaseModel):
    """全文搜索配置"""
    language: str = "chinese"
    weights: dict[str, str] = Field(default_factory=lambda: {"title": "A", "content": "B", "summary": "C"})
    min_rank: float = 0.1


class SearchConfig(BaseModel):
    """搜索配置"""
    default_strategy: str = "fulltext"
    fulltext: FulltextConfig = Field(default_factory=FulltextConfig)


class RetrievalConfig(BaseModel):
    """检索配置"""
    top_k: int = 10
    relevance_threshold: float = 0.7


class IndexerConfig(BaseModel):
    """Indexer 服务配置"""
    storage: StorageConfig = Field(default_factory=StorageConfig)
    # 保留 database 字段以保持向后兼容
    database: PostgresConfig = Field(default_factory=PostgresConfig)
    watch: WatchConfig = Field(default_factory=WatchConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)


class ServerAppConfig(BaseModel):
    """Server 服务配置"""
    server: ServerConfig = Field(default_factory=ServerConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    # 保留 database 字段以保持向后兼容
    database: PostgresConfig = Field(default_factory=PostgresConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    agentic: "AgenticConfig" = Field(default_factory=lambda: _get_agentic_config())


def _get_agentic_config() -> "AgenticConfig":
    """延迟导入 AgenticConfig 以避免循环依赖"""
    from dropqa.server.agentic.config import AgenticConfig
    return AgenticConfig()


def _expand_env_vars(config: dict[str, Any]) -> dict[str, Any]:
    """递归展开配置中的环境变量"""
    result = {}
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = _expand_env_vars(value)
        elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            result[key] = os.environ.get(env_var, "")
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path, config_class: type[BaseModel]) -> BaseModel:
    """加载配置文件

    Args:
        config_path: 配置文件路径
        config_class: 配置类（IndexerConfig 或 ServerAppConfig）

    Returns:
        配置对象
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    # 展开环境变量
    config = _expand_env_vars(raw_config)

    return config_class.model_validate(config)


def load_indexer_config(config_path: str | Path) -> IndexerConfig:
    """加载 Indexer 配置"""
    return load_config(config_path, IndexerConfig)  # type: ignore


def load_server_config(config_path: str | Path) -> ServerAppConfig:
    """加载 Server 配置"""
    return load_config(config_path, ServerAppConfig)  # type: ignore


def create_repository_factory(storage_config: StorageConfig) -> "RepositoryFactory":
    """根据配置创建 RepositoryFactory

    Args:
        storage_config: 存储配置

    Returns:
        RepositoryFactory 实例
    """
    from dropqa.common.repository import PostgresRepositoryFactory, SQLiteRepositoryFactory

    if storage_config.backend == StorageBackend.POSTGRES:
        return PostgresRepositoryFactory(storage_config.postgres)
    elif storage_config.backend == StorageBackend.SQLITE:
        return SQLiteRepositoryFactory(storage_config.sqlite)
    else:
        raise ValueError(f"不支持的存储后端: {storage_config.backend}")


def _rebuild_server_config() -> None:
    """重建 ServerAppConfig 模型以解析前向引用

    在模块加载时自动调用，确保 ServerAppConfig 可直接使用。
    """
    from dropqa.server.agentic.config import AgenticConfig
    ServerAppConfig.model_rebuild()


# 模块加载时自动重建模型
_rebuild_server_config()

