"""配置模块测试"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from dropqa.common.config import (
    DatabaseConfig,
    IndexerConfig,
    ServerAppConfig,
    WatchConfig,
    load_indexer_config,
    load_server_config,
)


class TestDatabaseConfig:
    """数据库配置测试"""

    def test_default_values(self):
        """测试默认值"""
        config = DatabaseConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.name == "dropqa"
        assert config.user == "postgres"
        assert config.password == ""

    def test_url_generation(self):
        """测试 URL 生成"""
        config = DatabaseConfig(
            host="db.example.com",
            port=5433,
            name="testdb",
            user="testuser",
            password="testpass",
        )
        assert config.url == "postgresql+asyncpg://testuser:testpass@db.example.com:5433/testdb"
        assert config.sync_url == "postgresql://testuser:testpass@db.example.com:5433/testdb"


class TestWatchConfig:
    """文件监控配置测试"""

    def test_default_values(self):
        """测试默认值"""
        config = WatchConfig()
        assert config.directories == ["~/dropqa_watching"]
        assert config.extensions == [".md"]

    def test_get_directories_expands_tilde(self):
        """测试路径展开"""
        config = WatchConfig(directories=["~/test_dir"])
        dirs = config.get_directories()
        assert len(dirs) == 1
        assert "~" not in str(dirs[0])
        assert dirs[0].is_absolute()


class TestLoadConfig:
    """配置加载测试"""

    def test_load_indexer_config(self, tmp_path: Path):
        """测试加载 Indexer 配置"""
        config_content = {
            "database": {
                "host": "testhost",
                "port": 5432,
                "name": "testdb",
                "user": "testuser",
                "password": "testpass",
            },
            "watch": {
                "directories": ["/tmp/test"],
                "extensions": [".md", ".txt"],
            },
        }
        config_file = tmp_path / "indexer.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        config = load_indexer_config(config_file)
        assert isinstance(config, IndexerConfig)
        assert config.database.host == "testhost"
        assert config.watch.extensions == [".md", ".txt"]

    def test_load_server_config(self, tmp_path: Path):
        """测试加载 Server 配置"""
        config_content = {
            "server": {
                "host": "0.0.0.0",
                "port": 9000,
            },
            "database": {
                "host": "testhost",
                "port": 5432,
                "name": "testdb",
                "user": "testuser",
                "password": "testpass",
            },
            "llm": {
                "api_base": "http://localhost:11434/v1",
                "model": "test-model",
            },
        }
        config_file = tmp_path / "server.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        config = load_server_config(config_file)
        assert isinstance(config, ServerAppConfig)
        assert config.server.port == 9000
        assert config.llm.model == "test-model"

    def test_load_config_file_not_found(self):
        """测试配置文件不存在"""
        with pytest.raises(FileNotFoundError):
            load_indexer_config("/nonexistent/config.yaml")

    def test_env_var_expansion(self, tmp_path: Path):
        """测试环境变量展开"""
        os.environ["TEST_DB_PASSWORD"] = "secret123"
        try:
            config_content = {
                "database": {
                    "host": "localhost",
                    "port": 5432,
                    "name": "testdb",
                    "user": "testuser",
                    "password": "${TEST_DB_PASSWORD}",
                },
            }
            config_file = tmp_path / "indexer.yaml"
            with open(config_file, "w") as f:
                yaml.dump(config_content, f)

            config = load_indexer_config(config_file)
            assert config.database.password == "secret123"
        finally:
            del os.environ["TEST_DB_PASSWORD"]
