"""Server 服务入口

使用方法:
    python -m dropqa.server --config config/server.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import uvicorn

from dropqa.common.config import load_server_config
from dropqa.server.app import create_app

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,  # 改为 DEBUG 级别以显示 RAG 流程日志
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# 降低第三方库的日志级别
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.INFO)
logging.getLogger("uvicorn.access").setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def main(config_path: str) -> None:
    """主入口函数

    Args:
        config_path: 配置文件路径
    """
    # 1. 加载配置
    logger.info(f"加载配置: {config_path}")
    config = load_server_config(config_path)

    # 2. 创建应用
    app = create_app(config)

    # 3. 启动服务
    logger.info(f"启动服务: {config.server.host}:{config.server.port}")
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level="info",
    )


def cli() -> None:
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="DropQA QA 服务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/server.yaml",
        help="配置文件路径 (默认: config/server.yaml)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        logger.info("请复制 config/server.example.yaml 为 config/server.yaml 并修改配置")
        sys.exit(1)

    main(str(config_path))


if __name__ == "__main__":
    cli()
