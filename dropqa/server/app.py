"""FastAPI 应用"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dropqa.common.config import ServerAppConfig
from dropqa.common.db import Database, init_db
from dropqa.server.llm import LLMService
from dropqa.server.qa import QAService
from dropqa.server.search import SearchService

# 静态文件目录
STATIC_DIR = Path(__file__).parent / "static"


# 请求/响应模型
class AskRequest(BaseModel):
    """问答请求"""
    question: str


class SourceResponse(BaseModel):
    """来源响应"""
    document_name: str
    path: str
    content_snippet: str


class AskResponse(BaseModel):
    """问答响应"""
    answer: str
    sources: list[SourceResponse]


def create_app(config: ServerAppConfig) -> FastAPI:
    """创建 FastAPI 应用

    Args:
        config: 服务配置

    Returns:
        FastAPI 应用实例
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """应用生命周期管理"""
        # 启动时初始化数据库
        db = init_db(config.database)
        app.state.db = db
        app.state.config = config

        # 初始化服务
        search_service = SearchService(db)
        llm_service = LLMService(config.llm)
        qa_service = QAService(
            search_service,
            llm_service,
            top_k=config.retrieval.top_k,
        )
        app.state.qa_service = qa_service

        yield

        # 关闭时清理资源
        await db.close()

    app = FastAPI(
        title="DropQA",
        description="文档问答系统 API",
        version="0.1.0",
        lifespan=lifespan,
    )

    # 注册路由
    _register_routes(app)

    # 注册错误处理
    _register_error_handlers(app)

    return app


def _register_routes(app: FastAPI) -> None:
    """注册路由"""

    @app.get("/health")
    async def health_check():
        """健康检查"""
        return {"status": "ok"}

    @app.get("/")
    async def index():
        """首页"""
        return FileResponse(STATIC_DIR / "index.html")

    @app.post("/api/qa/ask", response_model=AskResponse)
    async def qa_ask(request: Request, body: AskRequest):
        """问答接口

        基于文档内容回答用户问题。
        """
        qa_service: QAService = request.app.state.qa_service
        response = await qa_service.ask(body.question)

        return AskResponse(
            answer=response.answer,
            sources=[
                SourceResponse(
                    document_name=s.document_name,
                    path=s.path,
                    content_snippet=s.content_snippet,
                )
                for s in response.sources
            ],
        )


def _register_error_handlers(app: FastAPI) -> None:
    """注册错误处理器"""

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """通用异常处理"""
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": str(exc),
            },
        )
