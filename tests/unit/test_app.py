"""FastAPI 应用测试"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from dropqa.server.app import create_app
from dropqa.common.config import ServerAppConfig


class TestHealthEndpoint:
    """健康检查接口测试"""

    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        config = ServerAppConfig()
        app = create_app(config)
        return TestClient(app)

    def test_health_check(self, client):
        """测试健康检查接口"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_health_check_method_not_allowed(self, client):
        """测试健康检查接口不允许 POST"""
        response = client.post("/health")
        assert response.status_code == 405


class TestAppCreation:
    """应用创建测试"""

    def test_create_app_with_default_config(self):
        """测试使用默认配置创建应用"""
        config = ServerAppConfig()
        app = create_app(config)
        assert app is not None
        assert app.title == "DropQA"

    def test_app_has_docs(self):
        """测试应用有文档接口"""
        config = ServerAppConfig()
        app = create_app(config)
        client = TestClient(app)

        response = client.get("/docs")
        assert response.status_code == 200


class TestQAEndpoint:
    """QA 接口测试"""

    @pytest.fixture
    def mock_qa_service(self):
        """创建模拟 QA 服务"""
        from dropqa.server.qa import QAResponse, SourceReference
        service = MagicMock()
        service.ask = AsyncMock(return_value=QAResponse(
            answer="这是回答",
            sources=[
                SourceReference(
                    document_name="doc.md",
                    path="Chapter 1",
                    content_snippet="snippet",
                )
            ],
        ))
        return service

    @pytest.fixture
    def client_with_mock(self, mock_qa_service):
        """创建带模拟服务的测试客户端"""
        config = ServerAppConfig()
        app = create_app(config)
        # 直接设置 mock 服务
        app.state.qa_service = mock_qa_service
        return TestClient(app, raise_server_exceptions=False)

    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        config = ServerAppConfig()
        app = create_app(config)
        return TestClient(app, raise_server_exceptions=False)

    def test_qa_ask_missing_question(self, client):
        """测试缺少 question 参数"""
        response = client.post("/api/qa/ask", json={})
        assert response.status_code == 422  # Validation error

    def test_qa_ask_valid_question(self, client_with_mock):
        """测试有效问题"""
        response = client_with_mock.post(
            "/api/qa/ask",
            json={"question": "什么是项目背景？"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert data["answer"] == "这是回答"
        assert len(data["sources"]) == 1

    def test_qa_ask_empty_question(self, mock_qa_service):
        """测试空问题"""
        from dropqa.server.qa import QAResponse
        mock_qa_service.ask = AsyncMock(return_value=QAResponse(
            answer="请输入您的问题。",
            sources=[],
        ))

        config = ServerAppConfig()
        app = create_app(config)
        app.state.qa_service = mock_qa_service
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post("/api/qa/ask", json={"question": ""})
        assert response.status_code == 200
        assert response.json()["answer"] == "请输入您的问题。"
