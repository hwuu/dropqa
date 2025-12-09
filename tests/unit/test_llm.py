"""LLM 服务测试"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dropqa.common.config import LLMConfig
from dropqa.server.llm import LLMService


class TestLLMService:
    """LLMService 测试"""

    @pytest.fixture
    def llm_config(self):
        """创建 LLM 配置"""
        return LLMConfig(
            api_base="http://localhost:11434/v1",
            api_key="test-key",
            model="test-model",
            temperature=0.2,
            max_tokens=1000,
            system_prompt="You are a helpful assistant.",
        )

    @pytest.fixture
    def llm_service(self, llm_config):
        """创建 LLMService 实例"""
        return LLMService(llm_config)

    def test_llm_service_init(self, llm_service, llm_config):
        """测试 LLMService 初始化"""
        assert llm_service.config == llm_config
        assert llm_service.client is not None

    def test_build_messages_with_system_prompt(self, llm_service):
        """测试构建消息（包含系统提示）"""
        user_messages = [{"role": "user", "content": "Hello"}]
        messages = llm_service._build_messages(user_messages)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_build_messages_without_system_prompt(self, llm_config):
        """测试构建消息（无系统提示）"""
        llm_config.system_prompt = ""
        service = LLMService(llm_config)

        user_messages = [{"role": "user", "content": "Hello"}]
        messages = service._build_messages(user_messages)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_chat_returns_response(self, llm_service):
        """测试 chat 方法返回响应"""
        # 模拟 OpenAI 响应
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello! How can I help?"))]

        with patch.object(llm_service.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            result = await llm_service.chat([{"role": "user", "content": "Hi"}])

            assert result == "Hello! How can I help?"
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_with_custom_temperature(self, llm_service):
        """测试 chat 方法自定义温度"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]

        with patch.object(llm_service.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            await llm_service.chat(
                [{"role": "user", "content": "Hi"}],
                temperature=0.8,
            )

            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["temperature"] == 0.8

    @pytest.mark.asyncio
    async def test_chat_empty_response(self, llm_service):
        """测试空响应处理"""
        mock_response = MagicMock()
        mock_response.choices = []

        with patch.object(llm_service.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            result = await llm_service.chat([{"role": "user", "content": "Hi"}])

            assert result == ""


class TestLLMConfig:
    """LLMConfig 测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = LLMConfig()
        assert config.api_base == "http://localhost:11434/v1"
        assert config.temperature == 0.2

    def test_custom_config(self):
        """测试自定义配置"""
        config = LLMConfig(
            api_base="https://api.openai.com/v1",
            model="gpt-4",
            temperature=0.5,
        )
        assert config.api_base == "https://api.openai.com/v1"
        assert config.model == "gpt-4"
        assert config.temperature == 0.5
