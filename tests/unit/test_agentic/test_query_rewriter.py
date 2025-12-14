"""查询改写测试"""

import json
from unittest.mock import AsyncMock

import pytest

from dropqa.server.agentic.query_rewriter import QueryRewriter, RewrittenQuery


class TestRewrittenQuery:
    """RewrittenQuery 数据类测试"""

    def test_creation(self):
        """测试创建 RewrittenQuery"""
        query = RewrittenQuery(
            original="什么是 DropQA？",
            keywords=["DropQA"],
            fulltext_query="DropQA 是什么",
            semantic_query="请解释 DropQA 是什么系统",
        )

        assert query.original == "什么是 DropQA？"
        assert query.keywords == ["DropQA"]
        assert query.fulltext_query == "DropQA 是什么"
        assert query.semantic_query == "请解释 DropQA 是什么系统"


class TestQueryRewriter:
    """QueryRewriter 测试"""

    @pytest.mark.asyncio
    async def test_rewrite_empty_question(self):
        """测试空问题"""
        mock_llm = AsyncMock()
        rewriter = QueryRewriter(mock_llm)

        result = await rewriter.rewrite("")

        assert result.original == ""
        assert result.keywords == []
        assert result.fulltext_query == ""
        assert result.semantic_query == ""

    @pytest.mark.asyncio
    async def test_rewrite_whitespace_only(self):
        """测试仅有空白的问题"""
        mock_llm = AsyncMock()
        rewriter = QueryRewriter(mock_llm)

        result = await rewriter.rewrite("   ")

        assert result.original == ""
        assert result.fulltext_query == ""

    @pytest.mark.asyncio
    async def test_rewrite_success(self):
        """测试成功改写"""
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value=json.dumps({
            "keywords": ["DropQA", "文档问答"],
            "fulltext_query": "DropQA 文档问答系统",
            "semantic_query": "DropQA 是什么样的文档问答系统",
        }))
        rewriter = QueryRewriter(mock_llm)

        result = await rewriter.rewrite("什么是 DropQA？")

        assert result.original == "什么是 DropQA？"
        assert "DropQA" in result.keywords
        assert "DropQA" in result.fulltext_query
        mock_llm.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_rewrite_with_markdown_json(self):
        """测试包含 markdown 代码块的响应"""
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value="""```json
{
    "keywords": ["关键词1"],
    "fulltext_query": "全文查询",
    "semantic_query": "语义查询"
}
```""")
        rewriter = QueryRewriter(mock_llm)

        result = await rewriter.rewrite("测试问题")

        assert result.keywords == ["关键词1"]
        assert result.fulltext_query == "全文查询"

    @pytest.mark.asyncio
    async def test_rewrite_llm_error_fallback(self):
        """测试 LLM 错误时的回退"""
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(side_effect=Exception("LLM 服务不可用"))
        rewriter = QueryRewriter(mock_llm)

        result = await rewriter.rewrite("DropQA 如何工作？")

        # 应该回退到原始问题
        assert result.original == "DropQA 如何工作？"
        assert result.fulltext_query == "DropQA 如何工作？"

    @pytest.mark.asyncio
    async def test_rewrite_invalid_json_fallback(self):
        """测试无效 JSON 响应时的回退"""
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value="这不是有效的 JSON")
        rewriter = QueryRewriter(mock_llm)

        result = await rewriter.rewrite("测试问题")

        assert result.original == "测试问题"
        assert result.fulltext_query == "测试问题"


class TestParseResponse:
    """_parse_response 方法测试"""

    @pytest.fixture
    def rewriter(self):
        """创建 QueryRewriter 实例"""
        return QueryRewriter(AsyncMock())

    def test_parse_direct_json(self, rewriter):
        """测试直接解析 JSON"""
        response = '{"keywords": ["a", "b"], "fulltext_query": "q1", "semantic_query": "q2"}'
        result = rewriter._parse_response(response)

        assert result is not None
        assert result["keywords"] == ["a", "b"]

    def test_parse_json_in_markdown(self, rewriter):
        """测试从 markdown 代码块提取 JSON"""
        response = """这是一些文本
```json
{"keywords": ["test"], "fulltext_query": "t", "semantic_query": "s"}
```
更多文本"""
        result = rewriter._parse_response(response)

        assert result is not None
        assert result["keywords"] == ["test"]

    def test_parse_json_with_braces(self, rewriter):
        """测试提取 {} 之间的 JSON"""
        response = '好的，结果如下：{"keywords": ["x"], "fulltext_query": "y", "semantic_query": "z"}'
        result = rewriter._parse_response(response)

        assert result is not None
        assert result["keywords"] == ["x"]

    def test_parse_invalid_response(self, rewriter):
        """测试无效响应"""
        result = rewriter._parse_response("这不是 JSON")
        assert result is None

    def test_parse_empty_response(self, rewriter):
        """测试空响应"""
        result = rewriter._parse_response("")
        assert result is None


class TestExtractKeywordsSimple:
    """_extract_keywords_simple 方法测试"""

    @pytest.fixture
    def rewriter(self):
        """创建 QueryRewriter 实例"""
        return QueryRewriter(AsyncMock())

    def test_extract_chinese_keywords(self, rewriter):
        """测试中文关键词提取"""
        keywords = rewriter._extract_keywords_simple("如何使用 DropQA 进行文档问答？")

        # 应该过滤掉 "如何"、"使用"、"进行" 等停用词
        assert "DropQA" in keywords or "文档问答" in keywords

    def test_extract_filters_stop_words(self, rewriter):
        """测试过滤停用词"""
        keywords = rewriter._extract_keywords_simple("什么是 Python 语言？")

        assert "Python" in keywords
        assert "什么" not in keywords
        assert "是" not in keywords

    def test_extract_max_keywords(self, rewriter):
        """测试最多返回 5 个关键词"""
        long_question = "a b c d e f g h i j k l m n o p"
        keywords = rewriter._extract_keywords_simple(long_question)

        assert len(keywords) <= 5

    def test_extract_empty_input(self, rewriter):
        """测试空输入"""
        keywords = rewriter._extract_keywords_simple("")
        assert keywords == []
