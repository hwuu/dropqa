"""Agentic 配置测试"""

import pytest

from dropqa.server.agentic.config import (
    AgenticConfig,
    HybridSearchConfig,
    MultiRoundConfig,
    QueryRewriteConfig,
    RerankConfig,
)


class TestQueryRewriteConfig:
    """查询改写配置测试"""

    def test_default_values(self):
        """测试默认值"""
        config = QueryRewriteConfig()
        assert config.enabled is True


class TestHybridSearchConfig:
    """混合搜索配置测试"""

    def test_default_values(self):
        """测试默认值"""
        config = HybridSearchConfig()
        assert config.enabled is True
        assert config.fulltext_weight == 0.5

    def test_custom_weight(self):
        """测试自定义权重"""
        config = HybridSearchConfig(fulltext_weight=0.7)
        assert config.fulltext_weight == 0.7

    def test_weight_validation_min(self):
        """测试权重最小值验证"""
        config = HybridSearchConfig(fulltext_weight=0.0)
        assert config.fulltext_weight == 0.0

    def test_weight_validation_max(self):
        """测试权重最大值验证"""
        config = HybridSearchConfig(fulltext_weight=1.0)
        assert config.fulltext_weight == 1.0

    def test_weight_validation_invalid(self):
        """测试无效权重"""
        with pytest.raises(ValueError):
            HybridSearchConfig(fulltext_weight=1.5)

        with pytest.raises(ValueError):
            HybridSearchConfig(fulltext_weight=-0.1)


class TestMultiRoundConfig:
    """多轮检索配置测试"""

    def test_default_values(self):
        """测试默认值"""
        config = MultiRoundConfig()
        assert config.enabled is True
        assert config.max_rounds == 2

    def test_custom_max_rounds(self):
        """测试自定义轮数"""
        config = MultiRoundConfig(max_rounds=3)
        assert config.max_rounds == 3

    def test_max_rounds_validation(self):
        """测试轮数验证"""
        with pytest.raises(ValueError):
            MultiRoundConfig(max_rounds=0)

        with pytest.raises(ValueError):
            MultiRoundConfig(max_rounds=6)


class TestRerankConfig:
    """结果重排序配置测试"""

    def test_default_values(self):
        """测试默认值"""
        config = RerankConfig()
        assert config.enabled is True
        assert config.top_k == 5

    def test_custom_top_k(self):
        """测试自定义 top_k"""
        config = RerankConfig(top_k=10)
        assert config.top_k == 10

    def test_top_k_validation(self):
        """测试 top_k 验证"""
        with pytest.raises(ValueError):
            RerankConfig(top_k=0)


class TestAgenticConfig:
    """Agentic RAG 总配置测试"""

    def test_default_values(self):
        """测试默认值"""
        config = AgenticConfig()
        assert config.enabled is False  # 默认关闭，保持向后兼容
        assert config.top_k == 10

    def test_default_sub_configs(self):
        """测试子配置默认值"""
        config = AgenticConfig()
        assert config.query_rewrite.enabled is True
        assert config.hybrid_search.enabled is True
        assert config.multi_round.enabled is True
        assert config.rerank.enabled is True

    def test_enabled_with_all_features(self):
        """测试启用所有功能"""
        config = AgenticConfig(
            enabled=True,
            top_k=20,
            query_rewrite=QueryRewriteConfig(enabled=True),
            hybrid_search=HybridSearchConfig(enabled=True, fulltext_weight=0.6),
            multi_round=MultiRoundConfig(enabled=True, max_rounds=3),
            rerank=RerankConfig(enabled=True, top_k=10),
        )

        assert config.enabled is True
        assert config.top_k == 20
        assert config.query_rewrite.enabled is True
        assert config.hybrid_search.fulltext_weight == 0.6
        assert config.multi_round.max_rounds == 3
        assert config.rerank.top_k == 10

    def test_disabled_features(self):
        """测试禁用部分功能"""
        config = AgenticConfig(
            enabled=True,
            query_rewrite=QueryRewriteConfig(enabled=False),
            hybrid_search=HybridSearchConfig(enabled=False),
        )

        assert config.enabled is True
        assert config.query_rewrite.enabled is False
        assert config.hybrid_search.enabled is False
        assert config.multi_round.enabled is True  # 其他保持默认
        assert config.rerank.enabled is True

    def test_from_dict(self):
        """测试从字典创建配置"""
        config_dict = {
            "enabled": True,
            "top_k": 15,
            "query_rewrite": {"enabled": True},
            "hybrid_search": {"enabled": True, "fulltext_weight": 0.4},
            "multi_round": {"enabled": False, "max_rounds": 1},
            "rerank": {"enabled": True, "top_k": 3},
        }

        config = AgenticConfig.model_validate(config_dict)

        assert config.enabled is True
        assert config.top_k == 15
        assert config.hybrid_search.fulltext_weight == 0.4
        assert config.multi_round.enabled is False
        assert config.rerank.top_k == 3
