"""
优雅降级管理器测试 - 符合SDD Constitution的Test-First原则
测试GracefulDegradationManager的各种降级策略和资源监控功能
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import threading
from contextlib import asynccontextmanager

from src.middleware.graceful_degradation import (
    GracefulDegradationManager,
    DegradationLevel,
    ResourceThreshold,
    DegradationStrategy,
    graceful_degradation_context,
    get_system_status,
    is_feature_available,
)


class TestGracefulDegradationManager:
    """优雅降级管理器测试类"""

    def setup_method(self):
        """测试前的设置"""
        # 创建一个不启动监控线程的实例用于测试
        with patch.object(GracefulDegradationManager, '_start_monitoring'):
            self.degradation_manager = GracefulDegradationManager()

    def test_initialization(self):
        """测试初始化"""
        assert self.degradation_manager.current_level == DegradationLevel.NONE
        assert self.degradation_manager.check_interval == 30
        assert len(self.degradation_manager.feature_flags) > 0
        assert all(flag is True for flag in self.degradation_manager.feature_flags.values())

    def test_degradation_levels(self):
        """测试降级级别枚举"""
        levels = [
            DegradationLevel.NONE,
            DegradationLevel.MINIMAL,
            DegradationLevel.MODERATE,
            DegradationLevel.SEVERE,
            DegradationLevel.CRITICAL
        ]

        assert len(levels) == 5
        assert DegradationLevel.NONE.value == "none"
        assert DegradationLevel.CRITICAL.value == "critical"

    def test_resource_threshold(self):
        """测试资源阈值配置"""
        threshold = ResourceThreshold()

        assert threshold.memory_percent == 85.0
        assert threshold.cpu_percent == 80.0
        assert threshold.disk_percent == 90.0
        assert threshold.response_time_ms == 5000.0

    def test_determine_degradation_level_none(self):
        """测试确定降级级别 - 无降级"""
        metrics = {
            "memory_percent": 50.0,
            "cpu_percent": 60.0,
            "disk_percent": 70.0
        }

        level = self.degradation_manager._determine_degradation_level(metrics)

        assert level == DegradationLevel.NONE

    def test_determine_degradation_level_critical(self):
        """测试确定降级级别 - 关键降级"""
        metrics = {
            "memory_percent": 96.0,
            "cpu_percent": 70.0,
            "disk_percent": 80.0
        }

        level = self.degradation_manager._determine_degradation_level(metrics)

        assert level == DegradationLevel.CRITICAL

    def test_determine_degradation_level_severe(self):
        """测试确定降级级别 - 严重降级"""
        metrics = {
            "memory_percent": 92.0,
            "cpu_percent": 85.0,
            "disk_percent": 85.0
        }

        level = self.degradation_manager._determine_degradation_level(metrics)

        assert level == DegradationLevel.SEVERE

    def test_strategies_initialization(self):
        """测试降级策略初始化"""
        strategies = self.degradation_manager.strategies

        assert DegradationLevel.MINIMAL in strategies
        assert DegradationLevel.MODERATE in strategies
        assert DegradationLevel.SEVERE in strategies
        assert DegradationLevel.CRITICAL in strategies

        # 验证策略结构
        for strategy in strategies.values():
            assert hasattr(strategy, 'level')
            assert hasattr(strategy, 'description')
            assert hasattr(strategy, 'affected_features')
            assert hasattr(strategy, 'mitigation_actions')
            assert hasattr(strategy, 'resource_thresholds')

    def test_execute_degradation_strategy_upgrade(self):
        """测试执行降级策略 - 降级升级"""
        old_level = self.degradation_manager.current_level
        new_level = DegradationLevel.MINIMAL

        with patch.object(self.degradation_manager, 'logger') as mock_logger:
            self.degradation_manager._execute_degradation_strategy(new_level)

            assert self.degradation_manager.current_level == new_level
            assert self.degradation_manager.degradation_stats["degradation_events"] == 1
            mock_logger.log_warning.assert_called_once()

    def test_execute_degradation_strategy_recovery(self):
        """测试执行降级策略 - 系统恢复"""
        # 先设置降级状态
        self.degradation_manager.current_level = DegradationLevel.MINIMAL
        old_level = self.degradation_manager.current_level

        with patch.object(self.degradation_manager, 'logger') as mock_logger:
            self.degradation_manager._execute_degradation_strategy(DegradationLevel.NONE)

            assert self.degradation_manager.current_level == DegradationLevel.NONE
            assert self.degradation_manager.degradation_stats["recovery_events"] == 1
            mock_logger.log_info.assert_called_once()

    def test_disable_performance_monitoring(self):
        """测试禁用性能监控"""
        self.degradation_manager._disable_performance_monitoring()

        assert self.degradation_manager.feature_flags["performance_monitoring"] is False

    def test_disable_caching(self):
        """测试禁用缓存"""
        self.degradation_manager._disable_caching()

        assert self.degradation_manager.feature_flags["data_cache"] is False

    def test_disable_real_time_updates(self):
        """测试禁用实时更新"""
        self.degradation_manager._disable_real_time_updates()

        assert self.degradation_manager.feature_flags["real_time_updates"] is False

    def test_restore_all_features(self):
        """测试恢复所有功能"""
        # 先禁用一些功能
        self.degradation_manager.feature_flags["data_cache"] = False
        self.degradation_manager.feature_flags["real_time_updates"] = False

        self.degradation_manager._restore_all_features()

        assert all(flag is True for flag in self.degradation_manager.feature_flags.values())

    def test_is_feature_available(self):
        """测试功能可用性检查"""
        # 功能可用
        assert self.degradation_manager.is_feature_available("real_time_updates") is True

        # 禁用功能
        self.degradation_manager.feature_flags["real_time_updates"] = False
        assert self.degradation_manager.is_feature_available("real_time_updates") is False

        # 不存在的功能
        assert self.degradation_manager.is_feature_available("nonexistent_feature") is False

    def test_get_system_status(self):
        """测试获取系统状态"""
        status = self.degradation_manager.get_system_status()

        assert "degradation_level" in status
        assert "system_resources" in status
        assert "feature_flags" in status
        assert "statistics" in status
        assert "recommendations" in status

        assert status["degradation_level"] == DegradationLevel.NONE.value

    def test_get_recommendations(self):
        """测试获取系统建议"""
        # 正常状态
        recommendations = self.degradation_manager._get_recommendations()
        assert len(recommendations) == 0

        # 设置降级状态
        self.degradation_manager.current_level = DegradationLevel.SEVERE

        recommendations = self.degradation_manager._get_recommendations()
        assert len(recommendations) > 0
        assert any("SEVERE" in rec for rec in recommendations)

    def test_get_degradation_statistics(self):
        """测试获取降级统计"""
        stats = self.degradation_manager.get_degradation_statistics()

        assert "degradation_events" in stats
        assert "recovery_events" in stats
        assert "total_checks" in stats
        assert "mitigation_actions_triggered" in stats

    @patch('backend.src.middleware.graceful_degradation.psutil')
    def test_check_system_resources(self, mock_psutil):
        """测试系统资源检查"""
        # 模拟psutil返回值
        mock_memory = Mock()
        mock_memory.percent = 80.0
        mock_psutil.virtual_memory.return_value = mock_memory

        mock_cpu = 75.0
        mock_psutil.cpu_percent.return_value = mock_cpu

        mock_disk = Mock()
        mock_disk.percent = 70.0
        mock_psutil.disk_usage.return_value = mock_disk

        with patch.object(self.degradation_manager, '_determine_degradation_level') as mock_determine:
            mock_determine.return_value = DegradationLevel.NONE

            self.degradation_manager._check_system_resources()

            # 验证资源历史记录
            assert len(self.degradation_manager.resource_history) == 1
            assert self.degradation_manager.resource_history[0]["memory_percent"] == 80.0
            assert self.degradation_manager.resource_history[0]["cpu_percent"] == 75.0

    @pytest.mark.asyncio
    async def test_process_with_degradation_success(self):
        """测试在降级模式下处理操作 - 成功"""
        operation_name = "test_operation"

        async with self.degradation_manager.process_with_degradation(operation_name) as context:
            assert context["operation"] == operation_name
            assert context["degradation_level"] == DegradationLevel.NONE.value

        # 验证统计信息
        assert self.degradation_manager.degradation_stats["total_checks"] >= 1

    @pytest.mark.asyncio
    async def test_process_with_degradation_error(self):
        """测试在降级模式下处理操作 - 错误"""
        operation_name = "test_operation"

        with pytest.raises(Exception):
            async with self.degradation_manager.process_with_degradation(operation_name):
                raise ValueError("Test error")

    @pytest.mark.asyncio
    async def test_process_with_degradation_critical_error(self):
        """测试在关键降级模式下的错误处理"""
        operation_name = "test_operation"

        # 设置关键降级状态
        self.degradation_manager.current_level = DegradationLevel.CRITICAL

        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            async with self.degradation_manager.process_with_degradation(operation_name):
                raise ValueError("Test error")

        assert exc_info.value.status_code == 503
        assert "Service temporarily unavailable" in str(exc_info.value.detail)

    def test_resource_history_limit(self):
        """测试资源历史记录限制"""
        # 填充超过限制的历史记录
        max_size = self.degradation_manager.max_history_size

        for i in range(max_size + 10):
            self.degradation_manager.resource_history.append({
                "memory_percent": 50.0 + i,
                "timestamp": time.time()
            })

        # 验证历史记录被限制
        assert len(self.degradation_manager.resource_history) == max_size

    @patch('backend.src.middleware.graceful_degradation.threading.Thread')
    def test_start_monitoring(self, mock_thread_class):
        """测试启动监控线程"""
        mock_thread = Mock()
        mock_thread_class.return_value = mock_thread

        # 创建新实例（会启动监控线程）
        with patch('backend.src.middleware.graceful_degradation.time.sleep'):  # 避免实际睡眠
            manager = GracefulDegradationManager()

            # 验证线程被创建和启动
            mock_thread_class.assert_called_once()
            mock_thread.start.assert_called_once()
            assert mock_thread.daemon is True


class TestConvenienceFunctions:
    """便捷函数测试类"""

    @pytest.mark.asyncio
    async def test_graceful_degradation_context(self):
        """测试优雅降级上下文管理器"""
        with patch.object(GracefulDegradationManager, 'process_with_degradation') as mock_process:
            # 正确设置异步上下文管理器的返回值
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = {"test": "context"}
            mock_context.__aexit__.return_value = None
            mock_process.return_value = mock_context

            async with graceful_degradation_context("test_operation") as context:
                assert context == {"test": "context"}

            mock_process.assert_called_once_with("test_operation")

    def test_get_system_status(self):
        """测试获取系统状态便捷函数"""
        with patch.object(GracefulDegradationManager, 'get_system_status') as mock_get_status:
            mock_get_status.return_value = {"status": "test"}

            result = get_system_status()

            assert result == {"status": "test"}
            mock_get_status.assert_called_once()

    def test_is_feature_available(self):
        """测试功能可用性检查便捷函数"""
        with patch.object(GracefulDegradationManager, 'is_feature_available') as mock_available:
            mock_available.return_value = True

            result = is_feature_available("test_feature")

            assert result is True
            mock_available.assert_called_once_with("test_feature")


class TestDegradationStrategies:
    """降级策略测试类"""

    def setup_method(self):
        """测试前的设置"""
        with patch.object(GracefulDegradationManager, '_start_monitoring'):
            self.manager = GracefulDegradationManager()

    def test_minimal_degradation_strategy(self):
        """测试最小降级策略"""
        strategy = self.manager.strategies[DegradationLevel.MINIMAL]

        assert strategy.level == DegradationLevel.MINIMAL
        assert "performance_monitoring" in strategy.affected_features
        assert len(strategy.mitigation_actions) == 1

    def test_moderate_degradation_strategy(self):
        """测试中等降级策略"""
        strategy = self.manager.strategies[DegradationLevel.MODERATE]

        assert strategy.level == DegradationLevel.MODERATE
        assert "data_cache" in strategy.affected_features
        assert "real_time_updates" in strategy.affected_features
        assert len(strategy.mitigation_actions) == 2

    def test_severe_degradation_strategy(self):
        """测试严重降级策略"""
        strategy = self.manager.strategies[DegradationLevel.SEVERE]

        assert strategy.level == DegradationLevel.SEVERE
        assert "batch_processing" in strategy.affected_features
        assert "export_high_quality" in strategy.affected_features
        assert len(strategy.mitigation_actions) == 2

    def test_critical_degradation_strategy(self):
        """测试关键降级策略"""
        strategy = self.manager.strategies[DegradationLevel.CRITICAL]

        assert strategy.level == DegradationLevel.CRITICAL
        assert "advanced_visualization" in strategy.affected_features
        assert len(strategy.mitigation_actions) == 1

    def test_strategy_resource_thresholds(self):
        """测试策略资源阈值"""
        for level, strategy in self.manager.strategies.items():
            assert hasattr(strategy.resource_thresholds, 'memory_percent')
            assert hasattr(strategy.resource_thresholds, 'cpu_percent')
            assert hasattr(strategy.resource_thresholds, 'disk_percent')
            assert hasattr(strategy.resource_thresholds, 'response_time_ms')

            # 验证阈值随降级级别递增
            if level == DegradationLevel.MINIMAL:
                assert strategy.resource_thresholds.memory_percent < 90
            elif level == DegradationLevel.CRITICAL:
                assert strategy.resource_thresholds.memory_percent >= 95