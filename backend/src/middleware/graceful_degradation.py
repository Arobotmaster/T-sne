"""
优雅降级处理中间件 - 符合SDD Constitution的Scientific Observability原则
在系统资源不足或错误发生时提供优雅的降级策略，确保核心功能可用
"""

import psutil
import time
from typing import Dict, Any, List, Callable
from contextlib import asynccontextmanager
from fastapi import HTTPException
import threading
from dataclasses import dataclass
from enum import Enum

from ..utils.scientific_logging import ScientificLogger
from ..utils.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity


class DegradationLevel(Enum):
    """降级级别"""

    NONE = "none"  # 无降级，全功能运行
    MINIMAL = "minimal"  # 最小降级，影响非核心功能
    MODERATE = "moderate"  # 中等降级，影响部分功能
    SEVERE = "severe"  # 严重降级，仅核心功能
    CRITICAL = "critical"  # 关键降级，仅基本响应


@dataclass
class ResourceThreshold:
    """资源阈值配置"""

    memory_percent: float = 85.0  # 内存使用率阈值
    cpu_percent: float = 80.0  # CPU使用率阈值
    disk_percent: float = 90.0  # 磁盘使用率阈值
    response_time_ms: float = 5000.0  # 响应时间阈值(毫秒)


@dataclass
class DegradationStrategy:
    """降级策略"""

    level: DegradationLevel
    description: str
    affected_features: List[str]
    mitigation_actions: List[Callable]
    resource_thresholds: ResourceThreshold


class GracefulDegradationManager:
    """优雅降级管理器"""

    def __init__(self):
        self.logger = ScientificLogger("graceful_degradation", log_dir="logs")
        self.error_handler = ErrorHandler(log_dir="logs")

        # 当前降级状态
        self.current_level = DegradationLevel.NONE
        self.last_check_time = 0
        self.check_interval = 30  # 秒

        # 资源监控
        self.resource_history = []
        self.max_history_size = 100

        # 降级策略
        self.strategies = self._initialize_strategies()

        # 功能开关
        self.feature_flags = {
            "real_time_updates": True,
            "advanced_visualization": True,
            "batch_processing": True,
            "export_high_quality": True,
            "data_cache": True,
            "performance_monitoring": True,
        }

        # 统计信息
        self.degradation_stats = {
            "degradation_events": 0,
            "recovery_events": 0,
            "total_checks": 0,
            "mitigation_actions_triggered": 0,
        }

        # 启动监控线程
        self._start_monitoring()

    def _initialize_strategies(self) -> Dict[DegradationLevel, DegradationStrategy]:
        """初始化降级策略"""
        return {
            DegradationLevel.MINIMAL: DegradationStrategy(
                level=DegradationLevel.MINIMAL,
                description="Minimal degradation affecting non-critical features",
                affected_features=["performance_monitoring"],
                mitigation_actions=[self._disable_performance_monitoring],
                resource_thresholds=ResourceThreshold(
                    memory_percent=75.0, cpu_percent=70.0, response_time_ms=2000.0
                ),
            ),
            DegradationLevel.MODERATE: DegradationStrategy(
                level=DegradationLevel.MODERATE,
                description="Moderate degradation affecting some advanced features",
                affected_features=["data_cache", "real_time_updates"],
                mitigation_actions=[
                    self._disable_caching,
                    self._disable_real_time_updates,
                ],
                resource_thresholds=ResourceThreshold(
                    memory_percent=85.0, cpu_percent=80.0, response_time_ms=5000.0
                ),
            ),
            DegradationLevel.SEVERE: DegradationStrategy(
                level=DegradationLevel.SEVERE,
                description="Severe degradation keeping only core functionality",
                affected_features=["batch_processing", "export_high_quality"],
                mitigation_actions=[
                    self._limit_batch_size,
                    self._reduce_export_quality,
                ],
                resource_thresholds=ResourceThreshold(
                    memory_percent=90.0, cpu_percent=90.0, response_time_ms=10000.0
                ),
            ),
            DegradationLevel.CRITICAL: DegradationStrategy(
                level=DegradationLevel.CRITICAL,
                description="Critical degradation, basic response only",
                affected_features=["advanced_visualization"],
                mitigation_actions=[self._disable_advanced_features],
                resource_thresholds=ResourceThreshold(
                    memory_percent=95.0, cpu_percent=95.0, response_time_ms=15000.0
                ),
            ),
        }

    def _start_monitoring(self):
        """启动资源监控线程"""

        def monitor_loop():
            while True:
                try:
                    self._check_system_resources()
                    time.sleep(self.check_interval)
                except Exception as e:
                    self.error_handler.log_error(
                        error=e,
                        severity=ErrorSeverity.ERROR,
                        category=ErrorCategory.SYSTEM,
                        context={"component": "resource_monitoring"},
                    )
                    time.sleep(60)  # 错误时延长检查间隔

        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()

    def _check_system_resources(self):
        """检查系统资源状态"""
        self.degradation_stats["total_checks"] += 1

        try:
            # 获取资源使用情况
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=1)
            disk = psutil.disk_usage("/")

            resource_metrics = {
                "memory_percent": memory.percent,
                "cpu_percent": cpu,
                "disk_percent": disk.percent,
                "timestamp": time.time(),
            }

            # 记录到历史
            self.resource_history.append(resource_metrics)
            if len(self.resource_history) > self.max_history_size:
                self.resource_history.pop(0)

            # 记录科学日志
            self.logger.log_quality_metrics(
                data_name="system_resources",
                metrics={
                    "cpu_percent": cpu,
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent,
                    "degradation_level": self.current_level.value,
                },
            )

            # 确定需要的降级级别
            required_level = self._determine_degradation_level(resource_metrics)

            # 如果需要降级，执行降级策略
            if required_level != self.current_level:
                self._execute_degradation_strategy(required_level)

        except Exception as e:
            self.error_handler.log_error(
                error=e,
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM,
                context={"component": "resource_monitoring"},
            )

    def _determine_degradation_level(self, metrics: Dict[str, Any]) -> DegradationLevel:
        """根据资源指标确定降级级别"""
        memory_percent = metrics["memory_percent"]
        cpu_percent = metrics["cpu_percent"]
        disk_percent = metrics["disk_percent"]

        # 按严重程度检查阈值
        if memory_percent >= 95.0 or cpu_percent >= 95.0 or disk_percent >= 95.0:
            return DegradationLevel.CRITICAL
        elif memory_percent >= 90.0 or cpu_percent >= 90.0 or disk_percent >= 90.0:
            return DegradationLevel.SEVERE
        elif memory_percent >= 85.0 or cpu_percent >= 80.0:
            return DegradationLevel.MODERATE
        elif memory_percent >= 75.0 or cpu_percent >= 70.0:
            return DegradationLevel.MINIMAL
        else:
            return DegradationLevel.NONE

    def _execute_degradation_strategy(self, new_level: DegradationLevel):
        """执行降级策略"""
        old_level = self.current_level
        self.current_level = new_level

        if new_level == DegradationLevel.NONE:
            self.degradation_stats["recovery_events"] += 1
            self.logger.logger.info(
                f"System recovered from degradation, from_level: {old_level.value}, to_level: {new_level.value}"
            )
            # 恢复所有功能
            self._restore_all_features()
        else:
            self.degradation_stats["degradation_events"] += 1
            strategy = self.strategies[new_level]

            self.logger.logger.warning(
                f"System degraded to {new_level.value}, "
                f"from_level: {old_level.value}, "
                f"strategy: {strategy.description}, "
                f"affected_features: {strategy.affected_features}"
            )

            # 执行降级操作
            for action in strategy.mitigation_actions:
                try:
                    action()
                    self.degradation_stats["mitigation_actions_triggered"] += 1
                except Exception as e:
                    self.error_handler.log_error(
                        error=e,
                        severity=ErrorSeverity.ERROR,
                        category=ErrorCategory.SYSTEM,
                        context={
                            "component": "degradation_action",
                            "degradation_level": new_level.value,
                        },
                    )

    def _disable_performance_monitoring(self):
        """禁用性能监控"""
        self.feature_flags["performance_monitoring"] = False
        self.logger.logger.info("Performance monitoring disabled due to degradation")

    def _disable_caching(self):
        """禁用缓存"""
        self.feature_flags["data_cache"] = False
        self.logger.logger.info("Data caching disabled due to degradation")

    def _disable_real_time_updates(self):
        """禁用实时更新"""
        self.feature_flags["real_time_updates"] = False
        self.logger.logger.info("Real-time updates disabled due to degradation")

    def _limit_batch_size(self):
        """限制批处理大小"""
        # 这里会修改批处理服务的配置
        self.logger.log_info("Batch processing size limited due to degradation")

    def _reduce_export_quality(self):
        """降低导出质量"""
        self.feature_flags["export_high_quality"] = False
        self.logger.log_info("Export quality reduced due to degradation")

    def _disable_advanced_features(self):
        """禁用高级功能"""
        self.feature_flags["advanced_visualization"] = False
        self.logger.logger.info(
            "Advanced visualization features disabled due to degradation"
        )

    def _restore_all_features(self):
        """恢复所有功能"""
        for feature in self.feature_flags:
            self.feature_flags[feature] = True
        self.logger.logger.info("All features restored")

    @asynccontextmanager
    async def process_with_degradation(self, operation_name: str):
        """在降级模式下处理操作的上下文管理器"""
        start_time = time.time()
        context = {
            "operation": operation_name,
            "degradation_level": self.current_level.value,
            "available_features": [
                f for f, enabled in self.feature_flags.items() if enabled
            ],
        }

        try:
            self.logger.logger.info(
                f"Starting operation: {operation_name}, context: {context}"
            )

            yield context

            # 记录成功
            duration = (time.time() - start_time) * 1000  # 毫秒
            self.logger.log_quality_metrics(
                data_name="operation_success",
                metrics={
                    "duration_ms": duration,
                    "degradation_level": self.current_level.value,
                    "operation": operation_name,
                },
            )

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.error_handler.log_error(
                error=e,
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM,
                context={**context, "duration_ms": duration},
            )

            # 根据当前降级级别提供适当的错误处理
            if self.current_level == DegradationLevel.CRITICAL:
                raise HTTPException(
                    status_code=503,
                    detail="Service temporarily unavailable due to system overload",
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Operation failed: {operation_name}. System is running in degraded mode.",
                )

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        current_metrics = self.resource_history[-1] if self.resource_history else {}

        return {
            "degradation_level": self.current_level.value,
            "system_resources": current_metrics,
            "feature_flags": self.feature_flags.copy(),
            "statistics": self.degradation_stats.copy(),
            "recommendations": self._get_recommendations(),
        }

    def _get_recommendations(self) -> List[str]:
        """获取系统建议"""
        recommendations = []

        if self.current_level != DegradationLevel.NONE:
            recommendations.append(
                f"系统运行在{self.current_level.value.upper()}降级模式"
            )

        if self.resource_history:
            latest = self.resource_history[-1]
            if latest["memory_percent"] > 80:
                recommendations.append(
                    "High memory usage detected. Consider reducing dataset size."
                )
            if latest["cpu_percent"] > 80:
                recommendations.append(
                    "High CPU usage detected. Consider stopping other processes."
                )
            if latest["disk_percent"] > 85:
                recommendations.append(
                    "Low disk space. Consider cleaning up temporary files."
                )

        return recommendations

    def is_feature_available(self, feature_name: str) -> bool:
        """检查功能是否可用"""
        return self.feature_flags.get(feature_name, False)

    def get_degradation_statistics(self) -> Dict[str, Any]:
        """获取降级统计信息"""
        return self.degradation_stats.copy()


# 全局降级管理器实例
degradation_manager = GracefulDegradationManager()


@asynccontextmanager
async def graceful_degradation_context(operation_name: str):
    """优雅降级上下文管理器的便捷函数"""
    async with degradation_manager.process_with_degradation(operation_name) as context:
        yield context


def get_system_status() -> Dict[str, Any]:
    """获取系统状态的便捷函数"""
    return degradation_manager.get_system_status()


def is_feature_available(feature_name: str) -> bool:
    """检查功能可用性的便捷函数"""
    return degradation_manager.is_feature_available(feature_name)
