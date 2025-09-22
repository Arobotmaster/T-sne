"""
性能监控模块

遵循SDD Constitution的Scientific Observability原则，
收集和管理系统性能指标，包括计算时间、内存使用、
CPU利用率等关键性能指标。
"""

import time
import psutil
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from contextlib import contextmanager
from collections import deque

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: str
    operation_name: str
    execution_time: float
    memory_usage_mb: float
    memory_peak_mb: float
    cpu_usage_percent: float
    success: bool
    error_message: Optional[str] = None
    additional_metrics: Optional[Dict[str, Any]] = None

@dataclass
class SystemResources:
    """系统资源数据类"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_percent: float
    disk_free_gb: float
    network_io_bytes_sent: int
    network_io_bytes_recv: int

class PerformanceMonitor:
    """性能监控器"""

    def __init__(self, log_dir: str = "logs", max_history_size: int = 1000):
        """
        初始化性能监控器

        Args:
            log_dir: 日志目录
            max_history_size: 最大历史记录大小
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # 性能指标历史记录
        self.metrics_history: deque = deque(maxlen=max_history_size)
        self.system_resources_history: deque = deque(maxlen=max_history_size)

        # 监控线程控制
        self._monitoring = False
        self._monitor_thread = None
        self._monitor_interval = 5  # 秒

        # 性能阈值配置
        self.thresholds = {
            'memory_warning_mb': 1000,  # 1GB
            'memory_critical_mb': 2000,  # 2GB
            'cpu_warning_percent': 80,
            'cpu_critical_percent': 95,
            'execution_time_warning_seconds': 30,
            'execution_time_critical_seconds': 60
        }

        # 网络IO基准
        self._network_io_baseline = psutil.net_io_counters()

    def start_monitoring(self, interval: int = 5) -> None:
        """
        启动系统资源监控

        Args:
            interval: 监控间隔（秒）
        """
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_interval = interval
        self._monitor_thread = threading.Thread(target=self._monitor_system_resources, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        """停止系统资源监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)

    def _monitor_system_resources(self) -> None:
        """系统资源监控线程函数"""
        while self._monitoring:
            try:
                resources = self._collect_system_resources()
                self.system_resources_history.append(resources)

                # 检查资源警告
                self._check_resource_thresholds(resources)

            except Exception as e:
                print(f"Error in system resource monitoring: {e}")

            time.sleep(self._monitor_interval)

    def _collect_system_resources(self) -> SystemResources:
        """收集系统资源信息"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)

        # 内存使用情况
        memory = psutil.virtual_memory()

        # 磁盘使用情况
        disk = psutil.disk_usage('/')

        # 网络IO
        net_io = psutil.net_io_counters()
        network_io_bytes_sent = net_io.bytes_sent - self._network_io_baseline.bytes_sent
        network_io_bytes_recv = net_io.bytes_recv - self._network_io_baseline.bytes_recv

        return SystemResources(
            timestamp=datetime.now().isoformat(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available_gb=memory.available / (1024**3),
            disk_percent=disk.percent,
            disk_free_gb=disk.free / (1024**3),
            network_io_bytes_sent=network_io_bytes_sent,
            network_io_bytes_recv=network_io_bytes_recv
        )

    def _check_resource_thresholds(self, resources: SystemResources) -> None:
        """检查资源阈值警告"""
        warnings = []

        # 内存警告
        memory_used_gb = (100 - resources.memory_percent) * resources.memory_available_gb / 100
        memory_mb = memory_used_gb * 1024

        if memory_mb > self.thresholds['memory_critical_mb']:
            warnings.append(f"CRITICAL: Memory usage {memory_mb:.0f}MB exceeds critical threshold")
        elif memory_mb > self.thresholds['memory_warning_mb']:
            warnings.append(f"WARNING: Memory usage {memory_mb:.0f}MB exceeds warning threshold")

        # CPU警告
        if resources.cpu_percent > self.thresholds['cpu_critical_percent']:
            warnings.append(f"CRITICAL: CPU usage {resources.cpu_percent:.1f}% exceeds critical threshold")
        elif resources.cpu_percent > self.thresholds['cpu_warning_percent']:
            warnings.append(f"WARNING: CPU usage {resources.cpu_percent:.1f}% exceeds warning threshold")

        # 磁盘警告
        if resources.disk_percent > 90:
            warnings.append(f"WARNING: Disk usage {resources.disk_percent:.1f}% is high")

        # 记录警告
        for warning in warnings:
            self._log_warning(warning)

    def _log_warning(self, message: str) -> None:
        """记录警告信息"""
        warning_log = self.log_dir / "performance_warnings.log"
        timestamp = datetime.now().isoformat()
        log_entry = f"{timestamp} [WARNING] {message}\n"

        with open(warning_log, 'a', encoding='utf-8') as f:
            f.write(log_entry)

    @contextmanager
    def measure_performance(self, operation_name: str, **additional_metrics):
        """
        测量操作性能的上下文管理器

        Args:
            operation_name: 操作名称
            **additional_metrics: 额外的性能指标

        Yields:
            None
        """
        start_time = time.time()
        process = psutil.Process()

        # 记录开始时的内存和CPU
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = process.cpu_percent()

        success = True
        error_message = None

        try:
            yield
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            # 记录结束时的内存和CPU
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            end_cpu = process.cpu_percent()

            # 计算性能指标
            execution_time = end_time - start_time
            memory_usage = end_memory
            memory_peak = max(start_memory, end_memory)
            cpu_usage = (start_cpu + end_cpu) / 2

            # 检查执行时间阈值
            if execution_time > self.thresholds['execution_time_critical_seconds']:
                self._log_warning(f"CRITICAL: {operation_name} execution time {execution_time:.2f}s exceeds critical threshold")
            elif execution_time > self.thresholds['execution_time_warning_seconds']:
                self._log_warning(f"WARNING: {operation_name} execution time {execution_time:.2f}s exceeds warning threshold")

            # 创建性能指标对象
            metrics = PerformanceMetrics(
                timestamp=datetime.now().isoformat(),
                operation_name=operation_name,
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                memory_peak_mb=memory_peak,
                cpu_usage_percent=cpu_usage,
                success=success,
                error_message=error_message,
                additional_metrics=additional_metrics
            )

            # 添加到历史记录
            self.metrics_history.append(metrics)

            # 记录到文件
            self._log_performance_metrics(metrics)

    def _log_performance_metrics(self, metrics: PerformanceMetrics) -> None:
        """记录性能指标到文件"""
        metrics_file = self.log_dir / "performance_metrics.log"

        # 转换为字典
        metrics_dict = asdict(metrics)

        # 写入文件
        with open(metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metrics_dict, ensure_ascii=False) + '\n')

    def get_performance_summary(self, operation_name: Optional[str] = None,
                             time_range_hours: int = 24) -> Dict[str, Any]:
        """
        获取性能摘要

        Args:
            operation_name: 操作名称过滤器
            time_range_hours: 时间范围（小时）

        Returns:
            Dict[str, Any]: 性能摘要
        """
        # 过滤时间范围
        cutoff_time = datetime.now() - timedelta(hours=time_range_hours)

        # 过滤指标
        filtered_metrics = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m.timestamp) > cutoff_time
            and (operation_name is None or m.operation_name == operation_name)
        ]

        if not filtered_metrics:
            return {"error": "No performance data available"}

        # 计算统计信息
        execution_times = [m.execution_time for m in filtered_metrics]
        memory_usages = [m.memory_usage_mb for m in filtered_metrics]
        cpu_usages = [m.cpu_usage_percent for m in filtered_metrics]

        success_count = sum(1 for m in filtered_metrics if m.success)
        failure_count = len(filtered_metrics) - success_count

        summary = {
            "time_range_hours": time_range_hours,
            "operation_name": operation_name,
            "total_operations": len(filtered_metrics),
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": success_count / len(filtered_metrics) if filtered_metrics else 0,
            "execution_time_stats": {
                "min": min(execution_times),
                "max": max(execution_times),
                "mean": sum(execution_times) / len(execution_times),
                "median": sorted(execution_times)[len(execution_times) // 2]
            },
            "memory_usage_stats": {
                "min_mb": min(memory_usages),
                "max_mb": max(memory_usages),
                "mean_mb": sum(memory_usages) / len(memory_usages),
                "peak_mb": max(m.memory_peak_mb for m in filtered_metrics)
            },
            "cpu_usage_stats": {
                "min_percent": min(cpu_usages),
                "max_percent": max(cpu_usages),
                "mean_percent": sum(cpu_usages) / len(cpu_usages)
            }
        }

        return summary

    def export_performance_report(self, output_file: Optional[str] = None,
                                time_range_hours: int = 24) -> str:
        """
        导出性能报告

        Args:
            output_file: 输出文件路径
            time_range_hours: 时间范围（小时）

        Returns:
            str: 导出文件路径
        """
        if output_file is None:
            output_file = self.log_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # 收集所有性能数据
        cutoff_time = datetime.now() - timedelta(hours=time_range_hours)

        recent_metrics = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m.timestamp) > cutoff_time
        ]

        recent_resources = [
            r for r in self.system_resources_history
            if datetime.fromisoformat(r.timestamp) > cutoff_time
        ]

        # 生成报告
        report = {
            "export_time": datetime.now().isoformat(),
            "time_range_hours": time_range_hours,
            "performance_metrics": [asdict(m) for m in recent_metrics],
            "system_resources": [asdict(r) for r in recent_resources],
            "summary": self.get_performance_summary(time_range_hours=time_range_hours)
        }

        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        return str(output_file)

    def get_current_system_status(self) -> Dict[str, Any]:
        """获取当前系统状态"""
        resources = self._collect_system_resources()
        process = psutil.Process()

        return {
            "timestamp": datetime.now().isoformat(),
            "system_resources": asdict(resources),
            "current_process": {
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_percent": process.cpu_percent(),
                "thread_count": process.num_threads(),
                "open_files": process.num_fds()
            },
            "monitoring_status": {
                "is_monitoring": self._monitoring,
                "monitor_interval": self._monitor_interval,
                "metrics_history_size": len(self.metrics_history),
                "resources_history_size": len(self.system_resources_history)
            }
        }

    def clear_history(self) -> None:
        """清除历史记录"""
        self.metrics_history.clear()
        self.system_resources_history.clear()

    def set_thresholds(self, **thresholds) -> None:
        """
        设置性能阈值

        Args:
            **thresholds: 阈值参数
        """
        self.thresholds.update(thresholds)


# 全局性能监控器实例
performance_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """获取全局性能监控器实例"""
    return performance_monitor


# 便捷的装饰器
def measure_performance_decorator(operation_name: str = None):
    """
    性能测量装饰器

    Args:
        operation_name: 操作名称，如果为None则使用函数名
    """
    def decorator(func):
        name = operation_name or func.__name__

        def wrapper(*args, **kwargs):
            with performance_monitor.measure_performance(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator