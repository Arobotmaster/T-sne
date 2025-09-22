"""
科学计算日志记录模块

遵循SDD Constitution的Scientific Observability原则，
提供详细的科学计算过程日志记录，包括算法执行、
数据处理、性能指标等信息。
"""

import logging
import time
import json
import psutil
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path
from contextlib import contextmanager

class ScientificLogger:
    """科学计算日志记录器"""

    def __init__(self, name: str, log_dir: str = "logs"):
        """
        初始化科学计算日志记录器

        Args:
            name: 日志记录器名称
            log_dir: 日志目录
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # 设置日志记录器
        self.logger = logging.getLogger(f"scientific.{name}")
        self.logger.setLevel(logging.DEBUG)

        # 创建科学计算日志文件处理器
        self._setup_scientific_handlers()

        # 性能监控数据
        self.performance_data = []

    def _setup_scientific_handlers(self):
        """设置科学计算专用的日志处理器"""
        # 科学计算日志文件
        scientific_file = logging.FileHandler(
            self.log_dir / f"scientific_{self.name}.log",
            encoding='utf-8'
        )
        scientific_file.setLevel(logging.DEBUG)

        # 科学计算日志格式
        scientific_formatter = logging.Formatter(
            '%(asctime)s [SCIENTIFIC] %(name)s: %(message)s'
        )
        scientific_file.setFormatter(scientific_formatter)

        # 性能指标日志文件
        performance_file = logging.FileHandler(
            self.log_dir / f"performance_{self.name}.log",
            encoding='utf-8'
        )
        performance_file.setLevel(logging.INFO)

        performance_formatter = logging.Formatter(
            '%(asctime)s [PERFORMANCE] %(name)s: %(message)s'
        )
        performance_file.setFormatter(performance_formatter)

        # 添加处理器
        self.logger.addHandler(scientific_file)
        self.logger.addHandler(performance_file)

    def log_algorithm_start(self, algorithm_name: str, config: Dict[str, Any]) -> None:
        """
        记录算法开始执行

        Args:
            algorithm_name: 算法名称
            config: 算法配置
        """
        self.logger.info(f"ALGORITHM_START: {algorithm_name}")
        self.logger.info(f"CONFIG: {json.dumps(config, ensure_ascii=False)}")

        # 记录系统资源状态
        self._log_system_resources()

    def log_algorithm_step(self, step_name: str, input_shape: tuple, **kwargs) -> None:
        """
        记录算法步骤

        Args:
            step_name: 步骤名称
            input_shape: 输入数据形状
            **kwargs: 其他参数
        """
        step_info = {
            "step": step_name,
            "input_shape": input_shape,
            "timestamp": datetime.now().isoformat()
        }
        step_info.update(kwargs)

        self.logger.info(f"ALGORITHM_STEP: {json.dumps(step_info, ensure_ascii=False)}")

    def log_algorithm_complete(self, algorithm_name: str, result_shape: tuple,
                             execution_time: float, metrics: Dict[str, Any]) -> None:
        """
        记录算法完成

        Args:
            algorithm_name: 算法名称
            result_shape: 结果形状
            execution_time: 执行时间
            metrics: 性能指标
        """
        completion_info = {
            "algorithm": algorithm_name,
            "result_shape": result_shape,
            "execution_time": execution_time,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }

        self.logger.info(f"ALGORITHM_COMPLETE: {json.dumps(completion_info, ensure_ascii=False)}")

        # 记录性能指标
        self._log_performance_metrics(algorithm_name, execution_time, metrics)

    def log_data_transformation(self, transformation_name: str,
                               input_shape: tuple, output_shape: tuple,
                               transformation_details: Dict[str, Any]) -> None:
        """
        记录数据转换过程

        Args:
            transformation_name: 转换名称
            input_shape: 输入形状
            output_shape: 输出形状
            transformation_details: 转换详情
        """
        transformation_info = {
            "transformation": transformation_name,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "details": transformation_details,
            "timestamp": datetime.now().isoformat()
        }

        self.logger.info(f"DATA_TRANSFORMATION: {json.dumps(transformation_info, ensure_ascii=False)}")

    def log_parameter_sensitivity(self, parameter_name: str, parameter_value: Any,
                                effect_on_result: Dict[str, Any]) -> None:
        """
        记录参数敏感性分析

        Args:
            parameter_name: 参数名称
            parameter_value: 参数值
            effect_on_result: 对结果的影响
        """
        sensitivity_info = {
            "parameter": parameter_name,
            "value": parameter_value,
            "effect": effect_on_result,
            "timestamp": datetime.now().isoformat()
        }

        self.logger.info(f"PARAMETER_SENSITIVITY: {json.dumps(sensitivity_info, ensure_ascii=False)}")

    def log_quality_metrics(self, data_name: str, metrics: Dict[str, Any]) -> None:
        """
        记录数据质量指标

        Args:
            data_name: 数据名称
            metrics: 质量指标
        """
        quality_info = {
            "data": data_name,
            "quality_metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }

        self.logger.info(f"QUALITY_METRICS: {json.dumps(quality_info, ensure_ascii=False)}")

    def log_memory_usage(self, operation: str, memory_mb: float,
                        peak_memory_mb: float) -> None:
        """
        记录内存使用情况

        Args:
            operation: 操作名称
            memory_mb: 内存使用量(MB)
            peak_memory_mb: 峰值内存使用量(MB)
        """
        memory_info = {
            "operation": operation,
            "memory_mb": memory_mb,
            "peak_memory_mb": peak_memory_mb,
            "timestamp": datetime.now().isoformat()
        }

        self.logger.info(f"MEMORY_USAGE: {json.dumps(memory_info, ensure_ascii=False)}")

    def log_error_event(self, error_type: str, error_message: str,
                       context: Dict[str, Any]) -> None:
        """
        记录错误事件

        Args:
            error_type: 错误类型
            error_message: 错误消息
            context: 错误上下文
        """
        error_info = {
            "error_type": error_type,
            "error_message": error_message,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }

        self.logger.error(f"ERROR_EVENT: {json.dumps(error_info, ensure_ascii=False)}")

    def _log_system_resources(self) -> None:
        """记录系统资源状态"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)

            # 内存使用情况
            memory = psutil.virtual_memory()

            # 磁盘使用情况
            disk = psutil.disk_usage('/')

            resources_info = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3)
            }

            self.logger.info(f"SYSTEM_RESOURCES: {json.dumps(resources_info, ensure_ascii=False)}")

        except Exception as e:
            self.logger.error(f"Failed to log system resources: {str(e)}")

    def _log_performance_metrics(self, algorithm_name: str, execution_time: float,
                                metrics: Dict[str, Any]) -> None:
        """记录性能指标"""
        performance_info = {
            "algorithm": algorithm_name,
            "execution_time": execution_time,
            "performance_metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }

        self.logger.info(f"PERFORMANCE_METRICS: {json.dumps(performance_info, ensure_ascii=False)}")

    @contextmanager
    def performance_timer(self, operation_name: str):
        """
        性能计时器上下文管理器

        Args:
            operation_name: 操作名称

        Yields:
            None
        """
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory

            timer_info = {
                "operation": operation_name,
                "execution_time": execution_time,
                "memory_delta_mb": memory_delta,
                "start_memory_mb": start_memory,
                "end_memory_mb": end_memory
            }

            self.logger.info(f"PERFORMANCE_TIMER: {json.dumps(timer_info, ensure_ascii=False)}")

    def log_reproducibility_info(self, algorithm_name: str,
                                random_seed: Optional[int] = None,
                                version_info: Optional[Dict[str, str]] = None) -> None:
        """
        记录可重现性信息

        Args:
            algorithm_name: 算法名称
            random_seed: 随机种子
            version_info: 版本信息
        """
        reproducibility_info = {
            "algorithm": algorithm_name,
            "random_seed": random_seed,
            "numpy_version": np.__version__,
            "timestamp": datetime.now().isoformat()
        }

        if version_info:
            reproducibility_info.update(version_info)

        self.logger.info(f"REPRODUCIBILITY: {json.dumps(reproducibility_info, ensure_ascii=False)}")

    def export_session_log(self, session_id: str, output_file: Optional[str] = None) -> str:
        """
        导出会话日志

        Args:
            session_id: 会话ID
            output_file: 输出文件路径

        Returns:
            str: 导出文件路径
        """
        if output_file is None:
            output_file = self.log_dir / f"session_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        session_data = {
            "session_id": session_id,
            "export_time": datetime.now().isoformat(),
            "performance_data": self.performance_data,
            "logger_name": self.name
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Session log exported to: {output_file}")
        return str(output_file)


def get_scientific_logger(name: str, log_dir: str = "logs") -> ScientificLogger:
    """
    获取科学计算日志记录器

    Args:
        name: 日志记录器名称
        log_dir: 日志目录

    Returns:
        ScientificLogger: 科学计算日志记录器实例
    """
    return ScientificLogger(name, log_dir)


# 预定义的科学计算日志记录器
algorithm_logger = get_scientific_logger("algorithms")
data_logger = get_scientific_logger("data_processing")
performance_logger = get_scientific_logger("performance")