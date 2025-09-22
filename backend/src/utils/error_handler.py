"""
错误处理和调试信息模块

遵循SDD Constitution的Scientific Observability原则，
提供统一的错误处理机制、调试信息记录和错误恢复策略。
"""

import logging
import traceback
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional, Union, List, Callable
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from functools import wraps

class ErrorSeverity(Enum):
    """错误严重程度"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """错误类别"""
    VALIDATION = "validation"
    FILE_IO = "file_io"
    NETWORK = "network"
    DATABASE = "database"
    ALGORITHM = "algorithm"
    MEMORY = "memory"
    AUTHORIZATION = "authorization"
    SYSTEM = "system"
    UNKNOWN = "unknown"

@dataclass
class ErrorContext:
    """错误上下文信息"""
    timestamp: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    stack_trace: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None

@dataclass
class DebugInfo:
    """调试信息"""
    timestamp: str
    level: str
    message: str
    module: str
    function: str
    line_number: int
    variables: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None

class ErrorHandler:
    """错误处理器"""

    def __init__(self, log_dir: str = "logs"):
        """
        初始化错误处理器

        Args:
            log_dir: 日志目录
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # 设置日志记录器
        self.logger = self._setup_logger()

        # 错误历史记录
        self.error_history: List[ErrorContext] = []
        self.max_error_history = 1000

        # 错误恢复策略
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {}

        # 调试信息记录
        self.debug_info_enabled = True
        self.debug_level = "INFO"

    def _setup_logger(self) -> logging.Logger:
        """设置错误日志记录器"""
        logger = logging.getLogger("error_handler")
        logger.setLevel(logging.DEBUG)

        # 避免重复添加处理器
        if logger.handlers:
            return logger

        # 错误日志文件处理器
        error_file = logging.FileHandler(
            self.log_dir / "errors.log",
            encoding='utf-8'
        )
        error_file.setLevel(logging.ERROR)

        # 调试日志文件处理器
        debug_file = logging.FileHandler(
            self.log_dir / "debug.log",
            encoding='utf-8'
        )
        debug_file.setLevel(logging.DEBUG)

        # 格式化器
        error_formatter = logging.Formatter(
            '%(asctime)s [ERROR] %(name)s: %(message)s'
        )
        debug_formatter = logging.Formatter(
            '%(asctime)s [DEBUG] %(name)s: %(message)s'
        )

        error_file.setFormatter(error_formatter)
        debug_file.setFormatter(debug_formatter)

        # 添加处理器
        logger.addHandler(error_file)
        logger.addHandler(debug_file)

        return logger

    def log_error(self, error: Exception, severity: ErrorSeverity = ErrorSeverity.ERROR,
                  category: ErrorCategory = ErrorCategory.UNKNOWN,
                  **context_info) -> ErrorContext:
        """
        记录错误信息

        Args:
            error: 异常对象
            severity: 错误严重程度
            category: 错误类别
            **context_info: 上下文信息

        Returns:
            ErrorContext: 错误上下文对象
        """
        # 创建错误上下文
        error_context = ErrorContext(
            timestamp=datetime.now().isoformat(),
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            category=category,
            stack_trace=traceback.format_exc(),
            additional_info=context_info
        )

        # 添加到历史记录
        self.error_history.append(error_context)

        # 限制历史记录大小
        if len(self.error_history) > self.max_error_history:
            self.error_history.pop(0)

        # 记录错误日志
        self._log_error_context(error_context)

        # 尝试错误恢复
        self._attempt_recovery(error_context)

        return error_context

    def _log_error_context(self, error_context: ErrorContext) -> None:
        """记录错误上下文到日志"""
        log_entry = {
            "timestamp": error_context.timestamp,
            "error_type": error_context.error_type,
            "error_message": error_context.error_message,
            "severity": error_context.severity.value,
            "category": error_context.category.value,
            "stack_trace": error_context.stack_trace,
            "additional_info": error_context.additional_info
        }

        # 根据严重程度选择日志级别
        if error_context.severity in [ErrorSeverity.ERROR, ErrorSeverity.CRITICAL]:
            self.logger.error(f"ERROR_CONTEXT: {json.dumps(log_entry, ensure_ascii=False)}")
        elif error_context.severity == ErrorSeverity.WARNING:
            self.logger.warning(f"ERROR_CONTEXT: {json.dumps(log_entry, ensure_ascii=False)}")
        else:
            self.logger.info(f"ERROR_CONTEXT: {json.dumps(log_entry, ensure_ascii=False)}")

    def log_debug_info(self, level: str, message: str, module: str,
                      function: str, line_number: int,
                      variables: Optional[Dict[str, Any]] = None,
                      context: Optional[Dict[str, Any]] = None) -> None:
        """
        记录调试信息

        Args:
            level: 调试级别
            message: 调试消息
            module: 模块名
            function: 函数名
            line_number: 行号
            variables: 变量信息
            context: 上下文信息
        """
        if not self.debug_info_enabled:
            return

        debug_info = DebugInfo(
            timestamp=datetime.now().isoformat(),
            level=level,
            message=message,
            module=module,
            function=function,
            line_number=line_number,
            variables=variables,
            context=context
        )

        # 记录调试日志
        debug_entry = asdict(debug_info)
        self.logger.debug(f"DEBUG_INFO: {json.dumps(debug_entry, ensure_ascii=False)}")

    def register_recovery_strategy(self, category: ErrorCategory,
                                  strategy: Callable[[ErrorContext], bool]) -> None:
        """
        注册错误恢复策略

        Args:
            category: 错误类别
            strategy: 恢复策略函数
        """
        self.recovery_strategies[category] = strategy

    def _attempt_recovery(self, error_context: ErrorContext) -> bool:
        """
        尝试错误恢复

        Args:
            error_context: 错误上下文

        Returns:
            bool: 恢复是否成功
        """
        strategy = self.recovery_strategies.get(error_context.category)
        if strategy:
            try:
                success = strategy(error_context)
                if success:
                    self.logger.info(f"Recovery successful for {error_context.error_type}")
                return success
            except Exception as recovery_error:
                self.logger.error(f"Recovery strategy failed: {str(recovery_error)}")
                return False
        return False

    def get_error_statistics(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """
        获取错误统计信息

        Args:
            time_range_hours: 时间范围（小时）

        Returns:
            Dict[str, Any]: 错误统计信息
        """
        cutoff_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_time = cutoff_time.replace(hour=cutoff_time.hour - time_range_hours)

        # 过滤错误
        recent_errors = [
            e for e in self.error_history
            if datetime.fromisoformat(e.timestamp) > cutoff_time
        ]

        if not recent_errors:
            return {"message": "No errors in the specified time range"}

        # 统计信息
        total_errors = len(recent_errors)
        errors_by_severity = {}
        errors_by_category = {}

        for error in recent_errors:
            # 按严重程度统计
            severity = error.severity.value
            errors_by_severity[severity] = errors_by_severity.get(severity, 0) + 1

            # 按类别统计
            category = error.category.value
            errors_by_category[category] = errors_by_category.get(category, 0) + 1

        # 最常见的错误类型
        error_types = {}
        for error in recent_errors:
            error_type = error.error_type
            error_types[error_type] = error_types.get(error_type, 0) + 1

        most_common_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "time_range_hours": time_range_hours,
            "total_errors": total_errors,
            "errors_by_severity": errors_by_severity,
            "errors_by_category": errors_by_category,
            "most_common_errors": most_common_errors,
            "error_rate_per_hour": total_errors / time_range_hours
        }

    def export_error_report(self, output_file: Optional[str] = None,
                           time_range_hours: int = 24) -> str:
        """
        导出错误报告

        Args:
            output_file: 输出文件路径
            time_range_hours: 时间范围（小时）

        Returns:
            str: 导出文件路径
        """
        if output_file is None:
            output_file = self.log_dir / f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        cutoff_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_time = cutoff_time.replace(hour=cutoff_time.hour - time_range_hours)

        # 过滤错误
        recent_errors = [
            e for e in self.error_history
            if datetime.fromisoformat(e.timestamp) > cutoff_time
        ]

        # 生成报告
        report = {
            "export_time": datetime.now().isoformat(),
            "time_range_hours": time_range_hours,
            "errors": [asdict(e) for e in recent_errors],
            "statistics": self.get_error_statistics(time_range_hours)
        }

        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        return str(output_file)

    @contextmanager
    def error_context(self, operation_name: str, **context_info):
        """
        错误上下文管理器

        Args:
            operation_name: 操作名称
            **context_info: 上下文信息

        Yields:
            None
        """
        try:
            yield
        except Exception as e:
            # 自动确定错误类别
            category = self._categorize_error(e)

            # 记录错误
            self.log_error(
                error=e,
                category=category,
                operation_name=operation_name,
                **context_info
            )
            raise

    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """
        自动分类错误

        Args:
            error: 异常对象

        Returns:
            ErrorCategory: 错误类别
        """
        error_type = type(error).__name__

        # 根据异常类型进行分类
        if error_type in ['ValueError', 'TypeError']:
            return ErrorCategory.VALIDATION
        elif error_type in ['FileNotFoundError', 'IOError', 'PermissionError']:
            return ErrorCategory.FILE_IO
        elif error_type in ['ConnectionError', 'TimeoutError']:
            return ErrorCategory.NETWORK
        elif 'Memory' in error_type:
            return ErrorCategory.MEMORY
        elif 'Database' in error_type or 'SQL' in error_type:
            return ErrorCategory.DATABASE
        elif 'Auth' in error_type or 'Permission' in error_type:
            return ErrorCategory.AUTHORIZATION
        elif 'Algorithm' in error_type or 'Computation' in error_type:
            return ErrorCategory.ALGORITHM
        else:
            return ErrorCategory.UNKNOWN

    def clear_error_history(self) -> None:
        """清除错误历史记录"""
        self.error_history.clear()

    def set_debug_level(self, level: str) -> None:
        """
        设置调试级别

        Args:
            level: 调试级别
        """
        self.debug_level = level.upper()

    def enable_debug_info(self, enabled: bool) -> None:
        """
        启用/禁用调试信息

        Args:
            enabled: 是否启用
        """
        self.debug_info_enabled = enabled


def error_handler_decorator(category: ErrorCategory = ErrorCategory.UNKNOWN,
                          operation_name: str = None):
    """
    错误处理装饰器

    Args:
        category: 错误类别
        operation_name: 操作名称
    """
    def decorator(func):
        name = operation_name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                global_error_handler.log_error(
                    error=e,
                    category=category,
                    operation_name=name,
                    function=func.__name__,
                    module=func.__module__
                )
                raise
        return wrapper
    return decorator


# 全局错误处理器实例
global_error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """获取全局错误处理器实例"""
    return global_error_handler


# 便捷的错误记录函数
def log_error(error: Exception, severity: ErrorSeverity = ErrorSeverity.ERROR,
              category: ErrorCategory = ErrorCategory.UNKNOWN, **context_info) -> ErrorContext:
    """便捷的错误记录函数"""
    return global_error_handler.log_error(error, severity, category, **context_info)


def log_debug_info(level: str, message: str, variables: Optional[Dict[str, Any]] = None,
                   context: Optional[Dict[str, Any]] = None) -> None:
    """便捷的调试信息记录函数"""
    import inspect
    frame = inspect.currentframe().f_back
    module = frame.f_globals.get('__name__', 'unknown')
    function = frame.f_code.co_name
    line_number = frame.f_lineno

    global_error_handler.log_debug_info(
        level=level,
        message=message,
        module=module,
        function=function,
        line_number=line_number,
        variables=variables,
        context=context
    )


# 预定义的错误恢复策略
def memory_recovery_strategy(error_context: ErrorContext) -> bool:
    """内存错误恢复策略"""
    try:
        import gc
        gc.collect()  # 强制垃圾回收
        return True
    except Exception:
        return False


def file_io_recovery_strategy(error_context: ErrorContext) -> bool:
    """文件IO错误恢复策略"""
    try:
        # 创建日志目录
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        return True
    except Exception:
        return False


# 注册预定义的恢复策略
global_error_handler.register_recovery_strategy(ErrorCategory.MEMORY, memory_recovery_strategy)
global_error_handler.register_recovery_strategy(ErrorCategory.FILE_IO, file_io_recovery_strategy)