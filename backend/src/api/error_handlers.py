"""
统一的错误处理机制 - 符合SDD Constitution的Scientific Observability原则
提供FastAPI应用的统一错误处理、日志记录和错误分类
"""

import traceback
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import uuid

from ..utils.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
from ..utils.scientific_logging import ScientificLogger


class UnifiedErrorHandler:
    """统一错误处理器 - 集成科学日志和错误处理"""

    def __init__(self, app_name: str = "MOF_Visualization"):
        self.app_name = app_name
        self.error_handler = ErrorHandler(log_dir="logs")
        self.scientific_logger = ScientificLogger("error_handling", log_dir="logs")

        # 错误统计
        self.error_stats: Dict[str, int] = {}

    def register_exception_handlers(self, app):
        """注册所有异常处理器到FastAPI应用"""

        @app.exception_handler(StarletteHTTPException)
        async def http_exception_handler(request: Request, exc: StarletteHTTPException):
            """处理HTTP异常"""
            return await self._handle_http_exception(request, exc)

        @app.exception_handler(RequestValidationError)
        async def validation_exception_handler(
            request: Request, exc: RequestValidationError
        ):
            """处理请求验证异常"""
            return await self._handle_validation_exception(request, exc)

        @app.exception_handler(ValueError)
        async def value_error_handler(request: Request, exc: ValueError):
            """处理数值错误"""
            return await self._handle_value_error(request, exc)

        @app.exception_handler(KeyError)
        async def key_error_handler(request: Request, exc: KeyError):
            """处理键错误"""
            return await self._handle_key_error(request, exc)

        @app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            """处理通用异常"""
            return await self._handle_general_exception(request, exc)

    async def _handle_http_exception(
        self, request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """处理HTTP异常"""
        error_id = str(uuid.uuid4())

        # 记录科学日志
        self.scientific_logger.log_error_event(
            error_type="HTTP_EXCEPTION",
            error_message=f"HTTP {exc.status_code}: {exc.detail}",
            context={
                "status_code": exc.status_code,
                "detail": exc.detail,
                "error_id": error_id,
                "path": request.url.path,
                "method": request.method,
            },
        )

        # 使用错误处理器记录
        self.error_handler.log_error(
            error=exc,
            severity=self._get_severity_from_status(exc.status_code),
            category=ErrorCategory.NETWORK,
            request_path=request.url.path,
            method=request.method,
            error_id=error_id,
        )

        # 更新统计
        self._update_error_stats(f"http_{exc.status_code}")

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "timestamp": datetime.now().isoformat(),
                "request_id": error_id,
                "error": {
                    "code": f"HTTP_{exc.status_code}",
                    "message": exc.detail,
                    "details": {
                        "error_id": error_id,
                        "path": request.url.path,
                    },
                },
            },
        )

    async def _handle_validation_exception(
        self, request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """处理请求验证异常"""
        error_id = str(uuid.uuid4())

        # 记录科学日志
        self.scientific_logger.log_error_event(
            error_type="VALIDATION_ERROR",
            error_message="Request validation failed",
            context={
                "validation_errors": exc.errors(),
                "error_id": error_id,
                "path": request.url.path,
                "method": request.method,
            },
        )

        # 使用错误处理器记录
        self.error_handler.log_error(
            error=exc,
            severity=ErrorSeverity.WARNING,
            category=ErrorCategory.VALIDATION,
            request_path=request.url.path,
            method=request.method,
            error_id=error_id,
            validation_details=exc.errors(),
        )

        # 更新统计
        self._update_error_stats("validation_error")

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "timestamp": datetime.now().isoformat(),
                "request_id": error_id,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "请求参数验证失败",
                    "details": {
                        "error_id": error_id,
                        "validation_errors": exc.errors(),
                        "path": request.url.path,
                    },
                },
            },
        )

    async def _handle_value_error(
        self, request: Request, exc: ValueError
    ) -> JSONResponse:
        """处理数值错误"""
        error_id = str(uuid.uuid4())

        # 记录科学日志
        self.scientific_logger.log_error_event(
            error_type="VALUE_ERROR",
            error_message=str(exc),
            context={
                "error_id": error_id,
                "path": request.url.path,
                "method": request.method,
                "traceback": traceback.format_exc(),
            },
        )

        # 使用错误处理器记录
        self.error_handler.log_error(
            error=exc,
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.ALGORITHM,
            request_path=request.url.path,
            method=request.method,
            error_id=error_id,
        )

        # 更新统计
        self._update_error_stats("value_error")

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "timestamp": datetime.now().isoformat(),
                "request_id": error_id,
                "error": {
                    "code": "VALUE_ERROR",
                    "message": "数据处理错误",
                    "details": {
                        "error_id": error_id,
                        "error_message": str(exc),
                        "path": request.url.path,
                    },
                },
            },
        )

    async def _handle_key_error(self, request: Request, exc: KeyError) -> JSONResponse:
        """处理键错误"""
        error_id = str(uuid.uuid4())

        # 记录科学日志
        self.scientific_logger.log_error_event(
            error_type="KEY_ERROR",
            error_message=f"Missing key: {str(exc)}",
            context={
                "missing_key": str(exc),
                "error_id": error_id,
                "path": request.url.path,
                "method": request.method,
                "traceback": traceback.format_exc(),
            },
        )

        # 使用错误处理器记录
        self.error_handler.log_error(
            error=exc,
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.ALGORITHM,
            request_path=request.url.path,
            method=request.method,
            error_id=error_id,
            missing_key=str(exc),
        )

        # 更新统计
        self._update_error_stats("key_error")

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "timestamp": datetime.now().isoformat(),
                "request_id": error_id,
                "error": {
                    "code": "KEY_ERROR",
                    "message": "数据键缺失",
                    "details": {
                        "error_id": error_id,
                        "missing_key": str(exc),
                        "path": request.url.path,
                    },
                },
            },
        )

    async def _handle_general_exception(
        self, request: Request, exc: Exception
    ) -> JSONResponse:
        """处理通用异常"""
        error_id = str(uuid.uuid4())

        # 记录科学日志
        self.scientific_logger.log_error_event(
            error_type="GENERAL_EXCEPTION",
            error_message=str(exc),
            context={
                "exception_type": type(exc).__name__,
                "error_id": error_id,
                "path": request.url.path,
                "method": request.method,
                "traceback": traceback.format_exc(),
            },
        )

        # 使用错误处理器记录
        self.error_handler.log_error(
            error=exc,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.SYSTEM,
            request_path=request.url.path,
            method=request.method,
            error_id=error_id,
            exception_type=type(exc).__name__,
        )

        # 更新统计
        self._update_error_stats("general_exception")

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "timestamp": datetime.now().isoformat(),
                "request_id": error_id,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "服务器内部错误",
                    "details": {
                        "error_id": error_id,
                        "path": request.url.path,
                        "exception_type": type(exc).__name__,
                    },
                },
            },
        )

    def _get_severity_from_status(self, status_code: int) -> ErrorSeverity:
        """根据HTTP状态码确定错误严重程度"""
        if status_code >= 500:
            return ErrorSeverity.CRITICAL
        elif status_code >= 400:
            return ErrorSeverity.WARNING
        else:
            return ErrorSeverity.INFO

    def _update_error_stats(self, error_type: str):
        """更新错误统计"""
        self.error_stats[error_type] = self.error_stats.get(error_type, 0) + 1

        # 记录统计信息到科学日志
        self.scientific_logger.log_quality_metrics(
            data_name="error_statistics",
            metrics={
                "error_type": error_type,
                "count": self.error_stats[error_type],
                "unit": "count",
            },
        )

    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        return {
            "total_errors": sum(self.error_stats.values()),
            "error_breakdown": self.error_stats.copy(),
            "most_common_error": (
                max(self.error_stats.items(), key=lambda x: x[1])[0]
                if self.error_stats
                else None
            ),
        }


# 全局错误处理器实例
unified_error_handler = UnifiedErrorHandler()


def setup_error_handling(app):
    """设置FastAPI应用的错误处理"""
    unified_error_handler.register_exception_handlers(app)

    # 记录启动信息
    unified_error_handler.scientific_logger.logger.info(
        f"Error handling system initialized for app: {unified_error_handler.app_name}"
    )


# 便捷的错误抛出函数
def raise_http_error(status_code: int, detail: str, context: Optional[Dict] = None):
    """抛出HTTP错误并记录日志"""
    error_id = str(uuid.uuid4())

    unified_error_handler.scientific_logger.log_error_event(
        error_type="RAISED_HTTP_ERROR",
        error_message=f"HTTP {status_code}: {detail}",
        context={
            "status_code": status_code,
            "detail": detail,
            "error_id": error_id,
            "context": context or {},
        },
    )

    raise HTTPException(
        status_code=status_code,
        detail={"error_id": error_id, "message": detail, "context": context},
    )


def raise_validation_error(
    field_name: str, message: str, context: Optional[Dict] = None
):
    """抛出验证错误"""
    error_id = str(uuid.uuid4())

    unified_error_handler.scientific_logger.log_error_event(
        error_type="RAISED_VALIDATION_ERROR",
        error_message=f"Validation failed for field '{field_name}': {message}",
        context={
            "field_name": field_name,
            "message": message,
            "error_id": error_id,
            "context": context or {},
        },
    )

    raise ValueError(f"Validation failed for field '{field_name}': {message}")


def get_error_statistics() -> Dict[str, Any]:
    """获取错误统计信息的便捷函数"""
    return unified_error_handler.get_error_statistics()
