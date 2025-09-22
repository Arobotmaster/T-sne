"""
统一错误处理器测试 - 符合SDD Constitution的Test-First原则
测试UnifiedErrorHandler的各种异常处理功能
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi import Request, HTTPException
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import uuid
from datetime import datetime

from src.api.error_handlers import (
    UnifiedErrorHandler,
    setup_error_handling,
    raise_http_error,
    raise_validation_error,
    get_error_statistics,
)


class TestUnifiedErrorHandler:
    """统一错误处理器测试类"""

    def setup_method(self):
        """测试前的设置"""
        self.error_handler = UnifiedErrorHandler("test_app")

    def test_initialization(self):
        """测试初始化"""
        assert self.error_handler.app_name == "test_app"
        assert self.error_handler.error_stats == {}
        assert hasattr(self.error_handler, 'error_handler')
        assert hasattr(self.error_handler, 'scientific_logger')

    @pytest.mark.asyncio
    async def test_handle_http_exception(self):
        """测试HTTP异常处理"""
        # 创建模拟请求和异常
        request = Mock(spec=Request)
        request.url.path = "/test"
        request.method = "GET"

        exc = StarletteHTTPException(status_code=404, detail="Not Found")

        # 调用处理方法
        response = await self.error_handler._handle_http_exception(request, exc)

        # 验证响应
        assert response.status_code == 404
        response_data = response.body.decode()
        assert "success" in response_data
        assert "error_id" in response_data
        assert "Not Found" in response_data

    @pytest.mark.asyncio
    async def test_handle_validation_exception(self):
        """测试验证异常处理"""
        request = Mock(spec=Request)
        request.url.path = "/upload"
        request.method = "POST"

        # 模拟验证错误
        validation_errors = [
            {
                "loc": ["body", "file"],
                "msg": "field required",
                "type": "value_error.missing"
            }
        ]
        exc = RequestValidationError(validation_errors)

        response = await self.error_handler._handle_validation_exception(request, exc)

        assert response.status_code == 422
        response_data = response.body.decode()
        assert "validation_errors" in response_data

    @pytest.mark.asyncio
    async def test_handle_value_error(self):
        """测试数值错误处理"""
        request = Mock(spec=Request)
        request.url.path = "/process"
        request.method = "POST"

        exc = ValueError("Invalid data format")

        response = await self.error_handler._handle_value_error(request, exc)

        assert response.status_code == 400
        response_data = response.body.decode()
        assert "数据处理错误" in response_data

    @pytest.mark.asyncio
    async def test_handle_key_error(self):
        """测试键错误处理"""
        request = Mock(spec=Request)
        request.url.path = "/datasets"
        request.method = "GET"

        exc = KeyError("missing_column")

        response = await self.error_handler._handle_key_error(request, exc)

        assert response.status_code == 400
        response_data = response.body.decode()
        assert "数据键缺失" in response_data

    @pytest.mark.asyncio
    async def test_handle_general_exception(self):
        """测试通用异常处理"""
        request = Mock(spec=Request)
        request.url.path = "/api/data"
        request.method = "GET"

        exc = RuntimeError("Database connection failed")

        response = await self.error_handler._handle_general_exception(request, exc)

        assert response.status_code == 500
        response_data = response.body.decode()
        assert "服务器内部错误" in response_data

    def test_get_severity_from_status(self):
        """测试根据状态码确定严重程度"""
        # 测试不同状态码的严重程度
        assert self.error_handler._get_severity_from_status(500).name == "CRITICAL"
        assert self.error_handler._get_severity_from_status(404).name == "WARNING"
        assert self.error_handler._get_severity_from_status(200).name == "INFO"

    def test_update_error_stats(self):
        """测试错误统计更新"""
        initial_total = sum(self.error_handler.error_stats.values())

        self.error_handler._update_error_stats("test_error")

        assert self.error_handler.error_stats["test_error"] == 1
        assert sum(self.error_handler.error_stats.values()) == initial_total + 1

    def test_get_error_statistics(self):
        """测试获取错误统计信息"""
        # 添加一些测试数据
        self.error_handler.error_stats = {
            "http_404": 5,
            "validation_error": 3,
            "value_error": 2
        }

        stats = self.error_handler.get_error_statistics()

        assert stats["total_errors"] == 10
        assert stats["error_breakdown"]["http_404"] == 5
        assert stats["most_common_error"] == "http_404"

    def test_register_exception_handlers(self):
        """测试异常处理器注册"""
        mock_app = Mock()

        self.error_handler.register_exception_handlers(mock_app)

        # 验证所有异常处理器都被注册
        assert mock_app.exception_handler.call_count >= 4  # 至少4种异常类型

    def test_setup_error_handling(self):
        """测试错误处理系统设置"""
        mock_app = Mock()

        with patch('src.api.error_handlers.unified_error_handler') as mock_handler:
            setup_error_handling(mock_app)

            mock_handler.register_exception_handlers.assert_called_once_with(mock_app)
            mock_handler.scientific_logger.logger.info.assert_called_once()

    def test_raise_http_error(self):
        """测试HTTP错误抛出函数"""
        with pytest.raises(HTTPException) as exc_info:
            raise_http_error(400, "Bad Request", {"field": "test"})

        assert exc_info.value.status_code == 400
        assert "Bad Request" in str(exc_info.value.detail)

    def test_raise_validation_error(self):
        """测试验证错误抛出函数"""
        with pytest.raises(ValueError) as exc_info:
            raise_validation_error("email", "Invalid email format")

        assert "Validation failed for field 'email'" in str(exc_info.value)

    def test_get_error_statistics_convenience(self):
        """测试获取错误统计的便捷函数"""
        with patch('src.api.error_handlers.unified_error_handler') as mock_handler:
            mock_handler.get_error_statistics.return_value = {"total": 5}

            result = get_error_statistics()

            assert result == {"total": 5}
            mock_handler.get_error_statistics.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_id_generation(self):
        """测试错误ID生成"""
        request = Mock(spec=Request)
        request.url.path = "/test"
        request.method = "GET"

        exc = StarletteHTTPException(status_code=400, detail="Test error")

        response1 = await self.error_handler._handle_http_exception(request, exc)
        response2 = await self.error_handler._handle_http_exception(request, exc)

        # 验证每次都生成不同的错误ID
        response_data1 = response1.body.decode()
        response_data2 = response2.body.decode()

        import json
        data1 = json.loads(response_data1)
        data2 = json.loads(response_data2)

        assert data1["request_id"] != data2["request_id"]

    @pytest.mark.asyncio
    async def test_logging_integration(self):
        """测试日志记录集成"""
        request = Mock(spec=Request)
        request.url.path = "/test"
        request.method = "GET"

        exc = StarletteHTTPException(status_code=500, detail="Server Error")

        with patch.object(self.error_handler.scientific_logger, 'log_error_event') as mock_log:
            with patch.object(self.error_handler.error_handler, 'log_error') as mock_handle:
                await self.error_handler._handle_http_exception(request, exc)

                # 验证日志记录被调用
                mock_log.assert_called_once()
                mock_handle.assert_called_once()

    def test_error_stats_increment(self):
        """测试错误统计计数器增加"""
        initial_count = self.error_handler.error_stats.get("http_500", 0)

        # 模拟多次相同错误
        self.error_handler._update_error_stats("http_500")
        self.error_handler._update_error_stats("http_500")

        assert self.error_handler.error_stats["http_500"] == initial_count + 2