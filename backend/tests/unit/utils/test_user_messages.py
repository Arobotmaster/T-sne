"""
用户消息管理器测试 - 符合SDD Constitution的Test-First原则
测试UserMessageManager的各种消息创建和管理功能
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from src.utils.user_messages import (
    UserMessage,
    UserMessageManager,
    MessageType,
    ErrorDomain,
    user_message_manager,
    create_user_message,
    create_info_message,
    create_warning_message,
    create_success_message,
    format_message_for_response,
    get_recent_messages,
)


class TestUserMessage:
    """用户消息测试类"""

    def test_initialization(self):
        """测试消息初始化"""
        message = UserMessage(
            message_type=MessageType.ERROR,
            domain=ErrorDomain.FILE_UPLOAD,
            title="Test Title",
            description="Test Description"
        )

        assert message.message_type == MessageType.ERROR
        assert message.domain == ErrorDomain.FILE_UPLOAD
        assert message.title == "Test Title"
        assert message.description == "Test Description"
        assert message.suggestions == []  # 默认值
        assert message.severity.value == "MEDIUM"  # 默认值


class TestUserMessageManager:
    """用户消息管理器测试类"""

    def setup_method(self):
        """测试前的设置"""
        self.manager = UserMessageManager("zh-CN")

    def test_initialization(self):
        """测试管理器初始化"""
        assert self.manager.language == "zh-CN"
        assert hasattr(self.manager, 'logger')
        assert hasattr(self.manager, 'error_handler')
        assert hasattr(self.manager, 'message_templates')
        assert len(self.manager.message_history) == 0
        assert self.manager.max_history_size == 1000

    def test_message_templates_initialization(self):
        """测试消息模板初始化"""
        templates = self.manager.message_templates

        # 验证主要错误域的模板存在
        assert ErrorDomain.FILE_UPLOAD in templates
        assert ErrorDomain.DATA_VALIDATION in templates
        assert ErrorDomain.DATA_PROCESSING in templates
        assert ErrorDomain.ALGORITHM in templates
        assert ErrorDomain.VISUALIZATION in templates
        assert ErrorDomain.EXPORT in templates
        assert ErrorDomain.SYSTEM in templates
        assert ErrorDomain.NETWORK in templates

        # 验证文件上传域的具体模板
        file_upload_templates = templates[ErrorDomain.FILE_UPLOAD]
        assert "size_exceeded" in file_upload_templates
        assert "invalid_format" in file_upload_templates
        assert "encoding_error" in file_upload_templates
        assert "empty_file" in file_upload_templates

    def test_error_code_mapping(self):
        """测试错误代码映射"""
        mapping = self.manager.error_code_mapping

        assert mapping["FILE_TOO_LARGE"] == "file_upload.size_exceeded"
        assert mapping["INVALID_FORMAT"] == "file_upload.invalid_format"
        assert mapping["MISSING_COLUMNS"] == "data_validation.missing_columns"
        assert mapping["ALGORITHM_FAILURE"] == "algorithm.algorithm_failure"

    def test_create_message_from_exception(self):
        """测试根据异常创建消息"""
        error = ValueError("File size exceeds limit")
        context = {"file_size": 150000000}

        message = self.manager.create_message(error, context)

        assert message.message_type == MessageType.ERROR
        assert message.domain == ErrorDomain.DATA_VALIDATION  # ValueError默认映射到此域
        assert isinstance(message, UserMessage)
        assert "ValueError" in message.technical_details

    def test_create_info_message(self):
        """测试创建信息消息"""
        message = self.manager.create_info_message(
            ErrorDomain.DATA_PROCESSING,
            "Processing Started",
            "Data processing has begun",
            ["Please wait", "This may take some time"]
        )

        assert message.message_type == MessageType.INFO
        assert message.domain == ErrorDomain.DATA_PROCESSING
        assert message.title == "Processing Started"
        assert len(message.suggestions) == 2
        assert message.severity.value == "LOW"

    def test_create_success_message(self):
        """测试创建成功消息"""
        message = self.manager.create_success_message(
            ErrorDomain.EXPORT,
            "Export Completed",
            "File exported successfully"
        )

        assert message.message_type == MessageType.SUCCESS
        assert message.domain == ErrorDomain.EXPORT
        assert message.title == "Export Completed"
        assert message.severity.value == "LOW"

    def test_create_warning_message(self):
        """测试创建警告消息"""
        message = self.manager.create_warning_message(
            ErrorDomain.ALGORITHM,
            "Parameter Warning",
            "Parameters may affect results",
            ["Consider adjusting parameters", "Check documentation"]
        )

        assert message.message_type == MessageType.WARNING
        assert message.domain == ErrorDomain.ALGORITHM
        assert message.title == "Parameter Warning"
        assert len(message.suggestions) == 2
        assert message.severity.value == "MEDIUM"

    def test_determine_error_domain(self):
        """测试确定错误域"""
        # 文件相关错误
        file_error = FileNotFoundError("File not found")
        assert self.manager._determine_error_domain(file_error, None) == ErrorDomain.FILE_UPLOAD

        # 数据验证错误
        value_error = ValueError("Invalid data")
        assert self.manager._determine_error_domain(value_error, None) == ErrorDomain.DATA_VALIDATION

        # 算法错误
        tsne_error = RuntimeError("t-SNE convergence failed")
        assert self.manager._determine_error_domain(tsne_error, None) == ErrorDomain.ALGORITHM

        # 系统错误
        memory_error = MemoryError("Out of memory")
        assert self.manager._determine_error_domain(memory_error, None) == ErrorDomain.DATA_PROCESSING

    def test_find_template_key(self):
        """测试查找模板键"""
        # 文件大小错误
        size_error = ValueError("File size exceeds 100MB limit")
        key = self.manager._find_template_key(size_error, ErrorDomain.FILE_UPLOAD, None)
        assert key == "size_exceeded"

        # 格式错误
        format_error = ValueError("Unsupported file format")
        key = self.manager._find_template_key(format_error, ErrorDomain.FILE_UPLOAD, None)
        assert key == "invalid_format"

        # 编码错误
        encoding_error = ValueError("Cannot detect file encoding")
        key = self.manager._find_template_key(encoding_error, ErrorDomain.FILE_UPLOAD, None)
        assert key == "encoding_error"

        # 默认模板
        unknown_error = ValueError("Unknown error occurred")
        key = self.manager._find_template_key(unknown_error, ErrorDomain.SYSTEM, None)
        assert key == "server_error"

    def test_get_template(self):
        """测试获取消息模板"""
        # 获取存在的模板
        template = self.manager._get_template(ErrorDomain.FILE_UPLOAD, "size_exceeded")

        assert "title" in template
        assert "description" in template
        assert "suggestions" in template
        assert template["title"] == "文件过大"

        # 获取不存在的模板（应该返回默认模板）
        default_template = self.manager._get_template(ErrorDomain.FILE_UPLOAD, "nonexistent")
        assert default_template["title"] == "未知错误"

    def test_determine_severity(self):
        """测试确定错误严重程度"""
        # 关键错误
        memory_error = MemoryError("Out of memory")
        severity = self.manager._determine_severity(memory_error)
        assert severity.value == "CRITICAL"

        # 高严重程度错误
        value_error = ValueError("Invalid data")
        severity = self.manager._determine_severity(value_error)
        assert severity.value == "HIGH"

        # 中等严重程度错误
        file_error = FileNotFoundError("File not found")
        severity = self.manager._determine_severity(file_error)
        assert severity.value == "MEDIUM"

    def test_to_dict(self):
        """测试消息转换为字典"""
        message = UserMessage(
            message_type=MessageType.ERROR,
            domain=ErrorDomain.FILE_UPLOAD,
            title="Test Title",
            description="Test Description",
            technical_details="Technical details"
        )

        message_dict = self.manager.to_dict(message)

        assert message_dict["type"] == "error"
        assert message_dict["domain"] == "file_upload"
        assert message_dict["title"] == "Test Title"
        assert message_dict["description"] == "Test Description"
        assert message_dict["technical_details"] == "Technical details"

    def test_to_response_format(self):
        """测试转换为API响应格式"""
        message = UserMessage(
            message_type=MessageType.ERROR,
            domain=ErrorDomain.FILE_UPLOAD,
            title="Test Title",
            description="Test Description"
        )

        response = self.manager.to_response_format(message)

        assert response["success"] is False  # 错误消息success为False
        assert "message" in response
        assert response["message"]["type"] == "error"
        assert response["message"]["title"] == "Test Title"
        assert "error_code" in response
        assert "technical_details" in response

    def test_to_response_format_success(self):
        """测试成功消息的响应格式"""
        message = UserMessage(
            message_type=MessageType.SUCCESS,
            domain=ErrorDomain.EXPORT,
            title="Success",
            description="Operation completed"
        )

        response = self.manager.to_response_format(message)

        assert response["success"] is True  # 成功消息success为True
        assert response["technical_details"] is None  # 成功消息无技术细节

    def test_message_history_management(self):
        """测试消息历史管理"""
        # 添加消息
        for i in range(5):
            message = self.manager.create_info_message(
                ErrorDomain.SYSTEM,
                f"Test Message {i}",
                f"Description {i}"
            )
            # 消息应该自动添加到历史

        assert len(self.manager.message_history) == 5

        # 获取最近的消息
        recent = self.manager.get_recent_messages(3)
        assert len(recent) == 3
        assert recent[0]["title"] == "Test Message 4"  # 最新的消息

        # 测试历史大小限制
        self.manager.max_history_size = 3
        self.manager._add_to_history(message)  # 添加额外消息
        assert len(self.manager.message_history) == 3  # 应该被限制

    def test_clear_history(self):
        """测试清空消息历史"""
        # 添加一些消息
        for i in range(3):
            self.manager.create_info_message(
                ErrorDomain.SYSTEM,
                f"Message {i}",
                f"Description {i}"
            )

        assert len(self.manager.message_history) == 3

        # 清空历史
        self.manager.clear_history()
        assert len(self.manager.message_history) == 0

    def test_get_recent_messages(self):
        """测试获取最近消息"""
        # 添加消息
        for i in range(10):
            self.manager.create_info_message(
                ErrorDomain.SYSTEM,
                f"Message {i}",
                f"Description {i}"
            )

        # 获取所有最近消息
        recent_all = self.manager.get_recent_messages()
        assert len(recent_all) == 10

        # 获取有限数量的最近消息
        recent_limited = self.manager.get_recent_messages(5)
        assert len(recent_limited) == 5

        # 验证顺序（最新的在前）
        assert recent_limited[0]["title"] == "Message 9"

    @patch('src.utils.user_messages.ScientificLogger')
    def test_logging_integration(self, mock_logger_class):
        """测试日志记录集成"""
        mock_logger = Mock()
        mock_logger_class.return_value = mock_logger

        manager = UserMessageManager()

        # 创建消息（应该触发日志记录）
        error = ValueError("Test error")
        manager.create_message(error)

        # 验证日志记录被调用
        mock_logger.log_error.assert_called()

    def test_message_types_enum(self):
        """测试消息类型枚举"""
        assert MessageType.ERROR.value == "error"
        assert MessageType.WARNING.value == "warning"
        assert MessageType.INFO.value == "info"
        assert MessageType.SUCCESS.value == "success"
        assert MessageType.SUGGESTION.value == "suggestion"

    def test_error_domains_enum(self):
        """测试错误域枚举"""
        assert ErrorDomain.FILE_UPLOAD.value == "file_upload"
        assert ErrorDomain.DATA_VALIDATION.value == "data_validation"
        assert ErrorDomain.DATA_PROCESSING.value == "data_processing"
        assert ErrorDomain.ALGORITHM.value == "algorithm"
        assert ErrorDomain.VISUALIZATION.value == "visualization"
        assert ErrorDomain.EXPORT.value == "export"
        assert ErrorDomain.SYSTEM.value == "system"
        assert ErrorDomain.NETWORK.value == "network"


class TestConvenienceFunctions:
    """便捷函数测试类"""

    def test_create_user_message(self):
        """测试创建用户消息便捷函数"""
        error = ValueError("Test error")

        with patch.object(user_message_manager, 'create_message') as mock_create:
            mock_create.return_value = "test_message"

            result = create_user_message(error)

            assert result == "test_message"
            mock_create.assert_called_once_with(error, None)

    def test_create_info_message(self):
        """测试创建信息消息便捷函数"""
        with patch.object(user_message_manager, 'create_info_message') as mock_create:
            mock_create.return_value = "info_message"

            result = create_info_message(
                ErrorDomain.SYSTEM,
                "Test Title",
                "Test Description",
                ["Suggestion 1"]
            )

            assert result == "info_message"
            mock_create.assert_called_once_with(
                ErrorDomain.SYSTEM,
                "Test Title",
                "Test Description",
                ["Suggestion 1"]
            )

    def test_create_warning_message(self):
        """测试创建警告消息便捷函数"""
        with patch.object(user_message_manager, 'create_warning_message') as mock_create:
            mock_create.return_value = "warning_message"

            result = create_warning_message(
                ErrorDomain.ALGORITHM,
                "Test Warning",
                "Test Warning Description"
            )

            assert result == "warning_message"
            mock_create.assert_called_once_with(
                ErrorDomain.ALGORITHM,
                "Test Warning",
                "Test Warning Description",
                None
            )

    def test_create_success_message(self):
        """测试创建成功消息便捷函数"""
        with patch.object(user_message_manager, 'create_success_message') as mock_create:
            mock_create.return_value = "success_message"

            result = create_success_message(
                ErrorDomain.EXPORT,
                "Test Success",
                "Test Success Description"
            )

            assert result == "success_message"
            mock_create.assert_called_once_with(
                ErrorDomain.EXPORT,
                "Test Success",
                "Test Success Description"
            )

    def test_format_message_for_response(self):
        """测试格式化消息为响应便捷函数"""
        test_message = UserMessage(
            message_type=MessageType.INFO,
            domain=ErrorDomain.SYSTEM,
            title="Test",
            description="Test description"
        )

        with patch.object(user_message_manager, 'to_response_format') as mock_format:
            mock_format.return_value = {"formatted": "response"}

            result = format_message_for_response(test_message)

            assert result == {"formatted": "response"}
            mock_format.assert_called_once_with(test_message)

    def test_get_recent_messages(self):
        """测试获取最近消息便捷函数"""
        with patch.object(user_message_manager, 'get_recent_messages') as mock_get:
            mock_get.return_value = [{"recent": "message"}]

            result = get_recent_messages(5)

            assert result == [{"recent": "message"}]
            mock_get.assert_called_once_with(5)