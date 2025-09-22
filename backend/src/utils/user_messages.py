"""
用户友好的错误消息系统 - 符合SDD Constitution的Anti-Abstraction原则
提供清晰、易懂的错误消息，帮助用户理解和解决问题
"""

from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass

from .scientific_logging import ScientificLogger
from .error_handler import ErrorHandler, ErrorSeverity


class MessageType(Enum):
    """消息类型"""

    ERROR = "error"  # 错误消息
    WARNING = "warning"  # 警告消息
    INFO = "info"  # 信息消息
    SUCCESS = "success"  # 成功消息
    SUGGESTION = "suggestion"  # 建议消息


class ErrorDomain(Enum):
    """错误领域"""

    FILE_UPLOAD = "file_upload"  # 文件上传
    DATA_VALIDATION = "data_validation"  # 数据验证
    DATA_PROCESSING = "data_processing"  # 数据处理
    ALGORITHM = "algorithm"  # 算法执行
    VISUALIZATION = "visualization"  # 可视化
    EXPORT = "export"  # 导出功能
    SYSTEM = "system"  # 系统错误
    NETWORK = "network"  # 网络错误


@dataclass
class UserMessage:
    """用户友好的消息"""

    message_type: MessageType
    domain: ErrorDomain
    title: str
    description: str
    technical_details: Optional[str] = None
    suggestions: List[str] = None
    error_code: Optional[str] = None
    severity: ErrorSeverity = ErrorSeverity.ERROR

    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []


class UserMessageManager:
    """用户消息管理器"""

    def __init__(self, language: str = "zh-CN"):
        self.language = language
        self.logger = ScientificLogger("user_messages", log_dir="logs")
        self.error_handler = ErrorHandler(log_dir="logs")

        # 消息模板库
        self.message_templates = self._initialize_message_templates()

        # 消息历史
        self.message_history: List[UserMessage] = []
        self.max_history_size = 1000

        # 错误代码映射
        self.error_code_mapping = self._initialize_error_code_mapping()

    def _initialize_message_templates(
        self,
    ) -> Dict[ErrorDomain, Dict[str, Dict[str, str]]]:
        """初始化消息模板"""
        return {
            ErrorDomain.FILE_UPLOAD: {
                "size_exceeded": {
                    "title": "文件过大",
                    "description": "您上传的文件超过了最大允许大小（100MB）。",
                    "suggestions": [
                        "压缩文件或减少数据量",
                        "删除不必要的列",
                        "使用数据采样功能",
                    ],
                },
                "invalid_format": {
                    "title": "文件格式不支持",
                    "description": "仅支持CSV格式的文件。",
                    "suggestions": [
                        "将文件转换为CSV格式",
                        "使用Excel或文本编辑器另存为CSV",
                        "确保文件扩展名为.csv",
                    ],
                },
                "encoding_error": {
                    "title": "文件编码问题",
                    "description": "无法读取文件的编码格式。",
                    "suggestions": [
                        "将文件保存为UTF-8编码",
                        "尝试使用GBK编码重新保存",
                        "检查文件是否损坏",
                    ],
                },
                "empty_file": {
                    "title": "文件为空",
                    "description": "上传的文件不包含任何数据。",
                    "suggestions": [
                        "检查文件是否正确保存",
                        "确保文件包含数据行",
                        "重新导出数据",
                    ],
                },
            },
            ErrorDomain.DATA_VALIDATION: {
                "missing_columns": {
                    "title": "缺少必需列",
                    "description": "数据文件中缺少必需的列（mofid, category）。",
                    "suggestions": [
                        "确保数据包含样本标识符列",
                        "确保数据包含分类标签列",
                        "检查列名拼写是否正确",
                    ],
                },
                "insufficient_numeric": {
                    "title": "数值列不足",
                    "description": "需要至少2个数值列进行分析。",
                    "suggestions": [
                        "确保数据包含数值特征",
                        "检查数值列格式是否正确",
                        "删除非数值列或转换为数值",
                    ],
                },
                "too_many_categories": {
                    "title": "分类过多",
                    "description": "发现的分类数量可能过多，影响可视化效果。",
                    "suggestions": [
                        "合并相似的分类",
                        "考虑数据预处理",
                        "检查分类是否正确",
                    ],
                },
                "data_quality_issues": {
                    "title": "数据质量问题",
                    "description": "检测到数据质量问题，可能影响分析结果。",
                    "suggestions": ["检查数据完整性", "处理缺失值", "移除异常值"],
                },
            },
            ErrorDomain.DATA_PROCESSING: {
                "memory_limit": {
                    "title": "内存不足",
                    "description": "数据量过大，系统内存不足。",
                    "suggestions": ["减少数据量", "使用数据采样", "分批处理数据"],
                },
                "processing_timeout": {
                    "title": "处理超时",
                    "description": "数据处理时间过长，已超时。",
                    "suggestions": ["减少数据量", "调整算法参数", "使用更简单的配置"],
                },
                "algorithm_failure": {
                    "title": "算法执行失败",
                    "description": "数据处理算法执行过程中发生错误。",
                    "suggestions": [
                        "检查数据格式",
                        "调整算法参数",
                        "尝试不同的算法配置",
                    ],
                },
            },
            ErrorDomain.ALGORITHM: {
                "tsne_convergence": {
                    "title": "t-SNE未收敛",
                    "description": "t-SNE算法未能正常收敛。",
                    "suggestions": [
                        "增加迭代次数",
                        "调整perplexity参数",
                        "检查数据质量",
                    ],
                },
                "invalid_parameters": {
                    "title": "参数无效",
                    "description": "提供的算法参数不在有效范围内。",
                    "suggestions": ["检查参数范围", "使用默认参数", "参考算法文档"],
                },
            },
            ErrorDomain.VISUALIZATION: {
                "render_error": {
                    "title": "可视化渲染失败",
                    "description": "无法生成可视化图表。",
                    "suggestions": ["刷新页面", "检查浏览器兼容性", "减少数据量"],
                },
                "browser_compatibility": {
                    "title": "浏览器兼容性问题",
                    "description": "当前浏览器可能不完全支持所有功能。",
                    "suggestions": [
                        "使用最新版Chrome或Firefox",
                        "启用JavaScript",
                        "清除浏览器缓存",
                    ],
                },
            },
            ErrorDomain.EXPORT: {
                "export_failed": {
                    "title": "导出失败",
                    "description": "无法导出可视化图像。",
                    "suggestions": ["检查文件权限", "减少图像分辨率", "尝试其他格式"],
                },
                "file_permission": {
                    "title": "文件权限错误",
                    "description": "没有权限写入指定位置。",
                    "suggestions": [
                        "选择其他保存位置",
                        "检查文件权限",
                        "以管理员身份运行",
                    ],
                },
            },
            ErrorDomain.SYSTEM: {
                "server_error": {
                    "title": "服务器错误",
                    "description": "服务器遇到了内部错误。",
                    "suggestions": ["稍后重试", "检查网络连接", "联系技术支持"],
                },
                "maintenance": {
                    "title": "系统维护",
                    "description": "系统正在维护中，暂时无法使用。",
                    "suggestions": ["等待维护完成", "查看维护公告", "联系管理员"],
                },
            },
            ErrorDomain.NETWORK: {
                "connection_error": {
                    "title": "连接错误",
                    "description": "无法连接到服务器。",
                    "suggestions": ["检查网络连接", "确认服务器地址", "稍后重试"],
                },
                "timeout": {
                    "title": "请求超时",
                    "description": "服务器响应超时。",
                    "suggestions": ["检查网络连接", "减少数据量", "稍后重试"],
                },
            },
        }

    def _initialize_error_code_mapping(self) -> Dict[str, str]:
        """初始化错误代码映射"""
        return {
            "FILE_TOO_LARGE": "file_upload.size_exceeded",
            "INVALID_FORMAT": "file_upload.invalid_format",
            "ENCODING_ERROR": "file_upload.encoding_error",
            "MISSING_COLUMNS": "data_validation.missing_columns",
            "INSUFFICIENT_NUMERIC": "data_validation.insufficient_numeric",
            "MEMORY_LIMIT": "data_processing.memory_limit",
            "ALGORITHM_FAILURE": "algorithm.algorithm_failure",
            "TSNE_CONVERGENCE": "algorithm.tsne_convergence",
            "SERVER_ERROR": "system.server_error",
            "CONNECTION_ERROR": "network.connection_error",
        }

    def create_message(
        self, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> UserMessage:
        """根据异常创建用户友好的消息"""
        error_type = type(error).__name__
        error_message = str(error)

        # 确定错误领域
        domain = self._determine_error_domain(error, context)

        # 查找对应的模板
        template_key = self._find_template_key(error, domain, context)
        template = self._get_template(domain, template_key)

        # 创建消息
        message = UserMessage(
            message_type=MessageType.ERROR,
            domain=domain,
            title=template["title"],
            description=template["description"],
            technical_details=f"{error_type}: {error_message}",
            suggestions=template.get("suggestions", []),
            error_code=template_key,
            severity=self._determine_severity(error),
        )

        # 记录消息
        self._log_message(message, context)

        # 添加到历史
        self._add_to_history(message)

        return message

    def create_info_message(
        self,
        domain: ErrorDomain,
        title: str,
        description: str,
        suggestions: Optional[List[str]] = None,
    ) -> UserMessage:
        """创建信息消息"""
        message = UserMessage(
            message_type=MessageType.INFO,
            domain=domain,
            title=title,
            description=description,
            suggestions=suggestions or [],
            severity=ErrorSeverity.LOW,
        )

        self._add_to_history(message)
        return message

    def create_success_message(
        self, domain: ErrorDomain, title: str, description: str
    ) -> UserMessage:
        """创建成功消息"""
        message = UserMessage(
            message_type=MessageType.SUCCESS,
            domain=domain,
            title=title,
            description=description,
            severity=ErrorSeverity.LOW,
        )

        self._add_to_history(message)
        return message

    def create_warning_message(
        self,
        domain: ErrorDomain,
        title: str,
        description: str,
        suggestions: Optional[List[str]] = None,
    ) -> UserMessage:
        """创建警告消息"""
        message = UserMessage(
            message_type=MessageType.WARNING,
            domain=domain,
            title=title,
            description=description,
            suggestions=suggestions or [],
            severity=ErrorSeverity.ERROR,
        )

        self._add_to_history(message)
        return message

    def _determine_error_domain(
        self, error: Exception, context: Optional[Dict[str, Any]]
    ) -> ErrorDomain:
        """确定错误领域"""
        error_type = type(error).__name__

        # 根据异常类型确定领域
        if error_type in ["FileNotFoundError", "FileTooLargeError"]:
            return ErrorDomain.FILE_UPLOAD
        elif error_type in ["ValueError", "KeyError"]:
            return ErrorDomain.DATA_VALIDATION
        elif error_type in ["MemoryError", "TimeoutError"]:
            return ErrorDomain.DATA_PROCESSING
        elif "tsne" in str(error).lower() or "pca" in str(error).lower():
            return ErrorDomain.ALGORITHM
        elif "export" in str(error).lower():
            return ErrorDomain.EXPORT
        elif "connection" in str(error).lower() or "network" in str(error).lower():
            return ErrorDomain.NETWORK
        else:
            return ErrorDomain.SYSTEM

    def _find_template_key(
        self, error: Exception, domain: ErrorDomain, context: Optional[Dict[str, Any]]
    ) -> str:
        """查找对应的模板键"""
        error_message = str(error).lower()

        # 根据错误消息内容查找模板
        if "size" in error_message and "exceed" in error_message:
            return "size_exceeded"
        elif "format" in error_message and "support" in error_message:
            return "invalid_format"
        elif "encoding" in error_message:
            return "encoding_error"
        elif "column" in error_message or "missing" in error_message:
            return "missing_columns"
        elif "memory" in error_message:
            return "memory_limit"
        elif "timeout" in error_message:
            return "processing_timeout"
        elif "converge" in error_message:
            return "tsne_convergence"
        elif "parameter" in error_message:
            return "invalid_parameters"
        elif "permission" in error_message:
            return "file_permission"
        elif "connection" in error_message:
            return "connection_error"
        else:
            # 根据领域返回默认模板
            default_templates = {
                ErrorDomain.FILE_UPLOAD: "invalid_format",
                ErrorDomain.DATA_VALIDATION: "data_quality_issues",
                ErrorDomain.DATA_PROCESSING: "algorithm_failure",
                ErrorDomain.ALGORITHM: "invalid_parameters",
                ErrorDomain.VISUALIZATION: "render_error",
                ErrorDomain.EXPORT: "export_failed",
                ErrorDomain.SYSTEM: "server_error",
                ErrorDomain.NETWORK: "connection_error",
            }
            return default_templates.get(domain, "server_error")

    def _get_template(self, domain: ErrorDomain, template_key: str) -> Dict[str, Any]:
        """获取消息模板"""
        if (
            domain in self.message_templates
            and template_key in self.message_templates[domain]
        ):
            return self.message_templates[domain][template_key]
        else:
            # 返回默认模板
            return {
                "title": "未知错误",
                "description": "发生了未知错误，请稍后重试。",
                "suggestions": ["刷新页面", "联系技术支持"],
            }

    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """确定错误严重程度"""
        error_type = type(error).__name__

        if error_type in ["MemoryError", "SystemError"]:
            return ErrorSeverity.CRITICAL
        elif error_type in ["ValueError", "KeyError", "TimeoutError"]:
            return ErrorSeverity.ERROR
        elif error_type in ["FileNotFoundError", "PermissionError"]:
            return ErrorSeverity.WARNING
        else:
            return ErrorSeverity.INFO

    def _log_message(self, message: UserMessage, context: Optional[Dict[str, Any]]):
        """记录消息到日志"""
        log_context = {
            "message_type": message.message_type.value,
            "domain": message.domain.value,
            "title": message.title,
            "error_code": message.error_code,
            "severity": message.severity.value,
        }

        if context:
            log_context.update(context)

        if message.message_type == MessageType.ERROR:
            self.logger.log_error(
                error_type=message.domain.value,
                error_details={
                    "title": message.title,
                    "description": message.description,
                    "technical_details": message.technical_details,
                },
                context=log_context,
            )
        elif message.message_type == MessageType.WARNING:
            self.logger.log_warning(message=message.description, context=log_context)
        else:
            self.logger.log_info(message=message.description, context=log_context)

    def _add_to_history(self, message: UserMessage):
        """添加消息到历史"""
        self.message_history.append(message)
        if len(self.message_history) > self.max_history_size:
            self.message_history.pop(0)

    def to_dict(self, message: UserMessage) -> Dict[str, Any]:
        """将消息转换为字典"""
        return {
            "type": message.message_type.value,
            "domain": message.domain.value,
            "title": message.title,
            "description": message.description,
            "technical_details": message.technical_details,
            "suggestions": message.suggestions,
            "error_code": message.error_code,
            "severity": message.severity.value,
            "timestamp": message.created_at if hasattr(message, "created_at") else None,
        }

    def to_response_format(self, message: UserMessage) -> Dict[str, Any]:
        """转换为API响应格式"""
        return {
            "success": message.message_type != MessageType.ERROR,
            "message": {
                "type": message.message_type.value,
                "title": message.title,
                "description": message.description,
                "suggestions": message.suggestions,
            },
            "error_code": message.error_code,
            "technical_details": (
                message.technical_details
                if message.message_type == MessageType.ERROR
                else None
            ),
        }

    def get_recent_messages(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的消息"""
        recent_messages = self.message_history[-limit:] if self.message_history else []
        return [self.to_dict(msg) for msg in recent_messages]

    def clear_history(self):
        """清空消息历史"""
        self.message_history.clear()


# 全局用户消息管理器实例
user_message_manager = UserMessageManager()


# 便捷函数
def create_user_message(
    error: Exception, context: Optional[Dict[str, Any]] = None
) -> UserMessage:
    """创建用户友好消息的便捷函数"""
    return user_message_manager.create_message(error, context)


def create_info_message(
    domain: ErrorDomain,
    title: str,
    description: str,
    suggestions: Optional[List[str]] = None,
) -> UserMessage:
    """创建信息消息的便捷函数"""
    return user_message_manager.create_info_message(
        domain, title, description, suggestions
    )


def create_warning_message(
    domain: ErrorDomain,
    title: str,
    description: str,
    suggestions: Optional[List[str]] = None,
) -> UserMessage:
    """创建警告消息的便捷函数"""
    return user_message_manager.create_warning_message(
        domain, title, description, suggestions
    )


def format_message_for_response(message: UserMessage) -> Dict[str, Any]:
    """格式化消息为响应的便捷函数"""
    return user_message_manager.to_response_format(message)


def get_recent_messages(limit: int = 10) -> List[Dict[str, Any]]:
    """获取最近消息的便捷函数"""
    return user_message_manager.get_recent_messages(limit)
