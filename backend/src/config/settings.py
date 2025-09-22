"""
配置设置模块

提供统一的配置管理，支持环境变量覆盖和配置验证
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator


class Settings(BaseSettings):
    """
    应用配置类

    支持从环境变量和.env文件加载配置
    """

    # 模型配置
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='ignore'
    )

    # =============================================================================
    # 应用基础配置
    # =============================================================================
    APP_NAME: str = Field(default="MOF数据t-SNE可视化", env="APP_NAME")
    APP_VERSION: str = Field(default="1.0.0", env="APP_VERSION")
    DEBUG: bool = Field(default=False, env="DEBUG")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")

    # =============================================================================
    # 数据库配置
    # =============================================================================
    DATABASE_URL: str = Field(default="sqlite:///./data/mof_viz.db", env="DATABASE_URL")

    # =============================================================================
    # Redis配置
    # =============================================================================
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")

    # =============================================================================
    # 安全配置
    # =============================================================================
    SECRET_KEY: str = Field(default="your-secret-key-change-this-in-production", env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    ALGORITHM: str = Field(default="HS256", env="ALGORITHM")

    # =============================================================================
    # 日志配置
    # =============================================================================
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: str = Field(default="logs/app.log", env="LOG_FILE")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    LOG_DATE_FORMAT: str = Field(
        default="%Y-%m-%d %H:%M:%S",
        env="LOG_DATE_FORMAT"
    )

    # =============================================================================
    # 文件上传配置
    # =============================================================================
    MAX_UPLOAD_SIZE: int = Field(default=104857600, env="MAX_UPLOAD_SIZE")  # 100MB
    ALLOWED_EXTENSIONS: List[str] = Field(default=["csv"], env="ALLOWED_EXTENSIONS")
    UPLOAD_DIRECTORY: str = Field(default="data/uploads", env="UPLOAD_DIRECTORY")
    TEMP_DIRECTORY: str = Field(default="temp", env="TEMP_DIRECTORY")

    # =============================================================================
    # CORS配置
    # =============================================================================
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        env="ALLOWED_ORIGINS"
    )
    ALLOWED_METHODS: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        env="ALLOWED_METHODS"
    )
    ALLOWED_HEADERS: List[str] = Field(default=["*"], env="ALLOWED_HEADERS")

    # =============================================================================
    # API配置
    # =============================================================================
    API_V1_PREFIX: str = Field(default="/api", env="API_V1_PREFIX")
    API_DOCS_URL: str = Field(default="/docs", env="API_DOCS_URL")
    API_REDOC_URL: str = Field(default="/redoc", env="API_REDOC_URL")

    # =============================================================================
    # 处理配置
    # =============================================================================
    MAX_CONCURRENT_PROCESSES: int = Field(default=4, env="MAX_CONCURRENT_PROCESSES")
    PROCESSING_TIMEOUT: int = Field(default=3600, env="PROCESSING_TIMEOUT")  # 1 hour
    CLEANUP_INTERVAL_HOURS: int = Field(default=24, env="CLEANUP_INTERVAL_HOURS")

    # =============================================================================
    # 可视化配置
    # =============================================================================
    DEFAULT_PLOT_WIDTH: int = Field(default=1200, env="DEFAULT_PLOT_WIDTH")
    DEFAULT_PLOT_HEIGHT: int = Field(default=800, env="DEFAULT_PLOT_HEIGHT")
    DEFAULT_DPI: int = Field(default=300, env="DEFAULT_DPI")
    MAX_PLOT_SIZE: int = Field(default=4000, env="MAX_PLOT_SIZE")

    # =============================================================================
    # 导出配置
    # =============================================================================
    EXPORT_DIRECTORY: str = Field(default="data/exports", env="EXPORT_DIRECTORY")
    SUPPORTED_EXPORT_FORMATS: List[str] = Field(
        default=["png", "svg", "pdf"],
        env="SUPPORTED_EXPORT_FORMATS"
    )
    DEFAULT_EXPORT_FORMAT: str = Field(default="png", env="DEFAULT_EXPORT_FORMAT")

    # =============================================================================
    # 监控配置
    # =============================================================================
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(default=8090, env="METRICS_PORT")
    HEALTH_CHECK_INTERVAL: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")

    # =============================================================================
    # 开发环境特定配置
    # =============================================================================
    DEBUG_MODE: bool = Field(default=False, env="DEBUG_MODE")
    RELOAD: bool = Field(default=False, env="RELOAD")
    DEV_CORS_ENABLED: bool = Field(default=False, env="DEV_CORS_ENABLED")

    # =============================================================================
    # 验证器
    # =============================================================================
    @validator('LOG_LEVEL')
    def validate_log_level(cls, v):
        """验证日志级别"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'LOG_LEVEL必须是以下值之一: {valid_levels}')
        return v.upper()

    @validator('ALLOWED_EXTENSIONS')
    def validate_extensions(cls, v):
        """验证文件扩展名"""
        if not isinstance(v, list):
            raise ValueError('ALLOWED_EXTENSIONS必须是列表')
        return [ext.lower().lstrip('.') for ext in v]

    @validator('ENVIRONMENT')
    def validate_environment(cls, v):
        """验证环境类型"""
        valid_envs = ['development', 'testing', 'staging', 'production']
        if v.lower() not in valid_envs:
            raise ValueError(f'ENVIRONMENT必须是以下值之一: {valid_envs}')
        return v.lower()

    @validator('MAX_UPLOAD_SIZE')
    def validate_upload_size(cls, v):
        """验证上传文件大小"""
        if v <= 0:
            raise ValueError('MAX_UPLOAD_SIZE必须大于0')
        if v > 1024 * 1024 * 1024:  # 1GB
            raise ValueError('MAX_UPLOAD_SIZE不能超过1GB')
        return v

    @validator('PROCESSING_TIMEOUT')
    def validate_timeout(cls, v):
        """验证处理超时时间"""
        if v <= 0:
            raise ValueError('PROCESSING_TIMEOUT必须大于0')
        if v > 24 * 3600:  # 24 hours
            raise ValueError('PROCESSING_TIMEOUT不能超过24小时')
        return v

    @validator('DEFAULT_DPI')
    def validate_dpi(cls, v):
        """验证DPI设置"""
        if v < 72 or v > 600:
            raise ValueError('DEFAULT_DPI必须在72-600之间')
        return v

    @validator('DEFAULT_PLOT_WIDTH', 'DEFAULT_PLOT_HEIGHT')
    def validate_plot_dimensions(cls, v):
        """验证图表尺寸"""
        if v < 400 or v > 4000:
            raise ValueError('图表尺寸必须在400-4000像素之间')
        return v

    # =============================================================================
    # 属性方法
    # =============================================================================
    @property
    def is_development(self) -> bool:
        """是否为开发环境"""
        return self.ENVIRONMENT == 'development'

    @property
    def is_production(self) -> bool:
        """是否为生产环境"""
        return self.ENVIRONMENT == 'production'

    @property
    def is_testing(self) -> bool:
        """是否为测试环境"""
        return self.ENVIRONMENT == 'testing'

    @property
    def database_url_sync(self) -> str:
        """同步数据库连接URL"""
        if self.DATABASE_URL.startswith('sqlite'):
            return self.DATABASE_URL.replace('sqlite://', 'sqlite+pysqlite://')
        return self.DATABASE_URL

    @property
    def upload_path(self) -> Path:
        """上传目录路径"""
        return Path(self.UPLOAD_DIRECTORY)

    @property
    def export_path(self) -> Path:
        """导出目录路径"""
        return Path(self.EXPORT_DIRECTORY)

    @property
    def log_path(self) -> Path:
        """日志文件路径"""
        return Path(self.LOG_FILE)

    @property
    def temp_path(self) -> Path:
        """临时目录路径"""
        return Path(self.TEMP_DIRECTORY)

    # =============================================================================
    # 工具方法
    # =============================================================================
    def get_cors_origins(self) -> List[str]:
        """获取CORS允许的源列表"""
        origins = self.ALLOWED_ORIGINS.copy()

        if self.is_development and self.DEV_CORS_ENABLED:
            # 开发环境添加更多本地源
            origins.extend([
                "http://127.0.0.1:3000",
                "http://127.0.0.1:8000",
                "http://localhost:3001",
                "http://localhost:8001"
            ])

        return origins

    def get_logging_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "detailed": {
                    "format": self.LOG_FORMAT,
                    "datefmt": self.LOG_DATE_FORMAT
                },
                "simple": {
                    "format": "%(levelname)s - %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "DEBUG" if self.DEBUG else "INFO",
                    "formatter": "detailed",
                    "stream": "ext://sys.stdout"
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": self.LOG_LEVEL,
                    "formatter": "detailed",
                    "filename": self.LOG_FILE,
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5
                }
            },
            "loggers": {
                "": {
                    "handlers": ["console", "file"],
                    "level": self.LOG_LEVEL,
                    "propagate": False
                }
            }
        }

    def ensure_directories(self) -> None:
        """确保必要的目录存在"""
        directories = [
            self.upload_path,
            self.export_path,
            self.log_path.parent,
            self.temp_path
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_algorithm_defaults(self) -> Dict[str, Any]:
        """获取算法默认配置"""
        return {
            "pca": {
                "n_components": 50,
                "random_state": 42
            },
            "tsne": {
                "perplexity": 30,
                "n_components": 2,
                "random_state": 42,
                "learning_rate": 200,
                "n_iter": 1000
            },
            "preprocessing": {
                "missing_value_strategy": "mean",
                "scaling_method": "standard"
            }
        }


# 全局配置实例
settings = Settings()

# 确保必要的目录存在
settings.ensure_directories()
