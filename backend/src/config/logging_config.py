"""
日志配置模块

遵循SDD Constitution的Scientific Observability原则，
提供详细的科学计算日志记录。
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Dict, Any

def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> None:
    """
    设置日志配置

    Args:
        log_level: 日志级别
        log_dir: 日志目录
    """
    # 确保日志目录存在
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    LOGGING_CONFIG: Dict[str, Any] = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            },
            'simple': {
                'format': '%(levelname)s - %(message)s'
            },
            'scientific': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s | %(funcName)s:%(lineno)d'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'simple',
                'stream': sys.stdout
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': log_path / 'application.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'encoding': 'utf8'
            },
            'error_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'detailed',
                'filename': log_path / 'error.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'encoding': 'utf8'
            },
            'scientific_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'scientific',
                'filename': log_path / 'scientific.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 3,
                'encoding': 'utf8'
            }
        },
        'loggers': {
            '': {
                'handlers': ['console', 'file', 'error_file'],
                'level': log_level,
                'propagate': False
            },
            'scientific': {
                'handlers': ['scientific_file'],
                'level': 'DEBUG',
                'propagate': False
            }
        },
        'root': {
            'handlers': ['console', 'file'],
            'level': log_level
        }
    }

    logging.config.dictConfig(LOGGING_CONFIG)

def get_logger(name: str) -> logging.Logger:
    """
    获取日志记录器

    Args:
        name: 日志记录器名称

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    return logging.getLogger(name)

def log_scientific_event(logger: logging.Logger, event_type: str, data: Dict[str, Any]) -> None:
    """
    记录科学计算事件

    Args:
        logger: 日志记录器
        event_type: 事件类型
        data: 事件数据
    """
    scientific_logger = logging.getLogger('scientific')
    scientific_logger.info(f"EVENT: {event_type} | DATA: {data}")