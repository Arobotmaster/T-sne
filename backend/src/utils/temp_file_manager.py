"""
临时文件管理模块

提供临时文件的创建、清理和管理功能，符合SDD Constitution原则：
- Library-First: 独立的临时文件管理库
- CLI Interface: 支持命令行清理
- Scientific Observability: 详细的操作日志
"""

import os
import tempfile
import shutil
import time
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from src.config.settings import settings
from src.config.logging_config import get_logger

logger = get_logger(__name__)


class TempFileManager:
    """临时文件管理器

    提供临时文件的创建、跟踪和自动清理功能
    """

    def __init__(self):
        """初始化临时文件管理器"""
        self.temp_dir = Path(settings.TEMP_DIRECTORY)
        self.upload_dir = Path(settings.UPLOAD_DIRECTORY)
        self.file_registry: Dict[str, Dict[str, Any]] = {}
        self.cleanup_thread: Optional[threading.Thread] = None
        self._running = False

        # 确保目录存在
        self._ensure_directories()

        # 启动后台清理线程
        self._start_cleanup_thread()

        logger.info(f"临时文件管理器初始化完成，临时目录: {self.temp_dir}")

    def _ensure_directories(self):
        """确保必要的目录存在"""
        try:
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            self.upload_dir.mkdir(parents=True, exist_ok=True)

            # 设置目录权限 (仅用户可读写)
            self.temp_dir.chmod(0o700)
            self.upload_dir.chmod(0o700)

            logger.info("临时文件目录已创建并设置权限")
        except Exception as e:
            logger.error(f"创建临时文件目录失败: {e}")
            raise

    def create_temp_file(self, prefix: str = "mof_", suffix: str = ".tmp") -> str:
        """创建临时文件

        Args:
            prefix: 文件名前缀
            suffix: 文件名后缀

        Returns:
            str: 临时文件路径
        """
        try:
            # 使用系统临时目录创建文件
            fd, temp_path = tempfile.mkstemp(
                prefix=prefix,
                suffix=suffix,
                dir=str(self.temp_dir)
            )

            # 关闭文件描述符，保留文件
            os.close(fd)

            # 注册文件
            file_id = Path(temp_path).name
            self.file_registry[file_id] = {
                "path": temp_path,
                "created_at": time.time(),
                "accessed_at": time.time(),
                "size": 0,
                "purpose": prefix.rstrip("_")
            }

            logger.info(f"创建临时文件: {temp_path}")
            return temp_path

        except Exception as e:
            logger.error(f"创建临时文件失败: {e}")
            raise

    def create_upload_file(self, original_filename: str, user_id: str = "anonymous") -> str:
        """创建上传文件

        Args:
            original_filename: 原始文件名
            user_id: 用户标识

        Returns:
            str: 上传文件路径
        """
        try:
            # 生成安全的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = self._sanitize_filename(original_filename)
            upload_filename = f"{timestamp}_{user_id}_{safe_filename}"

            upload_path = self.upload_dir / upload_filename

            # 注册文件
            file_id = upload_filename
            self.file_registry[file_id] = {
                "path": str(upload_path),
                "created_at": time.time(),
                "accessed_at": time.time(),
                "size": 0,
                "purpose": "upload",
                "original_filename": original_filename,
                "user_id": user_id
            }

            logger.info(f"创建上传文件: {upload_path}")
            return str(upload_path)

        except Exception as e:
            logger.error(f"创建上传文件失败: {e}")
            raise

    def _sanitize_filename(self, filename: str) -> str:
        """清理文件名，移除危险字符

        Args:
            filename: 原始文件名

        Returns:
            str: 安全的文件名
        """
        # 移除路径分隔符和其他危险字符
        safe_chars = []
        for char in filename:
            if char.isalnum() or char in '._- ':
                safe_chars.append(char)
            else:
                safe_chars.append('_')

        return ''.join(safe_chars).strip('_')

    def register_file(self, file_path: str, purpose: str = "unknown", **metadata) -> str:
        """注册现有文件到管理系统

        Args:
            file_path: 文件路径
            purpose: 文件用途
            **metadata: 额外的元数据

        Returns:
            str: 文件ID
        """
        try:
            file_id = Path(file_path).name
            self.file_registry[file_id] = {
                "path": file_path,
                "created_at": time.time(),
                "accessed_at": time.time(),
                "size": Path(file_path).stat().st_size if Path(file_path).exists() else 0,
                "purpose": purpose,
                **metadata
            }

            logger.info(f"注册文件: {file_path} (用途: {purpose})")
            return file_id

        except Exception as e:
            logger.error(f"注册文件失败: {e}")
            raise

    def access_file(self, file_id: str) -> Optional[str]:
        """访问文件，更新访问时间

        Args:
            file_id: 文件ID

        Returns:
            Optional[str]: 文件路径，如果不存在返回None
        """
        if file_id in self.file_registry:
            file_info = self.file_registry[file_id]
            file_path = file_info["path"]

            if Path(file_path).exists():
                # 更新访问时间和大小
                file_info["accessed_at"] = time.time()
                file_info["size"] = Path(file_path).stat().st_size
                return file_path
            else:
                # 文件不存在，从注册表中移除
                del self.file_registry[file_id]
                logger.warning(f"文件不存在，从注册表移除: {file_id}")

        return None

    def delete_file(self, file_id: str) -> bool:
        """删除文件

        Args:
            file_id: 文件ID

        Returns:
            bool: 删除是否成功
        """
        try:
            if file_id in self.file_registry:
                file_info = self.file_registry[file_id]
                file_path = file_info["path"]

                if Path(file_path).exists():
                    Path(file_path).unlink()
                    logger.info(f"删除文件: {file_path}")

                # 从注册表中移除
                del self.file_registry[file_id]
                return True

            return False

        except Exception as e:
            logger.error(f"删除文件失败: {e}")
            return False

    def _start_cleanup_thread(self):
        """启动后台清理线程"""
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
            self._running = True
            self.cleanup_thread = threading.Thread(
                target=self._cleanup_worker,
                daemon=True,
                name="TempFileCleanup"
            )
            self.cleanup_thread.start()
            logger.info("后台清理线程已启动")

    def _cleanup_worker(self):
        """后台清理工作线程"""
        logger.info("临时文件清理工作线程启动")

        while self._running:
            try:
                # 每5分钟执行一次清理
                time.sleep(300)
                self._cleanup_expired_files()

            except Exception as e:
                logger.error(f"清理线程异常: {e}")

        logger.info("临时文件清理工作线程停止")

    def _cleanup_expired_files(self):
        """清理过期文件"""
        try:
            current_time = time.time()
            expired_files = []

            # 查找过期文件
            for file_id, file_info in self.file_registry.items():
                file_age = current_time - file_info["accessed_at"]

                # 根据文件类型设置不同的过期时间
                if file_info["purpose"] == "upload":
                    max_age = 24 * 60 * 60  # 24小时
                else:
                    max_age = 2 * 60 * 60   # 2小时

                if file_age > max_age:
                    expired_files.append(file_id)

            # 清理过期文件
            cleaned_count = 0
            for file_id in expired_files:
                if self.delete_file(file_id):
                    cleaned_count += 1

            if cleaned_count > 0:
                logger.info(f"清理过期文件: {cleaned_count} 个")

        except Exception as e:
            logger.error(f"清理过期文件失败: {e}")

    def cleanup_all_temp_files(self) -> int:
        """清理所有临时文件（强制清理）

        Returns:
            int: 清理的文件数量
        """
        try:
            cleaned_count = 0

            # 复制文件ID列表以避免在迭代时修改字典
            file_ids = list(self.file_registry.keys())

            for file_id in file_ids:
                file_info = self.file_registry.get(file_id, {})
                if file_info.get("purpose") in ["temp", "unknown"]:
                    if self.delete_file(file_id):
                        cleaned_count += 1

            logger.info(f"强制清理临时文件: {cleaned_count} 个")
            return cleaned_count

        except Exception as e:
            logger.error(f"强制清理临时文件失败: {e}")
            return 0

    def get_status(self) -> Dict[str, Any]:
        """获取临时文件管理状态

        Returns:
            Dict[str, Any]: 状态信息
        """
        try:
            total_files = len(self.file_registry)
            total_size = sum(info.get("size", 0) for info in self.file_registry.values())

            # 按用途统计
            purpose_stats = {}
            for info in self.file_registry.values():
                purpose = info.get("purpose", "unknown")
                purpose_stats[purpose] = purpose_stats.get(purpose, 0) + 1

            return {
                "total_files": total_files,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "temp_directory": str(self.temp_dir),
                "upload_directory": str(self.upload_dir),
                "cleanup_thread_running": self.cleanup_thread and self.cleanup_thread.is_alive(),
                "purpose_distribution": purpose_stats,
                "oldest_file_age_seconds": self._get_oldest_file_age()
            }

        except Exception as e:
            logger.error(f"获取状态失败: {e}")
            return {"error": str(e)}

    def _get_oldest_file_age(self) -> Optional[float]:
        """获取最老文件的年龄"""
        if not self.file_registry:
            return None

        current_time = time.time()
        oldest_age = min(
            current_time - info["accessed_at"]
            for info in self.file_registry.values()
        )
        return oldest_age

    def stop(self):
        """停止临时文件管理器"""
        logger.info("停止临时文件管理器")
        self._running = False

        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)


# 全局实例
_temp_file_manager: Optional[TempFileManager] = None


def get_temp_file_manager() -> TempFileManager:
    """获取临时文件管理器全局实例"""
    global _temp_file_manager
    if _temp_file_manager is None:
        _temp_file_manager = TempFileManager()
    return _temp_file_manager


def cleanup_temp_files() -> int:
    """清理临时文件的便捷函数"""
    return get_temp_file_manager().cleanup_all_temp_files()


def get_temp_file_status() -> Dict[str, Any]:
    """获取临时文件状态的便捷函数"""
    return get_temp_file_manager().get_status()