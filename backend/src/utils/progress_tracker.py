"""
处理进度跟踪系统

遵循SDD Constitution的Scientific Observability原则，
提供详细的处理进度跟踪、状态管理和实时更新功能。
"""

import asyncio
import threading
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict

class ProgressStatus(Enum):
    """进度状态"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class StepType(Enum):
    """步骤类型"""
    PREPROCESSING = "preprocessing"
    ALGORITHM = "algorithm"
    POSTPROCESSING = "postprocessing"
    VALIDATION = "validation"
    EXPORT = "export"

@dataclass
class ProgressStep:
    """进度步骤"""
    step_id: str
    name: str
    step_type: StepType
    status: ProgressStatus
    progress: float  # 0.0 to 1.0
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    estimated_duration: Optional[float] = None
    error_message: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None

@dataclass
class ProgressTracker:
    """进度跟踪器"""
    tracker_id: str
    name: str
    description: str
    status: ProgressStatus
    overall_progress: float  # 0.0 to 1.0
    steps: List[ProgressStep]
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    estimated_total_duration: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class ProgressTrackingSystem:
    """进度跟踪系统"""

    def __init__(self, log_dir: str = "logs"):
        """
        初始化进度跟踪系统

        Args:
            log_dir: 日志目录
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # 进度跟踪器存储
        self.trackers: Dict[str, ProgressTracker] = {}
        self.lock = threading.Lock()

        # 回调函数
        self.progress_callbacks: List[Callable[[ProgressTracker], None]] = []
        self.step_callbacks: List[Callable[[ProgressTracker, ProgressStep], None]] = []

        # 自动清理
        self.cleanup_interval = 3600  # 1小时
        self.max_tracker_age = 86400  # 24小时

        # 启动清理线程
        self._start_cleanup_thread()

    def create_tracker(self, tracker_id: str, name: str, description: str,
                      steps: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> ProgressTracker:
        """
        创建进度跟踪器

        Args:
            tracker_id: 跟踪器ID
            name: 跟踪器名称
            description: 跟踪器描述
            steps: 步骤列表
            metadata: 元数据

        Returns:
            ProgressTracker: 创建的进度跟踪器
        """
        # 创建步骤对象
        progress_steps = []
        for step_data in steps:
            step = ProgressStep(
                step_id=step_data['step_id'],
                name=step_data['name'],
                step_type=StepType(step_data.get('step_type', 'algorithm')),
                status=ProgressStatus.PENDING,
                progress=0.0,
                estimated_duration=step_data.get('estimated_duration')
            )
            progress_steps.append(step)

        # 创建跟踪器
        tracker = ProgressTracker(
            tracker_id=tracker_id,
            name=name,
            description=description,
            status=ProgressStatus.PENDING,
            overall_progress=0.0,
            steps=progress_steps,
            metadata=metadata or {}
        )

        # 存储跟踪器
        with self.lock:
            self.trackers[tracker_id] = tracker

        # 记录创建事件
        self._log_tracker_event(tracker_id, "created", {
            "name": name,
            "description": description,
            "steps_count": len(steps)
        })

        return tracker

    def start_tracker(self, tracker_id: str) -> bool:
        """
        启动进度跟踪器

        Args:
            tracker_id: 跟踪器ID

        Returns:
            bool: 是否成功启动
        """
        with self.lock:
            tracker = self.trackers.get(tracker_id)
            if not tracker:
                return False

            if tracker.status != ProgressStatus.PENDING:
                return False

            tracker.status = ProgressStatus.RUNNING
            tracker.start_time = datetime.now().isoformat()

        # 通知回调
        self._notify_progress_callbacks(tracker)

        # 记录启动事件
        self._log_tracker_event(tracker_id, "started")

        return True

    def update_step_progress(self, tracker_id: str, step_id: str,
                           progress: float, status: Optional[ProgressStatus] = None,
                           error_message: Optional[str] = None,
                           **additional_info) -> bool:
        """
        更新步骤进度

        Args:
            tracker_id: 跟踪器ID
            step_id: 步骤ID
            progress: 进度值 (0.0 to 1.0)
            status: 状态
            error_message: 错误消息
            **additional_info: 额外信息

        Returns:
            bool: 是否成功更新
        """
        with self.lock:
            tracker = self.trackers.get(tracker_id)
            if not tracker:
                return False

            # 查找步骤
            step = None
            for s in tracker.steps:
                if s.step_id == step_id:
                    step = s
                    break

            if not step:
                return False

            # 更新步骤信息
            old_progress = step.progress
            step.progress = max(0.0, min(1.0, progress))

            if status:
                step.status = status

            if error_message:
                step.error_message = error_message

            if additional_info:
                if step.additional_info is None:
                    step.additional_info = {}
                step.additional_info.update(additional_info)

            # 更新步骤时间
            if step.status == ProgressStatus.RUNNING and step.start_time is None:
                step.start_time = datetime.now().isoformat()
            elif step.status in [ProgressStatus.COMPLETED, ProgressStatus.FAILED] and step.end_time is None:
                step.end_time = datetime.now().isoformat()

            # 重新计算总体进度
            self._recalculate_overall_progress(tracker)

        # 通知回调
        self._notify_step_callbacks(tracker, step)

        # 记录进度更新事件
        if abs(progress - old_progress) > 0.01:  # 避免频繁记录
            self._log_step_event(tracker_id, step_id, "progress_updated", {
                "progress": progress,
                "status": status.value if status else None,
                "error_message": error_message
            })

        return True

    def complete_step(self, tracker_id: str, step_id: str,
                     **additional_info) -> bool:
        """
        完成步骤

        Args:
            tracker_id: 跟踪器ID
            step_id: 步骤ID
            **additional_info: 额外信息

        Returns:
            bool: 是否成功完成
        """
        return self.update_step_progress(
            tracker_id, step_id, 1.0, ProgressStatus.COMPLETED,
            **additional_info
        )

    def fail_step(self, tracker_id: str, step_id: str,
                  error_message: str, **additional_info) -> bool:
        """
        标记步骤失败

        Args:
            tracker_id: 跟踪器ID
            step_id: 步骤ID
            error_message: 错误消息
            **additional_info: 额外信息

        Returns:
            bool: 是否成功标记
        """
        return self.update_step_progress(
            tracker_id, step_id, 0.0, ProgressStatus.FAILED,
            error_message=error_message, **additional_info
        )

    def pause_tracker(self, tracker_id: str) -> bool:
        """
        暂停跟踪器

        Args:
            tracker_id: 跟踪器ID

        Returns:
            bool: 是否成功暂停
        """
        with self.lock:
            tracker = self.trackers.get(tracker_id)
            if not tracker or tracker.status != ProgressStatus.RUNNING:
                return False

            tracker.status = ProgressStatus.PAUSED

        # 通知回调
        self._notify_progress_callbacks(tracker)

        # 记录暂停事件
        self._log_tracker_event(tracker_id, "paused")

        return True

    def resume_tracker(self, tracker_id: str) -> bool:
        """
        恢复跟踪器

        Args:
            tracker_id: 跟踪器ID

        Returns:
            bool: 是否成功恢复
        """
        with self.lock:
            tracker = self.trackers.get(tracker_id)
            if not tracker or tracker.status != ProgressStatus.PAUSED:
                return False

            tracker.status = ProgressStatus.RUNNING

        # 通知回调
        self._notify_progress_callbacks(tracker)

        # 记录恢复事件
        self._log_tracker_event(tracker_id, "resumed")

        return True

    def complete_tracker(self, tracker_id: str, **additional_info) -> bool:
        """
        完成跟踪器

        Args:
            tracker_id: 跟踪器ID
            **additional_info: 额外信息

        Returns:
            bool: 是否成功完成
        """
        with self.lock:
            tracker = self.trackers.get(tracker_id)
            if not tracker:
                return False

            # 确保所有步骤都已完成
            all_completed = all(
                step.status == ProgressStatus.COMPLETED for step in tracker.steps
            )

            if not all_completed:
                return False

            tracker.status = ProgressStatus.COMPLETED
            tracker.end_time = datetime.now().isoformat()
            tracker.overall_progress = 1.0

            if additional_info:
                if tracker.metadata is None:
                    tracker.metadata = {}
                tracker.metadata.update(additional_info)

        # 通知回调
        self._notify_progress_callbacks(tracker)

        # 记录完成事件
        self._log_tracker_event(tracker_id, "completed", {
            "duration": self._calculate_duration(tracker),
            "steps_completed": len(tracker.steps)
        })

        return True

    def fail_tracker(self, tracker_id: str, error_message: str,
                     **additional_info) -> bool:
        """
        标记跟踪器失败

        Args:
            tracker_id: 跟踪器ID
            error_message: 错误消息
            **additional_info: 额外信息

        Returns:
            bool: 是否成功标记
        """
        with self.lock:
            tracker = self.trackers.get(tracker_id)
            if not tracker:
                return False

            tracker.status = ProgressStatus.FAILED
            tracker.end_time = datetime.now().isoformat()

            if tracker.metadata is None:
                tracker.metadata = {}
            tracker.metadata['error_message'] = error_message
            tracker.metadata.update(additional_info)

        # 通知回调
        self._notify_progress_callbacks(tracker)

        # 记录失败事件
        self._log_tracker_event(tracker_id, "failed", {
            "error_message": error_message,
            "duration": self._calculate_duration(tracker)
        })

        return True

    def cancel_tracker(self, tracker_id: str) -> bool:
        """
        取消跟踪器

        Args:
            tracker_id: 跟踪器ID

        Returns:
            bool: 是否成功取消
        """
        with self.lock:
            tracker = self.trackers.get(tracker_id)
            if not tracker:
                return False

            tracker.status = ProgressStatus.CANCELLED
            tracker.end_time = datetime.now().isoformat()

        # 通知回调
        self._notify_progress_callbacks(tracker)

        # 记录取消事件
        self._log_tracker_event(tracker_id, "cancelled")

        return True

    def get_tracker(self, tracker_id: str) -> Optional[ProgressTracker]:
        """
        获取跟踪器

        Args:
            tracker_id: 跟踪器ID

        Returns:
            Optional[ProgressTracker]: 跟踪器对象
        """
        with self.lock:
            return self.trackers.get(tracker_id)

    def get_tracker_status(self, tracker_id: str) -> Optional[Dict[str, Any]]:
        """
        获取跟踪器状态

        Args:
            tracker_id: 跟踪器ID

        Returns:
            Optional[Dict[str, Any]]: 状态信息
        """
        tracker = self.get_tracker(tracker_id)
        if not tracker:
            return None

        return {
            "tracker_id": tracker.tracker_id,
            "name": tracker.name,
            "status": tracker.status.value,
            "overall_progress": tracker.overall_progress,
            "start_time": tracker.start_time,
            "end_time": tracker.end_time,
            "steps": [
                {
                    "step_id": step.step_id,
                    "name": step.name,
                    "status": step.status.value,
                    "progress": step.progress,
                    "error_message": step.error_message
                }
                for step in tracker.steps
            ]
        }

    def get_all_trackers(self) -> List[Dict[str, Any]]:
        """
        获取所有跟踪器

        Returns:
            List[Dict[str, Any]]: 跟踪器列表
        """
        with self.lock:
            return [
                {
                    "tracker_id": tracker.tracker_id,
                    "name": tracker.name,
                    "status": tracker.status.value,
                    "overall_progress": tracker.overall_progress,
                    "start_time": tracker.start_time,
                    "end_time": tracker.end_time
                }
                for tracker in self.trackers.values()
            ]

    def add_progress_callback(self, callback: Callable[[ProgressTracker], None]) -> None:
        """
        添加进度回调函数

        Args:
            callback: 回调函数
        """
        self.progress_callbacks.append(callback)

    def add_step_callback(self, callback: Callable[[ProgressTracker, ProgressStep], None]) -> None:
        """
        添加步骤回调函数

        Args:
            callback: 回调函数
        """
        self.step_callbacks.append(callback)

    def _recalculate_overall_progress(self, tracker: ProgressTracker) -> None:
        """
        重新计算总体进度

        Args:
            tracker: 跟踪器对象
        """
        if not tracker.steps:
            tracker.overall_progress = 0.0
            return

        # 根据步骤状态计算权重
        total_weight = 0
        weighted_progress = 0

        for step in tracker.steps:
            # 简单的等权重计算
            weight = 1.0
            total_weight += weight

            if step.status == ProgressStatus.COMPLETED:
                weighted_progress += weight
            elif step.status == ProgressStatus.RUNNING:
                weighted_progress += weight * step.progress
            elif step.status == ProgressStatus.FAILED:
                weighted_progress += weight * 0.0  # 失败步骤不贡献进度

        tracker.overall_progress = weighted_progress / total_weight if total_weight > 0 else 0.0

        # 更新跟踪器状态
        if tracker.overall_progress >= 1.0:
            tracker.status = ProgressStatus.COMPLETED

    def _calculate_duration(self, tracker: ProgressTracker) -> Optional[float]:
        """
        计算跟踪器持续时间

        Args:
            tracker: 跟踪器对象

        Returns:
            Optional[float]: 持续时间（秒）
        """
        if not tracker.start_time:
            return None

        start_time = datetime.fromisoformat(tracker.start_time)
        end_time = datetime.fromisoformat(tracker.end_time) if tracker.end_time else datetime.now()

        return (end_time - start_time).total_seconds()

    def _notify_progress_callbacks(self, tracker: ProgressTracker) -> None:
        """通知进度回调函数"""
        for callback in self.progress_callbacks:
            try:
                callback(tracker)
            except Exception as e:
                print(f"Error in progress callback: {e}")

    def _notify_step_callbacks(self, tracker: ProgressTracker, step: ProgressStep) -> None:
        """通知步骤回调函数"""
        for callback in self.step_callbacks:
            try:
                callback(tracker, step)
            except Exception as e:
                print(f"Error in step callback: {e}")

    def _log_tracker_event(self, tracker_id: str, event_type: str,
                          additional_info: Optional[Dict[str, Any]] = None) -> None:
        """记录跟踪器事件"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "tracker_id": tracker_id,
            "event_type": event_type,
            "additional_info": additional_info
        }

        log_file = self.log_dir / "progress_events.log"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

    def _log_step_event(self, tracker_id: str, step_id: str, event_type: str,
                       additional_info: Optional[Dict[str, Any]] = None) -> None:
        """记录步骤事件"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "tracker_id": tracker_id,
            "step_id": step_id,
            "event_type": event_type,
            "additional_info": additional_info
        }

        log_file = self.log_dir / "step_events.log"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

    def _start_cleanup_thread(self) -> None:
        """启动清理线程"""
        def cleanup_worker():
            while True:
                time.sleep(self.cleanup_interval)
                self._cleanup_old_trackers()

        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()

    def _cleanup_old_trackers(self) -> None:
        """清理旧的跟踪器"""
        cutoff_time = datetime.now() - timedelta(seconds=self.max_tracker_age)

        with self.lock:
            expired_trackers = []
            for tracker_id, tracker in self.trackers.items():
                if tracker.start_time:
                    start_time = datetime.fromisoformat(tracker.start_time)
                    if start_time < cutoff_time:
                        expired_trackers.append(tracker_id)

            for tracker_id in expired_trackers:
                del self.trackers[tracker_id]

        if expired_trackers:
            self._log_tracker_event("system", "cleanup", {
                "expired_trackers": expired_trackers,
                "count": len(expired_trackers)
            })

    @contextmanager
    def progress_context(self, tracker_id: str, step_id: str):
        """
        进度上下文管理器

        Args:
            tracker_id: 跟踪器ID
            step_id: 步骤ID

        Yields:
            None
        """
        try:
            self.update_step_progress(tracker_id, step_id, 0.0, ProgressStatus.RUNNING)
            yield
        except Exception as e:
            self.fail_step(tracker_id, step_id, str(e))
            raise
        else:
            self.complete_step(tracker_id, step_id)

    def export_progress_report(self, output_file: Optional[str] = None) -> str:
        """
        导出进度报告

        Args:
            output_file: 输出文件路径

        Returns:
            str: 导出文件路径
        """
        if output_file is None:
            output_file = self.log_dir / f"progress_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with self.lock:
            report = {
                "export_time": datetime.now().isoformat(),
                "trackers": [asdict(tracker) for tracker in self.trackers.values()],
                "summary": {
                    "total_trackers": len(self.trackers),
                    "active_trackers": len([t for t in self.trackers.values() if t.status == ProgressStatus.RUNNING]),
                    "completed_trackers": len([t for t in self.trackers.values() if t.status == ProgressStatus.COMPLETED]),
                    "failed_trackers": len([t for t in self.trackers.values() if t.status == ProgressStatus.FAILED])
                }
            }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        return str(output_file)


# 全局进度跟踪系统实例
progress_tracking_system = ProgressTrackingSystem()


def get_progress_tracking_system() -> ProgressTrackingSystem:
    """获取全局进度跟踪系统实例"""
    return progress_tracking_system


# 便捷的装饰器
def track_progress(tracker_id: str, step_id: str):
    """
    进度跟踪装饰器

    Args:
        tracker_id: 跟踪器ID
        step_id: 步骤ID
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with progress_tracking_system.progress_context(tracker_id, step_id):
                return func(*args, **kwargs)
        return wrapper
    return decorator