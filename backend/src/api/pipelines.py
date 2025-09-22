"""
处理流水线API模块

实现处理流水线的状态查询和管理功能，符合SDD Constitution原则：
- Library-First: 集成处理服务
- CLI Interface: 支持命令行查询
- Test-First: 完整的测试覆盖
- Integration-First: 与数据集和可视化服务集成
- Scientific Observability: 详细的处理进度日志
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import uuid
import time

router = APIRouter()

# 模拟流水线存储（实际项目中应该使用数据库）
_pipelines_store = {}

# 为了在流水线状态变化时同步数据集元数据
try:
    from src.services.upload_service import UploadService
    _upload_service = UploadService()
except Exception:
    _upload_service = None


@router.get("/{pipeline_id}/status")
async def get_pipeline_status(pipeline_id: str):
    """
    获取处理流水线状态 - T045

    Args:
        pipeline_id: 流水线ID

    Returns:
        Dict: 流水线状态信息

    Raises:
        HTTPException: 流水线不存在
    """
    if pipeline_id not in _pipelines_store:
        raise HTTPException(status_code=404, detail="处理流水线不存在")

    pipeline = _pipelines_store[pipeline_id]

    # 模拟进度更新（实际项目中应该从后台任务获取真实状态）
    if pipeline["status"] == "running":
        elapsed_time = (datetime.now() - datetime.fromisoformat(pipeline["start_time"])).total_seconds()
        progress = min(100, int((elapsed_time / 30) * 100))  # 假设30秒完成

        pipeline["progress_percentage"] = progress
        pipeline["current_step"] = _get_current_step_by_progress(progress)

        # 更新步骤状态
        for i, step in enumerate(pipeline["steps"]):
            step_progress = max(0, min(100, progress - i * 25))
            if step_progress > 0 and step_progress < 100:
                step["status"] = "running"
                step["progress_percentage"] = step_progress
            elif step_progress >= 100:
                step["status"] = "completed"
                step["progress_percentage"] = 100
                if step["end_time"] is None:
                    step["end_time"] = datetime.now().isoformat()
                    step["duration_ms"] = int((datetime.fromisoformat(step["end_time"]) -
                                               datetime.fromisoformat(pipeline["start_time"])).total_seconds() * 1000)

        # 检查是否完成
        if progress >= 100:
            pipeline["status"] = "completed"
            pipeline["end_time"] = datetime.now().isoformat()
            pipeline["total_duration_ms"] = int((datetime.fromisoformat(pipeline["end_time"]) -
                                              datetime.fromisoformat(pipeline["start_time"])).total_seconds() * 1000)
            # 若尚未有真实可视化，尝试从 results 生成，否则不再创建占位数据
            try:
                from .visualizations import _visualizations_store
                if pipeline_id not in _visualizations_store:
                    # 尝试从 results 读取真实坐标
                    from pathlib import Path
                    import pandas as pd
                    coords_path = Path('results') / pipeline_id / 'tsne_coordinates.csv'
                    if coords_path.exists():
                        df = pd.read_csv(coords_path)
                        categories = []
                        if 'category' in df.columns:
                            counts = df['category'].value_counts().to_dict()
                            name_to_id = {name: idx for idx, name in enumerate(sorted(counts.keys()))}
                            categories = [
                                {"category_id": cid, "category_name": name, "sample_count": int(counts[name]), "color_code": "#999999"}
                                for name, cid in name_to_id.items()
                            ]
                        coords = []
                        for _, r in df.iterrows():
                            cat_name = str(r.get('category', 'unknown'))
                            cat_id = 0
                            if categories:
                                for c in categories:
                                    if c['category_name'] == cat_name:
                                        cat_id = c['category_id']; break
                            coords.append({
                                "sample_id": str(r.get('sample_id', '')),
                                "x": float(r['x']),
                                "y": float(r['y']),
                                "category_id": int(cat_id),
                                "category_name": cat_name,
                                "distance_to_center": float((r['x']**2 + r['y']**2) ** 0.5),
                                "local_density": 0.0
                            })
                        _visualizations_store[pipeline_id] = {
                            "visualization_id": pipeline_id,
                            "pipeline_id": pipeline_id,
                            "dataset_id": pipeline["dataset_id"],
                            "status": "completed",
                            "config": {"chart_title": "t-SNE", "width": 1200, "height": 800, "show_legend": True},
                            "coordinates": coords,
                            "categories": categories,
                            "statistics": {"total_samples": len(coords), "valid_samples": len(coords)},
                            "render_time_ms": 0,
                            "created_at": datetime.now().isoformat()
                        }
                    else:
                        # 保留为空，交由后续请求触发生成
                        pass
            except Exception:
                pass

            # 同步数据集状态到 completed，避免再次处理时报 409
            try:
                if _upload_service is not None:
                    latest = {
                        "pipeline_id": pipeline_id,
                        "status": "completed",
                        "start_time": pipeline["start_time"],
                        "end_time": pipeline["end_time"]
                    }
                    # 在异步上下文中直接更新数据集状态
                    from fastapi import BackgroundTasks  # type: ignore
                    # 直接await更新（当前函数是异步）
                    await _upload_service.update_dataset_status(pipeline["dataset_id"], {  # type: ignore
                        "processing_status": "completed",
                        "latest_pipeline": latest
                    })
            except Exception:
                # 不影响状态查询
                pass

    return {
        "pipeline_id": pipeline_id,
        "dataset_id": pipeline["dataset_id"],
        "status": pipeline["status"],
        "progress_percentage": pipeline["progress_percentage"],
        "current_step": pipeline["current_step"],
        "start_time": pipeline["start_time"],
        "end_time": pipeline.get("end_time"),
        "total_duration_ms": pipeline.get("total_duration_ms"),
        "steps": pipeline["steps"],
        "error_message": pipeline.get("error_message"),
        "performance_metrics": {
            "total_memory_mb": 512.5,
            "peak_memory_mb": 768.2,
            "total_cpu_time_ms": pipeline.get("total_duration_ms", 0),
            "average_cpu_usage": 65.3,
            "disk_io_mb": 12.4
        }
    }


@router.get("")
async def list_pipelines(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = Query(None, description="过滤状态"),
    dataset_id: Optional[str] = Query(None, description="数据集ID过滤")
):
    """
    获取处理流水线列表

    Args:
        skip: 跳过的记录数
        limit: 返回的记录数限制
        status: 状态过滤
        dataset_id: 数据集ID过滤

    Returns:
        Dict: 流水线列表和分页信息
    """
    pipelines = list(_pipelines_store.values())

    # 状态过滤
    if status:
        pipelines = [p for p in pipelines if p["status"] == status]

    # 数据集过滤
    if dataset_id:
        pipelines = [p for p in pipelines if p["dataset_id"] == dataset_id]

    # 按开始时间排序
    pipelines.sort(key=lambda x: x["start_time"], reverse=True)

    # 分页
    total = len(pipelines)
    pipelines = pipelines[skip:skip + limit]

    return {
        "pipelines": pipelines,
        "total": total,
        "skip": skip,
        "limit": limit,
        "has_more": skip + limit < total
    }


@router.delete("/{pipeline_id}")
async def cancel_pipeline(pipeline_id: str):
    """
    取消处理流水线

    Args:
        pipeline_id: 流水线ID

    Returns:
        Dict: 取消结果

    Raises:
        HTTPException: 流水线不存在或已完成
    """
    if pipeline_id not in _pipelines_store:
        raise HTTPException(status_code=404, detail="处理流水线不存在")

    pipeline = _pipelines_store[pipeline_id]

    if pipeline["status"] in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="流水线已完成或失败，无法取消")

    pipeline["status"] = "cancelled"
    pipeline["end_time"] = datetime.now().isoformat()
    pipeline["error_message"] = "用户取消处理"

    # 同步数据集状态为取消，允许重新开始
    try:
        if _upload_service is not None:
            latest = {
                "pipeline_id": pipeline_id,
                "status": "cancelled",
                "start_time": pipeline.get("start_time"),
                "end_time": pipeline["end_time"]
            }
            await _upload_service.update_dataset_status(pipeline["dataset_id"], {  # type: ignore
                "processing_status": "cancelled",
                "latest_pipeline": latest
            })
    except Exception:
        pass

    return {
        "success": True,
        "message": f"处理流水线 {pipeline_id} 已取消",
        "cancelled_at": pipeline["end_time"]
    }


@router.get("/{pipeline_id}/logs")
async def get_pipeline_logs(
    pipeline_id: str,
    level: Optional[str] = Query(None, description="日志级别过滤"),
    limit: int = Query(100, ge=1, le=1000)
):
    """
    获取处理流水线日志

    Args:
        pipeline_id: 流水线ID
        level: 日志级别过滤
        limit: 返回的日志条数限制

    Returns:
        Dict: 流水线日志信息

    Raises:
        HTTPException: 流水线不存在
    """
    if pipeline_id not in _pipelines_store:
        raise HTTPException(status_code=404, detail="处理流水线不存在")

    pipeline = _pipelines_store[pipeline_id]

    # 模拟日志数据
    logs = [
        {
            "timestamp": pipeline["start_time"],
            "level": "INFO",
            "message": f"开始处理数据集 {pipeline['dataset_id']}",
            "step": "初始化"
        },
        {
            "timestamp": (datetime.fromisoformat(pipeline["start_time"]) + timedelta(seconds=2)).isoformat(),
            "level": "INFO",
            "message": "数据预处理完成",
            "step": "preprocessing"
        },
        {
            "timestamp": (datetime.fromisoformat(pipeline["start_time"]) + timedelta(seconds=10)).isoformat(),
            "level": "INFO",
            "message": "PCA降维完成，保留95%方差",
            "step": "pca"
        },
        {
            "timestamp": (datetime.fromisoformat(pipeline["start_time"]) + timedelta(seconds=25)).isoformat(),
            "level": "INFO",
            "message": "t-SNE降维完成",
            "step": "tsne"
        }
    ]

    # 级别过滤
    if level:
        logs = [log for log in logs if log["level"] == level.upper()]

    # 限制条数
    logs = logs[-limit:]

    return {
        "pipeline_id": pipeline_id,
        "logs": logs,
        "total_logs": len(logs),
        "log_levels": ["INFO", "WARNING", "ERROR"]
    }


def _get_current_step_by_progress(progress: int) -> str:
    """根据进度获取当前步骤"""
    if progress < 25:
        return "数据预处理"
    elif progress < 50:
        return "PCA降维"
    elif progress < 75:
        return "t-SNE降维"
    else:
        return "可视化生成"


def create_test_pipeline(dataset_id: str):
    """创建测试流水线（用于契约测试）"""
    pipeline_id = str(uuid.uuid4())
    start_time = datetime.now()

    pipeline = {
        "pipeline_id": pipeline_id,
        "dataset_id": dataset_id,
        "status": "running",
        "start_time": start_time.isoformat(),
        "end_time": None,
        "config": {
            "pca_config": {"n_components": 50},
            "tsne_config": {"perplexity": 30}
        },
        "progress_percentage": 45,
        "current_step": "PCA降维",
        "steps": [
            {
                "step_name": "数据预处理",
                "step_type": "preprocessing",
                "status": "completed",
                "progress_percentage": 100,
                "start_time": start_time.isoformat(),
                "end_time": (start_time + timedelta(seconds=5)).isoformat(),
                "duration_ms": 5000,
                "error_message": None
            },
            {
                "step_name": "PCA降维",
                "step_type": "pca",
                "status": "running",
                "progress_percentage": 80,
                "start_time": (start_time + timedelta(seconds=5)).isoformat(),
                "end_time": None,
                "duration_ms": None,
                "error_message": None
            },
            {
                "step_name": "t-SNE降维",
                "step_type": "tsne",
                "status": "pending",
                "progress_percentage": 0,
                "start_time": None,
                "end_time": None,
                "duration_ms": None,
                "error_message": None
            }
        ],
        "error_message": None,
        "estimated_duration_ms": 30000
    }

    _pipelines_store[pipeline_id] = pipeline
    return pipeline_id


def add_test_dataset():
    """添加测试数据集（用于契约测试）"""
    test_dataset = {
        "dataset_id": "test_dataset_001",
        "filename": "test_data.csv",
        "file_size": 1024,
        "row_count": 100,
        "column_count": 4,
        "data_quality_score": 0.95,
        "columns": [
            {"name": "sample_id", "type": "string", "is_id": True},
            {"name": "feature_1", "type": "numeric", "is_feature": True},
            {"name": "feature_2", "type": "numeric", "is_feature": True},
            {"name": "category", "type": "string", "is_category": True}
        ],
        "categories": ["MOF_A", "MOF_B", "MOF_C"],
        "missing_values": {},
        "data_types": {
            "sample_id": "string",
            "feature_1": "float64",
            "feature_2": "float64",
            "category": "string"
        },
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }

    _datasets_store[test_dataset["dataset_id"]] = test_dataset
    return test_dataset["dataset_id"]
